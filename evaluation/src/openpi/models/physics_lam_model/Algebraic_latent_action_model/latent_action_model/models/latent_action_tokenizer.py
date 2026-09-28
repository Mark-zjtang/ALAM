from typing import Dict, List
import lpips
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from einops import rearrange, repeat
from openpi.models.physics_lam_model.uni_world_model.latent_action_model.modules.blocks import patchify, unpatchify, SpatioTemporalTransformer, SpatioTransformer, \
                                                     MVSpatioTemporalTransformer, MVSpatioTransformer
from openpi.models.physics_lam_model.uni_world_model.latent_action_model.modules.vector_quantizer import VectorQuantizer, VectorQuantizer2
from IPython import embed


class LatentActionTokenizer(nn.Module):
    """
    Latent action VQ-VAE.
    """

    def __init__(
            self,
            in_dim: int,
            model_dim: int,
            latent_dim: int,
            num_latents: int,
            patch_size: int,
            enc_blocks: int,
            dec_blocks: int,
            num_heads: int,
            action_num_codes: int,
            loss_config,
            dropout: float = 0.0,
    ) -> None:
        super(LatentActionTokenizer, self).__init__()
        self.latent_dim = latent_dim
        self.patch_size = patch_size
        patch_token_dim = in_dim * patch_size ** 2

        self.num_codes = action_num_codes
        self.action_latent = nn.Parameter(torch.empty(1, 1, self.num_codes, patch_token_dim))    # TODO: num of codes
        nn.init.uniform_(self.action_latent, a=-1, b=1)
        self.encoder = SpatioTemporalTransformer(
            in_dim=patch_token_dim,
            model_dim=model_dim,
            out_dim=latent_dim,
            num_blocks=enc_blocks,
            num_heads=num_heads,
            dropout=dropout,
            causal_temporal=True,
            to_out=False,
        )

        self.to_codebook = nn.Linear(model_dim, latent_dim)
        # self.vector_quantizer = VectorQuantizer(
        #     num_latents=num_latents,
        #     latent_dim=latent_dim,
        #     code_restart=True,
        # )

        self.vector_quantizer = VectorQuantizer2(
            num_latents=num_latents,
            latent_dim=latent_dim,
            beta=0.25,
            remap=None,
            sane_index_shape=True,
        )
        ## Decoder: Spatial Transformer
        self.patch_up = nn.Linear(patch_token_dim, model_dim)
        self.action_up = nn.Linear(latent_dim, model_dim)
        self.decoder = SpatioTransformer(
            in_dim=model_dim,
            model_dim=model_dim,
            out_dim=patch_token_dim,       
            num_blocks=dec_blocks,
            num_heads=num_heads,
            dropout=dropout,
        )
        self.loss_config = loss_config

        import os
        os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
        os.environ["TORCH_HOME"] = "/mnt/workspace/czj/models"
        self.loss_fn_lpips = lpips.LPIPS(net='vgg').requires_grad_(False).eval()


    def vq_encode(self, videos: Tensor, attention_mask: Tensor = None) -> Dict:
        # Preprocess videos
        B, T = videos.shape[:2]
        patches = patchify(videos, self.patch_size)

        action_pad = self.action_latent.expand(B, T, -1, -1)
        padded_patches = torch.cat([action_pad, patches], dim=2)

        # Encode
        z = self.encoder(padded_patches, attention_mask) 

        # Get latent action for all future frames
        z = self.to_codebook(z[:, 1:, :self.num_codes])  # (B, T-1, n, E)
        latent_action = z
        # Vector quantize
        z = z.reshape(B * (T - 1), self.num_codes, self.latent_dim)
        z_q, indices, commit_loss  = self.vector_quantizer(z)
        z_q = z_q.reshape(B, T - 1, self.num_codes, self.latent_dim)
        return {
            "patches": patches,
            "z_q": z_q,
            "indices": indices,
            "commit_loss": commit_loss,
            "latent_action": latent_action,
            "physical_latent_action": latent_action
        }
    
    def decode_additivity_lam(self, a_1, a_2) -> Dict:
        latent_action = a_1 + a_2  # (B, T-1, n, E)
        z_action = latent_action
        B, T = z_action.shape[:2]
        print("z_action shape in decode_additivity_lam:", z_action.shape)
        # Vector quantize
        z_flattened = z_action.reshape(-1, self.num_codes, self.latent_dim)
        z_q, indices, commit_loss  = self.vector_quantizer(z_flattened)
        z_q = z_q.reshape(B, T, self.num_codes, self.latent_dim)
        return {
            "z_q": z_q,
            "indices": indices,
            "latent_action": latent_action,
        }

    def forward(self, cond_pixel_values, target_pixel_values, return_action_token_ids_only=False, return_recons_only=False) -> Dict:
        video = torch.stack([cond_pixel_values, target_pixel_values], dim=1)
        # Encode + VQ
        B, T = video.shape[:2]
        H, W = video.shape[3:5]

        outputs = self.vq_encode(video) 
        video_patches = self.patch_up(outputs["patches"][:, :-1])
        action_patches = self.action_up(outputs["z_q"])
        video_action_patches = torch.cat([action_patches, video_patches], dim=2)

        # Decode
        video_recon = self.decoder(video_action_patches)
        video_recon = video_recon[:, :, self.num_codes: self.num_codes + video_patches.shape[2]] 
        video_recon = F.sigmoid(video_recon)

        target_video_recon = unpatchify(video_recon, self.patch_size, H, W)
        recon_pixel_values = target_video_recon.squeeze(1)

        if return_action_token_ids_only:
            return outputs['indices']
        
        if return_recons_only:
            return {
                "recon_pixel_values": recon_pixel_values,
                "indices": outputs['indices'],
                "latent_action": outputs['latent_action']
            }
        
        # Compute loss
        if self.loss_config.use_abs_recons_loss:
            recons_loss = torch.abs(recon_pixel_values - target_pixel_values).mean()
        else:
            recons_loss = F.mse_loss(recon_pixel_values, target_pixel_values)

        if self.loss_config.perceptual_loss_w > 0:
            with torch.no_grad():
                perceptual_loss = self.loss_fn_lpips.forward(
                    recon_pixel_values, target_pixel_values, normalize=True).mean()
        else:
            perceptual_loss = torch.zeros_like(recons_loss)

        commit_loss = outputs['commit_loss']
        loss =  self.loss_config.commit_loss_w * commit_loss + self.loss_config.recon_loss_w * recons_loss + \
                self.loss_config.perceptual_loss_w * perceptual_loss
        
        # active_code_num = torch.tensor(len(set(indices.long().reshape(-1).cpu().numpy().tolist()))).float().to(loss.device)
        active_code_num = torch.tensor(torch.unique(outputs['indices']).shape[0]).float().to(loss.device)

        loss_outputs = dict()
        loss_outputs.update(
            {
                "active_code_num": active_code_num,
                "loss": loss,
                "commit_loss": commit_loss,
                "recons_loss": recons_loss,
                "perceptual_loss": perceptual_loss,
            }
        )
        return loss_outputs
    
    def decode_image(self, cond_pixel_values, given_action_token_ids):
        B, C, H, W = cond_pixel_values.shape
        videos = cond_pixel_values.reshape(B, 1, C, H, W) # (b, 1, c, h, w)
        patches = patchify(videos, self.patch_size)
        video_patches = self.patch_up(patches)
        z_q = self.vector_quantizer.codebook(given_action_token_ids, shape=(B, self.num_codes, self.latent_dim))
        z_q = z_q.reshape(B, 1, self.num_codes, self.latent_dim)
        action_patches = self.action_up(z_q)
        video_action_patches = torch.cat([action_patches, video_patches], dim=2)

        # Decode
        video_recon = self.decoder(video_action_patches)
        video_recon = video_recon[:, :, self.num_codes: self.num_codes + video_patches.shape[2]] 
        video_recon = F.sigmoid(video_recon)

        target_video_recon = unpatchify(video_recon, self.patch_size, H, W)
        recon_pixel_values = target_video_recon.squeeze(1)

        return {
                "recon_pixel_values": recon_pixel_values,
            }


    @property
    def device(self):
        return next(self.parameters()).device

    def get_state_dict_to_save(self):
        modules_to_exclude = []
        state_dict = {k: v for k, v in self.state_dict().items() if
                      not any(module_name in k for module_name in modules_to_exclude)}
        return state_dict
