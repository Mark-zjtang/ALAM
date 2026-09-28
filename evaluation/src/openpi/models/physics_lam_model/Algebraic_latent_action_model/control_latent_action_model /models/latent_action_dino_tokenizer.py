from typing import Dict, List
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torchvision import transforms
from einops import rearrange, repeat
from uni_world_model.latent_action_model.modules.blocks import patchify, unpatchify, SpatioTemporalTransformer, SpatioTransformer, \
                                                     MVSpatioTemporalTransformer, MVSpatioTransformer
from uni_world_model.latent_action_model.modules.vector_quantizer import VectorQuantizer, VectorQuantizer2
from IPython import embed

# Use timm's names
IMAGENET_DEFAULT_MEAN = (0.485, 0.456, 0.406)
IMAGENET_DEFAULT_STD = (0.229, 0.224, 0.225)

class LatentActionDINOTokenizer(nn.Module):
    """
    Latent action VQ-VAE.
    """

    def __init__(
            self,
            dino_model,
            freeze_vision,
            dino_dim: int,
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
        super(LatentActionDINOTokenizer, self).__init__()
        self.latent_dim = latent_dim
        self.patch_size = patch_size

        self.dino_model = dino_model
        self.dino_transform = transforms.Normalize(mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD)
        self.freeze_vision = freeze_vision
        if freeze_vision:
            for _, param in self.dino_model.named_parameters():
                param.requires_grad = False


        self.num_codes = action_num_codes
        self.action_latent = nn.Parameter(torch.empty(1, 1, self.num_codes, dino_dim))    # TODO: num of codes
        nn.init.uniform_(self.action_latent, a=-1, b=1)
        self.encoder = SpatioTemporalTransformer(
            in_dim=dino_dim,
            model_dim=model_dim,
            out_dim=latent_dim,
            num_blocks=enc_blocks,
            num_heads=num_heads,
            dropout=dropout,
            causal_temporal=True,
            to_out=False,
        )

        self.to_codebook = nn.Linear(model_dim, latent_dim)

        self.vector_quantizer = VectorQuantizer2(
            num_latents=num_latents,
            latent_dim=latent_dim,
            beta=0.25,
            remap=None,
            sane_index_shape=True,
        )
        ## Decoder: Spatial Transformer
        self.patch_up = nn.Linear(dino_dim, model_dim)
        self.action_up = nn.Linear(latent_dim, model_dim)
        self.decoder = SpatioTransformer(
            in_dim=model_dim,
            model_dim=model_dim,
            out_dim=dino_dim,       
            num_blocks=dec_blocks,
            num_heads=num_heads,
            dropout=dropout,
        )
        self.loss_config = loss_config


    def vq_encode(self, video: Tensor, attention_mask: Tensor = None) -> Dict:

        dion_features = self.dino_model(video)  # (b, img_tokens, img_feat_dim)
        dion_features = rearrange(dion_features, "(b T) l d -> b T l d", T=2)

        # Preprocess videos
        B, T = dion_features.shape[:2]
        action_pad = self.action_latent.expand(B, T, -1, -1)
        padded_patches = torch.cat([action_pad, dion_features], dim=2)

        # Encode
        z = self.encoder(padded_patches, attention_mask) 

        # Get latent action for all future frames
        z = self.to_codebook(z[:, 1:, :self.num_codes])  # (B, T-1, n, E)

        # Vector quantize
        z = z.reshape(B * (T - 1), self.num_codes, self.latent_dim)
        z_q, indices, commit_loss  = self.vector_quantizer(z)
        z_q = z_q.reshape(B, T - 1, self.num_codes, self.latent_dim)
        return {
            "patches": dion_features,
            "z_q": z_q,
            "indices": indices,
            "commit_loss": commit_loss,
        }

    def forward(self, cond_pixel_values, target_pixel_values, return_action_token_ids_only=False, return_recons_only=False) -> Dict:
        video = torch.stack([cond_pixel_values, target_pixel_values], dim=1)
        # Encode + VQ
        video = rearrange(video, "b T c h w -> (b T) c h w")
        video = self.dino_transform(video)

        outputs = self.vq_encode(video) 

        video_patches = self.patch_up(outputs["patches"][:, :-1])
        action_patches = self.action_up(outputs["z_q"])
        video_action_patches = torch.cat([action_patches, video_patches], dim=2)

        # Decode
        video_recon = self.decoder(video_action_patches)
        video_recon = video_recon[:, :, self.num_codes: self.num_codes + video_patches.shape[2]] 

        if return_action_token_ids_only:
            return outputs['indices']
        
        if return_recons_only:
            return {
                "recon_pixel_values": video_recon,
                "indices": outputs['indices'],
            }
        
        # Compute loss
        if self.loss_config.use_abs_recons_loss:
            recons_loss = torch.abs(video_recon - outputs["patches"][:, [-1]]).mean()
        else:
            recons_loss = F.mse_loss(video_recon, outputs["patches"][:, [-1]])

        commit_loss = outputs['commit_loss']
        loss =  self.loss_config.commit_loss_w * commit_loss + self.loss_config.recon_loss_w * recons_loss 

        active_code_num = torch.tensor(torch.unique(outputs['indices']).shape[0]).float().to(loss.device)

        loss_outputs = dict()
        loss_outputs.update(
            {
                "active_code_num": active_code_num,
                "loss": loss,
                "commit_loss": commit_loss,
                "recons_loss": recons_loss,
            }
        )
        return loss_outputs
    
    def decode_image(self, cond_pixel_values, given_action_token_ids):
        return {
            "recon_pixel_values": cond_pixel_values,
        }
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

        target_video_recon = unpatchify(video_recon, self.patch_size, H, W)
        recon_pixel_values = target_video_recon.squeeze(1)

        return {
                "recon_pixel_values": recon_pixel_values,
            }

    def decode_obs(self, cond_pixel_values, target_pixel_values, obs_features):
        B, C, H, W = cond_pixel_values.shape
        video = torch.stack([cond_pixel_values, target_pixel_values], dim=1)
        # Encode + VQ
        video = rearrange(video, "b T c h w -> (b T) c h w")
        video = self.dino_transform(video)
        dion_features = self.dino_model(video)  # (b, img_tokens, img_feat_dim)
        dion_features = rearrange(dion_features, "(b T) l d -> b T l d", T=2)

        # patches = patchify(dion_features, self.patch_size)

        video_patches = self.patch_up(dion_features[:, :-1])
        z_q = obs_features.reshape(B, 1, self.num_codes, self.latent_dim)
        action_patches = self.action_up(z_q)

        video_action_patches = torch.cat([action_patches, video_patches], dim=2)

        # Decode
        video_recon = self.decoder(video_action_patches)
        video_recon = video_recon[:, :, self.num_codes: self.num_codes + video_patches.shape[2]] 

        recons_loss = F.mse_loss(video_recon, dion_features[:, [-1]])

        return {
                "recons_loss": recons_loss,
            }


    @property
    def device(self):
        return next(self.parameters()).device

    def get_state_dict_to_save(self):
        modules_to_exclude = []
        state_dict = {k: v for k, v in self.state_dict().items() if
                      not any(module_name in k for module_name in modules_to_exclude)}
        return state_dict
