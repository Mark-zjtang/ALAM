from typing import Dict, List
import lpips
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from einops import rearrange, repeat
from uni_world_model.latent_action_model.modules.blocks import patchify, unpatchify, SpatioTemporalTransformer, SpatioTransformer, \
                                                     MVSpatioTemporalTransformer, MVSpatioTransformer
from uni_world_model.latent_action_model.modules.vector_quantizer import VectorQuantizer, VectorQuantizer2
from IPython import embed
from torchvision import transforms

import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
os.environ["TORCH_HOME"] = "/mnt/workspace/czj/models"

IMAGENET_DEFAULT_MEAN = (0.485, 0.456, 0.406)
IMAGENET_DEFAULT_STD = (0.229, 0.224, 0.225)
class LatentActionTokenizer(nn.Module):
    """
    Latent action VQ-VAE.
    """

    def __init__(
            self,
            in_dim: int,
            model_dim: int,
            latent_dim: int,
            patch_size: int,
            enc_blocks: int,
            dec_blocks: int,
            num_heads: int,
            action_num_codes: int,
            num_frame_pred: int,
            loss_config,
            dropout: float = 0.0,
    ) -> None:
        super(LatentActionTokenizer, self).__init__()
        self.dino_transform = transforms.Normalize(mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD)
        self.dino_encoder = torch.hub.load('/mnt/workspace/czj/models/dinov2', 'dinov2_vitb14_reg', source='local')
        self.dino_encoder.requires_grad_(False)

        self.model_dim = model_dim

        dino_dim = 768
        self.latent_dim = latent_dim
        self.patch_size = patch_size
        patch_token_dim = in_dim * patch_size ** 2
        patches_depth_token_dim = 1 * patch_size ** 2
        self.num_codes = action_num_codes
        self.num_frame_pred = num_frame_pred
        self.action_latent = nn.Parameter(torch.empty(1, 1, 1, dino_dim))    # TODO: num of codes
        nn.init.uniform_(self.action_latent, a=-1, b=1)
        self.encoder = SpatioTemporalTransformer(
            in_dim=dino_dim,
            model_dim=model_dim,
            out_dim=model_dim,
            num_blocks=enc_blocks,
            num_heads=num_heads,
            dropout=dropout,
            causal_temporal=True,
        )
        self.fc = nn.Linear(model_dim, latent_dim * 2)
        ## Decoder: Spatial Transformer
        self.patch_up = nn.Linear(patch_token_dim, model_dim)
        self.patch_up_depth = nn.Linear(patches_depth_token_dim, model_dim)
        self.action_up = nn.Linear(latent_dim, model_dim)
        self.decoder = SpatioTransformer(
            in_dim=model_dim,
            model_dim=model_dim,
            out_dim=patch_token_dim,       
            num_blocks=dec_blocks,
            num_heads=num_heads,
            dropout=dropout,
        )
        self.timestep_embedding = nn.Parameter(torch.empty(self.num_frame_pred, 1, model_dim))
        nn.init.uniform_(self.timestep_embedding, a=-1, b=1)
        self.loss_config = loss_config
        self.loss_fn_lpips = lpips.LPIPS(net='vgg').requires_grad_(False).eval()

        self.training = True


    def vae_encode(self, videos: Tensor, attention_mask: Tensor = None) -> Dict:
        # Preprocess videos
        B, T = videos.shape[:2]
        patches = patchify(videos, self.patch_size)
        videos = rearrange(videos, "b T c h w -> (b T) c h w")
        videos = self.dino_transform(videos)
        dino_features = self.dino_encoder.forward_features(videos)['x_norm_patchtokens']
        dino_features = rearrange(dino_features, "(b T) l d -> b T l d", b = B, T = T)

        action_pad = self.action_latent.expand(B, T, -1, -1)
        padded_patches = torch.cat([action_pad, dino_features], dim=2)

        # Encode
        # full attention
        z = self.encoder(padded_patches)
      
        # Get latent action for all future frames
        z = z[:, 1:, 0]  # (B, T-1, 1, E)

        # Vector quantize
        z = z.reshape(B * (T - 1), self.model_dim)
        moments = self.fc(z)
        z_mu, z_var = torch.chunk(moments, 2, dim=-1)
        
        # Reparameterization
        print("is training", self.training)
        if not self.training:
            z_rep = z_mu
        else:
            z_rep = z_mu + torch.randn_like(z_var) * torch.exp(0.5 * z_var)
        z_rep = z_rep.reshape(B, T - 1, 1, self.latent_dim)

        if not self.training:
            if self.mu_record is None:
                self.mu_record = z_mu
            else:
                self.mu_record = torch.cat([self.mu_record, z_mu], dim=0)

        return {
            "dino_features": dino_features,
            "patches": patches,
            "z_mu": z_mu,
            "z_var": z_var,
        }

    def forward(self, cond_pixel_values, target_pixel_values, return_action_token_ids_only=False, return_recons_only=False) -> Dict:
        video = torch.cat([cond_pixel_values, target_pixel_values], dim=1)
        # Encode + VQ
        video = video[:,:,:3] # B,T,C,H,W
        B, T = video.shape[:2]
        H, W = video.shape[3:5]
        outputs = self.vae_encode(video) 
        if return_action_token_ids_only:
            return outputs['z_rep']
        video_patches = self.patch_up(outputs["patches"][:, :1]) # (B, 1, l, d)

        video_patches = video_patches.repeat(1, T-1, 1, 1) # (B, T-1, l, d)

        action_patches = self.action_up(outputs["z_rep"])    # (B, T-1, 1, d)
        video_action_patches = video_patches + action_patches # (B, T-1, num_patches=l, model_dim=d)

        # Decode
        # mask future frames
        video_recon_all = self.decoder(video_action_patches) # (B, T-1, l, p)
        video_recon = F.sigmoid(video_recon_all)

        target_video_recon = unpatchify(video_recon, self.patch_size, H, W) # (B, T-1, C, H, W)
        # recon_pixel_values = target_video_recon.squeeze(1)
        recon_pixel_values = target_video_recon
        
        if return_recons_only:
            return {
                "recon_pixel_values": recon_pixel_values,
                "indices": outputs['z_rep'], ## real continous lam value not indices 
            }
        
        # Compute loss
        recon_pixel_values = rearrange(recon_pixel_values, 'b t c h w -> (b t) c h w')
        target_pixel_values = rearrange(target_pixel_values[:,:,:3], 'b t c h w -> (b t) c h w')

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

        kl_loss = -0.5 * torch.sum(1 + outputs["z_var"] - outputs["z_mu"] ** 2 - outputs["z_var"].exp(), dim=1).mean()
        loss =  self.loss_config.commit_loss_w * kl_loss + self.loss_config.recon_loss_w * recons_loss + \
                self.loss_config.perceptual_loss_w * perceptual_loss
        
        # active_code_num = torch.tensor(len(set(indices.long().reshape(-1).cpu().numpy().tolist()))).float().to(loss.device)


        loss_outputs = dict()
        loss_outputs.update(
            {
                "loss": loss,
                "commit_loss": kl_loss,
                "recons_loss": recons_loss,
                "perceptual_loss": perceptual_loss,
            }
        )
        return loss_outputs
    
    @torch.no_grad()
    def decode_image(self, cond_pixel_values, given_action_token_ids):
        B, _, _, H, W = cond_pixel_values.shape
        given_action_token_ids = given_action_token_ids.reshape(B*self.num_frame_pred, self.num_codes)
        patches = patchify(cond_pixel_values, self.patch_size)
        video_patches = self.patch_up(patches)  # (B, 1, num_patches, model_dim)
        video_patches = video_patches.repeat(1, self.num_frame_pred, 1, 1) # (B, T-1, l, d)

        z_q = self.vector_quantizer.codebook(given_action_token_ids, shape=(B*self.num_frame_pred, self.num_codes, self.latent_dim))
        z_q = z_q.reshape(B, self.num_frame_pred, self.num_codes, self.latent_dim)
        action_patches = self.action_up(z_q)    # (B, T-1, n, d)
        timestep_embedding = self.timestep_embedding.repeat(B, 1, 1, 1)
        video_action_patches = torch.cat([action_patches, video_patches, timestep_embedding], dim=2) # (B, T-1, n+l, d)

        # Decode
        # mask future frames
        video_recon = self.decoder(video_action_patches) # (B, T-1, n+l, p)
        video_recon = video_recon[:, :, self.num_codes: self.num_codes + video_patches.shape[2]] 
        video_recon = F.sigmoid(video_recon)

        target_video_recon = unpatchify(video_recon, self.patch_size, H, W) # (B, T-1, C, H, W)
        # recon_pixel_values = target_video_recon.squeeze(1)
        recon_pixel_values = target_video_recon

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
