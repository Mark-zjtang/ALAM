from multiprocessing import set_forkserver_preload
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
import math


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
        self.model_dim = model_dim

        self.action_latent = nn.Parameter(torch.empty(1, 1, 1, patch_token_dim))    # TODO: num of codes
        nn.init.uniform_(self.action_latent, a=-1, b=1)
        self.encoder = SpatioTemporalTransformer(
            in_dim=patch_token_dim,
            model_dim=model_dim,
            out_dim=model_dim,
            num_blocks=enc_blocks,
            num_heads=num_heads,
            dropout=dropout,
            causal_temporal=True,
            to_out=False,
        )
        self.fc = nn.Linear(model_dim, latent_dim * 2)

        ## Decoder: Spatial Transformer
        self.patch_up = nn.Linear(patch_token_dim, model_dim)
        self.action_up = nn.Linear(latent_dim, model_dim)
        self.physical_latent_action_up = nn.Linear(latent_dim, model_dim)
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
        os.environ["TORCH_HOME"] = "/mnt/workspace/tangzuojin.tzj/models"
        self.loss_fn_lpips = lpips.LPIPS(net='vgg').requires_grad_(False).eval()
        self.training = True

    def vae_encode_pairs(self, patch_pairs: Tensor, attention_mask: Tensor = None) -> Dict:
        # Preprocess videos
        # input patch_pairs: (B*num_pairs, 2, num_patches, patch_dim), num_pairs: 01, 12, 02, 10
        
        NB, T = patch_pairs.shape[:2]

        action_pad = self.action_latent.expand(NB, T, -1, -1)
        padded_patches = torch.cat([action_pad, patch_pairs], dim=2)

        # Encode
        z = self.encoder(padded_patches, attention_mask) 

        # Get latent action for all future frames
        z = z[:, 1:, 0]  # (NB, T-1, 1, E)
        # Vector quantize
        z = z.reshape(NB * (T - 1), self.model_dim)
        moments = self.fc(z)

        z_mu, z_var = torch.chunk(moments, 2, dim=-1)
        # Reparameterization
        if not self.training:
            z_rep = z_mu
        else:
            z_rep = z_mu + torch.randn_like(z_var) * torch.exp(0.5 * z_var)

        z_rep = z_rep.reshape(NB, (T - 1), 1,  self.latent_dim)

        # if not self.training:
        #     if self.mu_record is None:
        #         self.mu_record = z_mu
        #     else:
        #         self.mu_record = torch.cat([self.mu_record, z_mu], dim=0)  

        return {
            "patches": patch_pairs,
            "z_mu": z_mu,
            "z_var": z_var,
            "physical_latent_action": z_rep,
        }
     
    
    # def decode_additivity_lam(self, phys_a_1, phys_a_2) -> Dict:
    #     physical_latent_action = phys_a_1 + phys_a_2  # (NB, T-1, n, E)
      
    #     return {
    #         "phys_latent_action": physical_latent_action,
    #     }

    def forward(self, cond_pixel_values, target_pixel_values, next_target_pixel_values=None,return_action_token_ids_only=False, return_recons_only=False) -> Dict:
        # Encode + VQ
        H, W = cond_pixel_values.shape[2], cond_pixel_values.shape[3]

        all_frames = torch.cat([cond_pixel_values, target_pixel_values, next_target_pixel_values], dim=0)
        all_patches = patchify(all_frames.unsqueeze(1), self.patch_size)
        p0, p1, p2 = torch.chunk(all_patches, 3, dim=0)
        B = p0.shape[0]

        batch_pairs = torch.stack([
            torch.stack([p0, p1], dim=1),   # 0->1
            torch.stack([p1, p2], dim=1),   # 1->2
            torch.stack([p0, p2], dim=1),   # 0->2
            torch.stack([p1, p0], dim=1),   # 1->0
        ], dim=0).reshape(4*B, 2, p0.shape[2], p0.shape[3])  # (B*4, 2, num_patches, patch_dim)
        vae_outputs = self.vae_encode_pairs(batch_pairs)

        z_mu_all = vae_outputs["z_mu"]
        z_var_all = vae_outputs["z_var"]
        physical_latent_action_all = vae_outputs["physical_latent_action"] 

        z_01, z_12, z_02, z_10 = torch.chunk(physical_latent_action_all, 4, dim=0)
        z_mu_01, z_mu_12, z_mu_02, z_mu_10 = torch.chunk(z_mu_all, 4, dim=0)
        z_var_01, z_var_12, z_var_02, z_var_10 = torch.chunk(z_var_all, 4, dim=0)

        dec_in_01 = torch.cat([self.action_up(z_01), self.patch_up(p0)], dim=2)
        dec_in_12 = torch.cat([self.action_up(z_12), self.patch_up(p1)], dim=2)
        dec_in_02 = torch.cat([self.action_up(z_02), self.patch_up(p0)], dim=2)

        dec_inputs_all = torch.cat([dec_in_01, dec_in_12, dec_in_02], dim=0)  # (B*3, 1, num_patches, model_dim)
        all_recons = self.decoder(dec_inputs_all)
        all_recons = all_recons[:, :, 1:, :]  # (B*3, 1, num_patches, patch_token_dim)
        
        all_recons = F.sigmoid(all_recons)
        recon_01, recon_12, recon_02 = torch.chunk(all_recons, 3, dim=0)

        recon_pixel_values_01 = unpatchify(recon_01, self.patch_size, H, W).squeeze(1)
        recon_pixel_values_12 = unpatchify(recon_12, self.patch_size, H, W).squeeze(1)
        recon_pixel_values_02 = unpatchify(recon_02, self.patch_size, H, W).squeeze(1)

        # 提取三个关键动作用于物理约束
        # a01: O_t -> O_t+k
        # a12: O_t+k -> O_t+2k
        # a02: O_t -> O_t+2k (跨步动作)
        # a10: O_t+k -> O_t (用于循环一致性/可逆性)

        if return_action_token_ids_only:
            return z_01
        
        if return_recons_only:
            return {
                "recon_pixel_values_01": recon_pixel_values_01,
                "phys_a_01": z_01,
                "recon_pixel_values_12": recon_pixel_values_12,
                "phys_a_12": z_12,
                "recon_pixel_values_02": recon_pixel_values_02,
                "phys_a_02": z_02,
            }
        
        # Compute loss
        if self.loss_config.use_abs_recons_loss:
            recons_loss_01 = torch.abs(recon_pixel_values_01 - target_pixel_values).mean()
            recons_loss_12 = torch.abs(recon_pixel_values_12 - next_target_pixel_values).mean()
            recons_loss_02 = torch.abs(recon_pixel_values_02 - next_target_pixel_values).mean()
            avg_recon_loss_012 = (recons_loss_01 + recons_loss_12 + recons_loss_02) / 3
          
        else:
            recons_loss_01 = F.mse_loss(recon_pixel_values_01, target_pixel_values)
            recons_loss_12 = F.mse_loss(recon_pixel_values_12, next_target_pixel_values)
            recons_loss_02 = F.mse_loss(recon_pixel_values_02, next_target_pixel_values)
            avg_recon_loss_012 = (recons_loss_01 + recons_loss_12 + recons_loss_02) / 3
        

        if self.loss_config.perceptual_loss_w > 0:
            with torch.no_grad():
                perceptual_loss_01 = self.loss_fn_lpips.forward(recon_pixel_values_01, target_pixel_values, normalize=True).mean()
                perceptual_loss_12 = self.loss_fn_lpips.forward(recon_pixel_values_12, next_target_pixel_values, normalize=True).mean()
                perceptual_loss_02 = self.loss_fn_lpips.forward(recon_pixel_values_02, next_target_pixel_values, normalize=True).mean()
                avg_perceptual_loss_012 = (perceptual_loss_01 + perceptual_loss_02 + perceptual_loss_12) / 3

        else:
            perceptual_loss_01 = torch.zeros_like(recons_loss_01)
            perceptual_loss_12 = torch.zeros_like(recons_loss_12)
            perceptual_loss_02 = torch.zeros_like(recons_loss_02)
            avg_perceptual_loss_012 = (perceptual_loss_01 + perceptual_loss_02 + perceptual_loss_12) / 3

        # 物理约束 Loss 计算
        # 对三个动作分别做标准 KL (对比 N(0,1))
        kl_01 = -0.5 * torch.sum(1 + z_var_01 - z_mu_01**2 - z_var_01.exp(), dim=1).mean()
        kl_12 = -0.5 * torch.sum(1 + z_var_12 - z_mu_12**2 - z_var_12.exp(), dim=1).mean()
        kl_02 = -0.5 * torch.sum(1 + z_var_02 - z_mu_02**2 - z_var_02.exp(), dim=1).mean()

        kl_loss = (kl_01 + kl_12 + kl_02) / 3

        # 可逆性约束: Encoder(s1, s0)_lam_10 -> Encoder(s0, s1)_lam_01
        kl_rev = -0.5 * torch.sum(1 - ( z_var_01 - z_var_10) - (z_var_10.exp() + (z_mu_01 + z_mu_10)**2) / (z_var_01.exp() + 1e-6), dim=1).mean()

        scale = math.sqrt(2)
        z_mu_02_target_add = ( z_mu_01 + z_mu_12 ) / scale
        z_var_02_target_add = (z_var_01.exp() + z_var_12.exp()) / scale**2
        logz_var_02_target_add = z_var_02_target_add.log()

        kl_add = -0.5 * torch.sum(1 - (logz_var_02_target_add - z_var_02) - (z_var_02.exp() + (z_mu_02 - z_mu_02_target_add)**2) / (z_var_02_target_add + 1e-6), dim=1).mean()
        
        loss =  self.loss_config.kl_loss * kl_loss + self.loss_config.recon_loss_w * avg_recon_loss_012 + \
                self.loss_config.perceptual_loss_w * avg_perceptual_loss_012 + self.loss_config.additivity_loss_w * kl_add + self.loss_config.cycle_loss_w * kl_rev

        loss_outputs = dict()
        loss_outputs.update(
            {
                "loss": loss,
                "kl_loss": self.loss_config.kl_loss * kl_loss,
                "avg_recons_loss": self.loss_config.recon_loss_w * avg_recon_loss_012,
                "avg_perceptual_loss": self.loss_config.perceptual_loss_w * avg_perceptual_loss_012,
                "add_loss": self.loss_config.additivity_loss_w * kl_add,
                "cyc_loss": self.loss_config.cycle_loss_w * kl_rev,
            }
        )
        return loss_outputs


    
    def decode_image(self, cond_pixel_values, given_action_token_ids):
        B, C, H, W = cond_pixel_values.shape
        videos = cond_pixel_values.reshape(B, 1, C, H, W) # (b, 1, c, h, w)
        patches = patchify(videos, self.patch_size)
        video_patches = self.patch_up(patches)
        z_q = given_action_token_ids.reshape(B, 1, 1, self.latent_dim)
        action_patches = self.action_up(z_q)
        video_action_patches = torch.cat([action_patches, video_patches], dim=2)

        # Decode
        video_recon = self.decoder(video_action_patches)
        video_recon = video_recon[:, :, 1:, :]  # (B*3, 1, num_patches, patch_token_dim)
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
