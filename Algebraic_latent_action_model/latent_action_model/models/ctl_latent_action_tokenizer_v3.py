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
        physical_latent_action = z

        # Vector quantize
        z = z.reshape(B * (T - 1), self.num_codes, self.latent_dim)
        z_q, indices, commit_loss  = self.vector_quantizer(z)
        z_q = z_q.reshape(B, T - 1, self.num_codes, self.latent_dim)
        return {
            "patches": patches,
            "z_q": z_q,
            "indices": indices,
            "commit_loss": commit_loss,
            "physical_latent_action": physical_latent_action,
        }

    def vq_encode_pairs(self, patch_pairs: Tensor, attention_mask: Tensor = None) -> Dict:
        # Preprocess videos
        # input patch_pairs: (B*num_pairs, 2, num_patches, patch_dim), num_pairs: 01, 12, 02, 10
        
        NB, T = patch_pairs.shape[:2]

        action_pad = self.action_latent.expand(NB, T, -1, -1)
        padded_patches = torch.cat([action_pad, patch_pairs], dim=2)

        # Encode
        z = self.encoder(padded_patches, attention_mask) 

        # Get latent action for all future frames
        z_action = self.to_codebook(z[:, 1:, :self.num_codes])  # (NB, T-1, n, E)
        physical_latent_action = z_action # extract physical latent action in continous space
        # Vector quantize
        z_flattened = z_action.reshape(NB * (T - 1), self.num_codes, self.latent_dim)
        z_q, indices, commit_loss  = self.vector_quantizer(z_flattened)
        z_q = z_q.reshape(NB, T - 1, self.num_codes, self.latent_dim)
        return {
            "patches": patch_pairs,
            "z_q": z_q,
            "indices": indices,
            "commit_loss": commit_loss,
            "physical_latent_action": physical_latent_action,
        }
    
    def decode_additivity_lam(self, phys_a_1, phys_a_2) -> Dict:
        physical_latent_action = phys_a_1 + phys_a_2  # (NB, T-1, n, E)
        z_action = physical_latent_action
        NB, T = z_action.shape[:2]
        print("z_action shape in decode_additivity_lam:", z_action.shape)
        # Vector quantize
        z_flattened = z_action.reshape(-1, self.num_codes, self.latent_dim)
        z_q, indices, commit_loss  = self.vector_quantizer(z_flattened)
        z_q = z_q.reshape(NB, T, self.num_codes, self.latent_dim)
        return {
            "z_q": z_q,
            "indices": indices,
            "physical_latent_action": physical_latent_action,
        }

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
        
        vq_vae_outputs = self.vq_encode_pairs(batch_pairs)

        z_q_all = vq_vae_outputs["z_q"]
        indices_all = vq_vae_outputs["indices"]
        commit_loss_all = vq_vae_outputs["commit_loss"]
        physical_latent_action_all = vq_vae_outputs["physical_latent_action"]   

        z_q_01, z_q_12, z_q_02, z_q_10 = torch.chunk(z_q_all, 4, dim=0)
        phys_a_01, phys_a_12, phys_a_02, phys_a_10 = torch.chunk(physical_latent_action_all, 4, dim=0)
        indices_01, indices_12, indices_02, indices_10 = torch.chunk(indices_all, 4, dim=0)
        dec_in_01 = torch.cat([self.action_up(z_q_01), self.patch_up(p0)], dim=2)
        dec_in_12 = torch.cat([self.action_up(z_q_12), self.patch_up(p1)], dim=2)
        dec_in_02 = torch.cat([self.action_up(z_q_02), self.patch_up(p0)], dim=2)
        dec_in_10 = torch.cat([self.action_up(z_q_10), self.patch_up(p1)], dim=2)

        dec_inputs_all = torch.cat([dec_in_01, dec_in_12, dec_in_02, dec_in_10], dim=0)  # (B*4, 1, num_patches, model_dim)
        all_recons = self.decoder(dec_inputs_all)
        all_recons = all_recons[:, :, self.num_codes: self.num_codes + p0.shape[2]]  # (B*4, 1, num_patches, patch_token_dim)
        
        all_recons = F.sigmoid(all_recons)
        recon_01, recon_12, recon_02, recon_10 = torch.chunk(all_recons, 4, dim=0)

        recon_pixel_values_01 = unpatchify(recon_01, self.patch_size, H, W).squeeze(1)
        recon_pixel_values_12 = unpatchify(recon_12, self.patch_size, H, W).squeeze(1)
        recon_pixel_values_02 = unpatchify(recon_02, self.patch_size, H, W).squeeze(1)
        recon_pixel_values_10 = unpatchify(recon_10, self.patch_size, H, W).squeeze(1)

        # 提取三个关键动作用于物理约束
        # a01: O_t -> O_t+k
        # a12: O_t+k -> O_t+2k
        # a02: O_t -> O_t+2k (跨步动作)
        # a10: O_t+k -> O_t (用于循环一致性/可逆性)

        if return_action_token_ids_only:
            return indices_01
        
        if return_recons_only:
            return {
                "recon_pixel_values_01": recon_pixel_values_01,
                "indices_01": indices_01,
                "phys_a_01": phys_a_01,
                "recon_pixel_values_12": recon_pixel_values_12,
                "indices_12": indices_12,
                "phys_a_12": phys_a_12,
                "recon_pixel_values_02": recon_pixel_values_02,
                "indices_02": indices_02,
                "phys_a_02": phys_a_02,
                "recon_pixel_values_10": recon_pixel_values_10,
                "indices_10": indices_10,
                "phys_a_10": phys_a_10,
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
        # 1. 动作可加性: a01 + a12 ≈ a02
        additivity_loss = F.mse_loss(phys_a_02, phys_a_01 + phys_a_12)
        print(f"Additivity Loss: {additivity_loss.item()}")

        # 2. 循环一致性 (可逆性): a01 + a10 ≈ 0 (或者 z_q_01 ≈ -z_q_10)
        cycle_loss = F.mse_loss(phys_a_01 + phys_a_10, torch.zeros_like(phys_a_10))
        # print(f"Cycle Loss: {cycle_loss.item()}")
        # print(f"Commit Loss: {commit_loss_all.item()}")
        # print(f"Recon Loss 01: {recons_loss_01.item()}, Recon Loss 12: {recons_loss_12.item()}, Recon Loss 02: {recons_loss_02.item()}")
        # print(f"Perceptual Loss 01: {perceptual_loss_01.item()}, Perceptual Loss 12: {perceptual_loss_12.item()}, Perceptual Loss 02: {perceptual_loss_02.item()}")
        # print("-----------------------------------------------------")
        
        loss =  self.loss_config.commit_loss_w * commit_loss_all + self.loss_config.recon_loss_w * avg_recon_loss_012 + \
                self.loss_config.perceptual_loss_w * avg_perceptual_loss_012 + self.loss_config.additivity_loss_w * additivity_loss + self.loss_config.cycle_loss_w * cycle_loss

        # active_code_num = torch.tensor(len(set(indices.long().reshape(-1).cpu().numpy().tolist()))).float().to(loss.device)
        all_indices = torch.cat([indices_01.reshape(-1), indices_12.reshape(-1), indices_02.reshape(-1)], dim=0)
        active_code_num = torch.unique(all_indices).numel()
        active_code_num = torch.tensor(active_code_num).float().to(loss.device)

        loss_outputs = dict()
        loss_outputs.update(
            {
                "active_code_num": active_code_num,
                "loss": loss,
                "avg_commit_loss": self.loss_config.commit_loss_w * commit_loss_all,
                "avg_recons_loss": self.loss_config.recon_loss_w * avg_recon_loss_012,
                "avg_perceptual_loss": self.loss_config.perceptual_loss_w * avg_perceptual_loss_012,
                "add_loss": self.loss_config.additivity_loss_w * additivity_loss,
                "cyc_loss": self.loss_config.cycle_loss_w * cycle_loss,
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
