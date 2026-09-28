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

##Use pretrained optical flow model
from torchvision import transforms
from torchvision.models.optical_flow import raft_small, Raft_Small_Weights  

# Use timm's names
IMAGENET_DEFAULT_MEAN = (0.485, 0.456, 0.406)
IMAGENET_DEFAULT_STD = (0.229, 0.224, 0.225)


class CKALoss(nn.Module):
    """计算两个表征空间拓扑结构的一致性"""
    def forward(self, X, Y):
        K = X @ X.t()
        L = Y @ Y.t()
        # 中心化矩阵 H = I - 1/n
        n = K.shape[0]
        H = torch.eye(n, device=K.device) - torch.ones((n, n), device=K.device) / n
        K_c = H @ K @ H
        L_c = H @ L @ H
        # 计算 HSIC 和 CKA
        hsic_kl = (K_c * L_c).sum()
        hsic_kk = (K_c * K_c).sum()
        hsic_ll = (L_c * L_c).sum()
        return 1 - hsic_kl / (torch.sqrt(hsic_kk * hsic_ll) + 1e-6)



class LatentActionTokenizer(nn.Module):
    """
    Latent action VQ-VAE.
    """

    def __init__(
            self,
            dino_model,
            freeze_vision,
            dino_dim: int,
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
        self.action_latent_dino = nn.Parameter(torch.empty(1, 1, self.num_codes, dino_dim))    # TODO: num of codes
        nn.init.uniform_(self.action_latent, a=-1, b=1)
        nn.init.uniform_(self.action_latent_dino, a=-1, b=1)
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

        self.encoder_dino = SpatioTemporalTransformer( 
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
        self.decoder_dino = SpatioTransformer(
            in_dim=model_dim,
            model_dim=model_dim,
            out_dim=dino_dim,       
            num_blocks=dec_blocks,
            num_heads=num_heads,
            dropout=dropout,
        )

        self.loss_config = loss_config
        import os
        os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
        os.environ["TORCH_HOME"] = "/mnt/workspace/tangzuojin.tzj/models"
        self.loss_fn_lpips = lpips.LPIPS(net='vgg').requires_grad_(False).eval()

        optical_weights = Raft_Small_Weights.DEFAULT
        self.optical_flow_model = raft_small(weights=optical_weights, progress=False).eval().to(self.device)
        self.optical_flow_model.eval()

        self.dino_model = dino_model
        self.dino_transform = transforms.Normalize(mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD)
        self.freeze_vision = freeze_vision
        if freeze_vision:
            for _, param in self.dino_model.named_parameters():
                param.requires_grad = False
        
        self.ckaloss = CKALoss()
        self.flow_proj = nn.Linear(latent_dim, 2)
        self.dino_proj = nn.Linear(latent_dim, dino_dim)

    def get_raft_flow_patch14(self, img1, img2):
        """
        img1, img2 形状为 [B, 3, H, W]，其中 H, W 是 14 的倍数
        """
        # 1. 计算填充
        # 例如：如果 H=224 (14*16)，224/8=28，恰好不需要填充
        # 例如：如果 H=196 (14*14)，196/8=24.5，需要填充到 200
        print("img1 shape:", img1.shape)
        print("img2 shape:", img2.shape)
        pad_h = (8 - img1.shape[-2] % 8) % 8
        pad_w = (8 - img1.shape[-1] % 8) % 8
        print("pad_h:", pad_h, "pad_w:", pad_w)
        
        img1_padded = F.pad(img1, (0, pad_w, 0, pad_h), mode='replicate')
        img2_padded = F.pad(img2, (0, pad_w, 0, pad_h), mode='replicate')
        print("img1_padded shape:", img1_padded.shape)
        print("img2_padded shape:", img2_padded.shape) 

        # 2. 推理
        # 注意：RAFT 内部会将输入缩小 1/8 处理
        with torch.no_grad():
            list_of_flows = self.optical_flow_model(img1_padded, img2_padded)
            flow = list_of_flows[-1] # [B, 2, H_padded, W_padded]
            print("Optical flow shape (padded):", flow.shape)

        # 3. 裁剪回原始尺寸
        if pad_h > 0 or pad_w > 0:
            flow = flow[:, :, :img1.shape[-2], :img1.shape[-1]]
            print("Optical flow shape (cropped):", flow.shape)
            
        return flow



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


    def vq_encode_pairs_dino(self, video: Tensor, attention_mask: Tensor = None) -> Dict:

        dion_features = self.dino_model(video)  # (b, img_tokens, img_feat_dim)
        print("DINO features shape (vq_encode_pairs_dino):", dion_features.shape)
        dion_features = rearrange(dion_features, "(b T) l d -> b T l d", T=2)

        # Preprocess videos
        NB, T = dion_features.shape[:2]
        action_pad = self.action_latent_dino.expand(NB, T, -1, -1)
        print("DINO features shape:", dion_features.shape)
        print("Action pad shape:", action_pad.shape)
        padded_patches = torch.cat([action_pad, dion_features], dim=2)

        # Encode
        z = self.encoder_dino(padded_patches, attention_mask) 

        # Get latent action for all future frames
        z = self.to_codebook(z[:, 1:, :self.num_codes])  # (B, T-1, n, E)
        physical_latent_action = z # extract physical latent action in continous space

        # Vector quantize
        z = z.reshape(NB * (T - 1), self.num_codes, self.latent_dim)
        z_q, indices, commit_loss  = self.vector_quantizer(z)
        z_q = z_q.reshape(NB, T - 1, self.num_codes, self.latent_dim)
        return {
            "patches": dion_features,
            "z_q": z_q,
            "indices": indices,
            "commit_loss": commit_loss,
            "physical_latent_action": physical_latent_action,
        }

    def forward(self, cond_pixel_values, target_pixel_values, next_target_pixel_values=None,return_action_token_ids_only=False, return_recons_only=False) -> Dict:
        # Encode + VQ
        H, W = cond_pixel_values.shape[2], cond_pixel_values.shape[3]
        print("cond_pixel_values shape:", cond_pixel_values.shape)
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
        print("Batch pairs shape:", batch_pairs.shape)

        batch_pairs_dino = torch.stack([
            torch.stack([cond_pixel_values, target_pixel_values], dim=1),   # 0->1
            torch.stack([target_pixel_values, next_target_pixel_values], dim=1),   # 1->2
        ], dim=0).reshape(-1, 2, 3, H, W)  # (B*2, 2, 3, H, W)
        print("Batch pairs DINO shape:", batch_pairs_dino.shape)

        batch_pairs_dino = rearrange(batch_pairs_dino, "b T c h w -> (b T) c h w")
        print("Batch pairs DINO shape:", batch_pairs_dino.shape)
        batch_pairs_dino = self.dino_transform(batch_pairs_dino)
        print("Batch pairs DINO after transform shape:", batch_pairs_dino.shape)

        vq_vae_outputs_dino = self.vq_encode_pairs_dino(batch_pairs_dino)
        print("DINO based VQ encoding done.", vq_vae_outputs_dino["z_q"].shape)
        z_q_dino_all = vq_vae_outputs_dino["z_q"]
        print("z_q_dino_all shape:", z_q_dino_all.shape)
        indices_dino_all = vq_vae_outputs_dino["indices"]
        commit_loss_dino_all = vq_vae_outputs_dino["commit_loss"]   
        physical_latent_action_dino_all = vq_vae_outputs_dino["physical_latent_action"]
        print("physical_latent_action_dino_all shape:", physical_latent_action_dino_all.shape)
        z_q_dino_01, z_q_dino_12 = torch.chunk(z_q_dino_all, 2, dim=0)
        print("DINO based VQ encoding done t01.", z_q_dino_01.shape)
        print("DINO based VQ encoding done t12.", z_q_dino_12.shape)
        
        vq_vae_outputs = self.vq_encode_pairs(batch_pairs)
        z_q_all = vq_vae_outputs["z_q"]
        print("z_q_all shape:", z_q_all.shape)
        indices_all = vq_vae_outputs["indices"]
        commit_loss_all = vq_vae_outputs["commit_loss"]
        physical_latent_action_all = vq_vae_outputs["physical_latent_action"]   

        z_q_01, z_q_12, z_q_02, z_q_10 = torch.chunk(z_q_all, 4, dim=0)
        phys_a_01, phys_a_12, phys_a_02, phys_a_10 = torch.chunk(physical_latent_action_all, 4, dim=0)
        print("phys_a_01 shape:", phys_a_01.shape)
        print("phys_a_12 shape:", phys_a_12.shape)
        print("phys_a_02 shape:", phys_a_02.shape)
        print("phys_a_10 shape:", phys_a_10.shape)

        indices_01, indices_12, indices_02, indices_10 = torch.chunk(indices_all, 4, dim=0)
        dec_in_01 = torch.cat([self.action_up(z_q_01), self.patch_up(p0)], dim=2)
        dec_in_12 = torch.cat([self.action_up(z_q_12), self.patch_up(p1)], dim=2)
        dec_in_02 = torch.cat([self.action_up(z_q_02), self.patch_up(p0)], dim=2)

        dec_inputs_all = torch.cat([dec_in_01, dec_in_12, dec_in_02], dim=0)  # (B*3, 1, num_patches, model_dim)
        all_recons = self.decoder(dec_inputs_all)
        all_recons = all_recons[:, :, self.num_codes: self.num_codes + p0.shape[2]]  # (B*3, 1, num_patches, patch_token_dim)
        
        all_recons = F.sigmoid(all_recons)
        recon_01, recon_12, recon_02 = torch.chunk(all_recons, 3, dim=0)

        recon_pixel_values_01 = unpatchify(recon_01, self.patch_size, H, W).squeeze(1)
        recon_pixel_values_12 = unpatchify(recon_12, self.patch_size, H, W).squeeze(1)
        recon_pixel_values_02 = unpatchify(recon_02, self.patch_size, H, W).squeeze(1)

        with torch.no_grad():
            optical_flow_outputs_01 = self.get_raft_flow_patch14(cond_pixel_values, target_pixel_values)
            optical_flow_outputs_12 = self.get_raft_flow_patch14(target_pixel_values, next_target_pixel_values)

            print("Optical flow extraction done t01", optical_flow_outputs_01.shape)
            print("Optical flow extraction done t12", optical_flow_outputs_12.shape)
        

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
                "recon_pixel_values_12": recon_pixel_values_12,
                "indices_12": indices_12,
                "recon_pixel_values_02": recon_pixel_values_02,
                "indices_02": indices_02,
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

        # 3. 光流一致性: flow(ot->ot+k) ≈ optical_flow(ot, ot+k)
        grad_size = int(self.num_codes ** 0.5)

        flow_downsampled_01 = F.adaptive_avg_pool2d(optical_flow_outputs_01, 
            output_size=(grad_size, grad_size)
        )
        flow_downsampled_12 = F.adaptive_avg_pool2d(optical_flow_outputs_12, 
            output_size=(grad_size, grad_size)
        )
        
        outputs_flow_01 = flow_downsampled_01.permute(0, 2, 3, 1).reshape(B, -1, 2)  # (B, num_patches, 2)
        outputs_flow_12 = flow_downsampled_12.permute(0, 2, 3, 1).reshape(B, -1, 2)  # (B, num_patches, 2)
        print("outputs_flow_01 shape:", outputs_flow_01.shape)
        print("outputs_flow_12 shape:", outputs_flow_12.shape)

        phys_a_01_flow = self.flow_proj(phys_a_01.squeeze(1))
        phys_a_12_flow = self.flow_proj(phys_a_12.squeeze(1))
        print("phys_a_01_flow shape:", phys_a_01_flow.shape)
        print("phys_a_12_flow shape:", phys_a_12_flow.shape)
    
        loss_flow = F.mse_loss(
            phys_a_01_flow, 
            outputs_flow_01
        ) + F.mse_loss(phys_a_12_flow, 
            outputs_flow_12
        )
       
        loss_cka = self.ckaloss(
            phys_a_01.squeeze(1).reshape(-1, self.latent_dim),
            z_q_dino_01.squeeze(1).reshape(-1, self.latent_dim)
        ) + self.ckaloss(
            phys_a_12.squeeze(1).reshape(-1, self.latent_dim), 
            z_q_dino_12.squeeze(1).reshape(-1, self.latent_dim)
        )

        loss =  self.loss_config.commit_loss_w * commit_loss_all + self.loss_config.recon_loss_w * avg_recon_loss_012 + \
                self.loss_config.perceptual_loss_w * avg_perceptual_loss_012 + self.loss_config.additivity_loss_w * additivity_loss + self.loss_config.cycle_loss_w * cycle_loss + self.loss_config.flow_loss_w * loss_flow + self.loss_config.cka_loss_w * loss_cka    

        # active_code_num = torch.tensor(len(set(indices.long().reshape(-1).cpu().numpy().tolist()))).float().to(loss.device)
        all_indices = torch.cat([indices_01.reshape(-1), indices_12.reshape(-1), indices_02.reshape(-1)], dim=0)
        active_code_num = torch.unique(all_indices).numel()
        active_code_num = torch.tensor(active_code_num).float().to(loss.device)

        loss_outputs = dict()
        loss_outputs.update(
            {
                "active_code_num": active_code_num,
                "loss": loss,
                "avg_commit_loss": commit_loss_all,
                "avg_recons_loss": avg_recon_loss_012,
                "avg_perceptual_loss": avg_perceptual_loss_012,
                "add_loss": additivity_loss,
                "cyc_loss": cycle_loss,
                "flow_loss": loss_flow,
                "cka_loss": loss_cka,
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
