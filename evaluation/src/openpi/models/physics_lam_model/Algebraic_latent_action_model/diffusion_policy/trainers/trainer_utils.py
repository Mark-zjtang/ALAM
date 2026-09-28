import torch.nn.functional as F
import math
import cv2
from PIL import Image, ImageFont, ImageDraw
import os
import torchvision.transforms as T
import numpy as np
import torch
import torch.nn as nn

def cross_entropy(pred, target, reduction):
    # print(pred.shape, target.shape)
    loss = F.cross_entropy(pred.reshape(-1, pred.shape[-1]), target.reshape(-1), reduction=reduction)
    loss = loss.reshape(target.shape)
    return loss


def masked_loss(pred, target, mask, skip_frame=0, loss_func=F.mse_loss):
    if skip_frame == 0:
        new_pred = pred
    else:
        new_pred = pred[:, :-skip_frame]
    new_target = target[:, skip_frame:]
    new_mask = mask[:, skip_frame:]
    data_shape, mask_shape = new_target.shape, new_mask.shape
    loss = loss_func(new_pred, new_target, reduction='none')
    for _ in range(len(data_shape) - len(mask_shape)):
        new_mask = new_mask.unsqueeze(-1)
    loss = (loss*new_mask).sum() / new_mask.sum() / math.prod(data_shape[len(mask_shape):])
    return loss



def visualize_latent_action_gen(
        lang_goal,
        orig_video, 
        decoding_mode2preds,
        path
    ):
    _, c, h, w = orig_video.shape
    n_rows = len(decoding_mode2preds)+1
    h = h + 30

    orig_video = list(map(T.ToPILImage(), orig_video.unbind(dim=0)))
    initial_frame = orig_video[0]
    gt_subsequent_frames = orig_video[1:]
    for decoding_mode, preds in decoding_mode2preds.items():
        preds['latent_action_id_preds'] = preds['latent_action_id_preds'].numpy().tolist()
        preds['frame_preds'] = list(map(T.ToPILImage(), preds['frame_preds'].unbind(dim=0)))
        n_cols = len(preds['frame_preds']) + 1

    font_path = os.path.join(cv2.__path__[0],'qt','fonts','DejaVuSans.ttf')
    font = ImageFont.truetype(font_path, size=12)
    compare_img = Image.new('RGB', size=(n_cols*w, n_rows*h))
    draw_compare_img = ImageDraw.Draw(compare_img)
    
    for i in range(n_rows):
        compare_img.paste(initial_frame, box=(0, i*h))

    for j in range(n_cols-1):
        if j < len(gt_subsequent_frames):
            compare_img.paste(gt_subsequent_frames[j], box=((j+1)*w, 0))

        for i, (decoding_mode, preds) in enumerate(decoding_mode2preds.items()):
            if j < len(preds['frame_preds']):
                compare_img.paste(preds['frame_preds'][j], box=((j+1)*w, (i+1)*h))
                draw_compare_img.text(((j+1)*w, (i+2)*h-20), f"{preds['latent_action_id_preds'][j]}", font=font, fill=(0, 255, 0))
            
            if j == 0:
                draw_compare_img.text((0, (i+2)*h-20), f"{decoding_mode}", font=font, fill=(0, 255, 0))

    draw_compare_img.text((0, h-20), f"{lang_goal}", font=font, fill=(0, 255, 0))
    compare_img.save(f"{path}-{'_'.join(lang_goal.split())}.png")


    h = h - 30
    fps = 4
    for i, (decoding_mode, preds) in enumerate(decoding_mode2preds.items()):
        output_video_path = f"{path}-{'_'.join(lang_goal.split())}-{decoding_mode}.mp4"
        images = preds['frame_preds']
        images = [initial_frame] + images

        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        video_writer = cv2.VideoWriter(output_video_path, fourcc, fps, (w, h))

        for image in images:
            image_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
            video_writer.write(image_cv)
        video_writer.release()


# ────────────────────────────────────────────────────────────
# ★ 新增：物理耦合一致性损失
# 约束：世界潜向量的累积变化 ↔ 动作积分的预测变化 保持一致
# ────────────────────────────────────────────────────────────
class WorldActionCouplingLoss(nn.Module):
    """
    计算 pred_w 与 pred_v 之间的物理耦合一致性损失。

    直觉：
        如果动作 a_i 导致了世界状态变化 w_i，
        那么 cumsum(w) 与 f(cumsum(a)) 应该在投影空间内对齐，
        其中 f 是可学习的线性映射（运动学前向模型的近似）。

    Args:
        action_dim  (int): 动作维度
        latent_dim  (int): 世界潜向量维度
    """
    def __init__(self, action_dim: int, latent_dim: int):
        super().__init__()
        # 可学习的近似运动学映射 a → w 空间
        self.kinematic_proj = nn.Linear(action_dim, latent_dim, bias=False)
        nn.init.orthogonal_(self.kinematic_proj.weight)   # 正交初始化，避免早期坍塌

    def forward(
        self,
        pred_w: torch.Tensor,   # (B, H, latent_dim)  世界速度场预测
        pred_v: torch.Tensor,   # (B, H, action_dim)  动作速度场预测
        mask:   torch.Tensor,   # (B,)  有效样本 mask
    ) -> torch.Tensor:
        """
        Returns:
            scalar loss
        """
        # ── 1. 累积求和（沿时间轴 H）────────────────────────────────────
        w_cumsum = torch.cumsum(pred_w, dim=1)   # (B, H, latent_dim)
        a_cumsum = torch.cumsum(pred_v, dim=1)   # (B, H, action_dim)

        # ── 2. 将动作累积量投影到世界潜空间 ──────────────────────────────
        a_proj = self.kinematic_proj(a_cumsum)   # (B, H, latent_dim)

        # ── 3. 全局一致性损失（所有时间步）──────────────────────────────
        #    stop_gradient on a_proj：只拉动 world 分支对齐动作分支
        #    避免两个分支互相坍塌到零
        L_global = F.mse_loss(
            w_cumsum,
            a_proj.detach(),
            reduction='none'
        ).mean(dim=[1, 2])                        # (B,)

        # ── 4. 时序一致性损失（相邻步差分）──────────────────────────────
        dw    = pred_w[:, 1:] - pred_w[:, :-1]   # (B, H-1, latent_dim)
        da    = pred_v[:, 1:] - pred_v[:, :-1]   # (B, H-1, action_dim)
        da_proj = self.kinematic_proj(da)          # (B, H-1, latent_dim)

        L_temporal = F.mse_loss(
            dw,
            da_proj.detach(),
            reduction='none'
        ).mean(dim=[1, 2])                         # (B,)

        # ── 5. mask 加权求均值 ────────────────────────────────────────────
        mask = mask.float()
        loss = (0.5 * L_global + 0.5 * L_temporal) * mask
        loss = loss.sum() / (mask.sum() + 1e-8)

        return loss
