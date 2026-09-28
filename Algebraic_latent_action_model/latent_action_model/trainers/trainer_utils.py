import torch.nn.functional as F
import math
import cv2
from PIL import Image, ImageFont, ImageDraw
import os
import torchvision.transforms as T
import numpy as np
import torch
import lpips
from torchmetrics.functional import peak_signal_noise_ratio, structural_similarity_index_measure


def visualize_latent_motion_reconstruction(
    initial_frame,
    next_frame,
    recons_next_frame,
    latent_motion_ids,
    path
):
    c, h, w = initial_frame.shape
    h = h + 30
    initial_frame = T.ToPILImage()(initial_frame)
    next_frame = T.ToPILImage()(next_frame)
    recons_next_frame = T.ToPILImage()(recons_next_frame)
    latent_motion_ids = latent_motion_ids.numpy().tolist()

    compare_img = Image.new('RGB', size=(3*w, h))
    draw_compare_img = ImageDraw.Draw(compare_img)

    compare_img.paste(initial_frame, box=(0, 0))
    compare_img.paste(next_frame, box=(w, 0))
    compare_img.paste(recons_next_frame, box=(2*w, 0))

    font_path = os.path.join(cv2.__path__[0],'qt','fonts','DejaVuSans.ttf')
    font = ImageFont.truetype(font_path, size=12)
    draw_compare_img.text((w, h-20), f"{latent_motion_ids}", font=font, fill=(0, 255, 0))
    compare_img.save(path)



# def visualize_eval_lam_reconstruction(
#     gt_images,          #  [gt0, gt1, gt2, gt3, gt4]
#     rec_images,         #  [rec1, rec2, rec3, rec4]
#     latent_motion_ids,  #  [id_for_rec1, id_for_rec2, id_for_rec3, id_for_rec4]
#     path
# ):
#     c, h, w = gt_images[0].shape

#     row1_padding = 30
#     row2_padding = 45
    
#     to_pil = T.ToPILImage()

#     ids_list = None
#     if latent_motion_ids is not None:
#         if torch.is_tensor(latent_motion_ids):
#             ids_list = latent_motion_ids.cpu().numpy().tolist()
#         else:
#             ids_list = list(latent_motion_ids)
   
#     canvas_w = 5 * w
#     canvas_h = h + row1_padding + h + row2_padding
#     compare_img = Image.new('RGB', size=(canvas_w, canvas_h), color=(0, 0, 0))
#     draw = ImageDraw.Draw(compare_img)

#     try:
#         font_path = os.path.join(cv2.__path__[0], 'qt', 'fonts', 'DejaVuSans.ttf')
#         font = ImageFont.truetype(font_path, size=12)
#     except:
#         font = ImageFont.load_default()

#     green = (0, 255, 0)

#     # --- 第一行绘制 (Ground Truth) ---
#     row1_labels = [
#         "ground_truth image 0", 
#         "ground_truth image 1", 
#         "ground_truth image 2", 
#         "ground_truth image 2",
#         "ground_truth image 2_additivity"
#     ]
    
#     for i in range(5):
#         x = i * w
#         y = 0
#         compare_img.paste(to_pil(gt_images[i]), box=(x, y))
#         draw.text((x + 5, h + 5), row1_labels[i], font=font, fill=green)

#     row2_images = [gt_images[0]] + rec_images
#     row2_labels = [
#         "ground_truth image 0", 
#         "input_gt image 0-add_phys_a_01", 
#         "input_gt image 0-add_phys_a_02", 
#         "input_gt image 1-add_phys_a_12", 
#         "input_gt_image 0-add_phys_a_(01_add_12)"
#     ]
    
#     row2_y_start = h + row1_padding 
    
#     for i in range(5):
#         x = i * w
#         y = row2_y_start
#         compare_img.paste(to_pil(row2_images[i]), box=(x, y))
#         label_y = y + h + 5
#         draw.text((x + 5, label_y), row2_labels[i], font=font, fill=green)

#         if i > 0 and ids_list is not None:
#             if (i-1) < len(ids_list):
#                 current_id = ids_list[i-1] 
#                 ids_y = label_y + 15
#                 draw.text((x + 5, ids_y), f"ID: {current_id}", font=font, fill=green)
        

#     compare_img.save(path)



# def calculate_psl_metrics(img1, img2, device='cuda'):
#     """
#     img1, img2: torch.Tensor, shape [B, 3, H, W], range [0, 1]
#     """
#     img1 = img1.to(device)
#     img2 = img2.to(device)

#     # 1. PSNR (Higher is better)
#     # data_range=1.0 表示输入是 [0, 1]
#     psnr_val = peak_signal_noise_ratio(img1, img2, data_range=1.0)

#     # 2. SSIM (Higher is better, range 0~1)
#     ssim_val = structural_similarity_index_measure(img1, img2, data_range=1.0)

#     # 3. LPIPS (Lower is better, perceptual similarity)
#     # 注意：LPIPS 默认期望输入范围是 [-1, 1]
#     # 我们需要将 [0, 1] 转换为 [-1, 1]
#     loss_fn_alex = lpips.LPIPS(net='alex').to(device) # 或者用 'vgg'
    
#     img1_norm = img1 * 2 - 1
#     img2_norm = img2 * 2 - 1
    
#     # lpips 返回的是每个样本的距离，取平均值
#     lpips_val = loss_fn_alex(img1_norm, img2_norm).mean()
#     print(f"PSNR======: {psnr_val.item():.4f} dB")
#     print(f"SSIM======: {ssim_val.item():.4f}")
#     print(f"LPIPS=====: {lpips_val.item():.4f}")
#     return {
#         "PSNR": psnr_val.item(),
#         "SSIM": ssim_val.item(),
#         "LPIPS": lpips_val.item()
#     }

def visualize_eval_lam_reconstruction(
    gt_images,          # [gt0, gt1, gt2, gt3, gt4]
    rec_images,         # [rec1, rec2, rec3, rec4]
    latent_motion_ids,  # [id_for_rec1, id_for_rec2, id_for_rec3, id_for_rec4]
    path
):
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    from matplotlib import rcParams
    rcParams['text.usetex']     = False
    rcParams['mathtext.fontset'] = 'stix'

    c, h, w = gt_images[0].shape
    to_pil = T.ToPILImage()

    SCALE = 4
    sh, sw = h * SCALE, w * SCALE

    # ── 标签文字（LaTeX 公式）────────────────────────────────────────
    row1_labels = [
        r"$O_t$  (Ground Truth)",
        r"$O_{t+k}$  (Ground Truth)",
        r"$O_{t+2k}$  (Ground Truth)",
        r"$O_{t+2k}$  (Ground Truth)",
        r"$O_{t+2k}$  (Ground Truth)",
    ]
    row2_labels = [
        r"$O_t$  (Conditioned Input)",
        r"$Dec(O_t,\ \vec{z}_{t}^{\,t+k}) \rightarrow \hat{O}_{t+k}$",
        r"$Dec(O_{t+k},\ \vec{z}_{t+k}^{\,t+2k}) \rightarrow \hat{O}_{t+2k}$",
        r"$Dec(O_t,\ \vec{z}_{t}^{\,t+2k}) \rightarrow \hat{O}_{t+2k}$",
        r"$Dec(O_t,\ \vec{z}_{t}^{\,t+k} + \vec{z}_{t+k}^{\,t+2k}) \rightarrow \hat{O}_{t+2k}$",
    ]

    # ── 颜色 ──────────────────────────────────────────────────────────
    C_GT_BORDER  = '#22 8B22'
    C_REC_BORDER = '#1 95AE6'

    C_GT_BORDER  = (34,  139,  34)
    C_REC_BORDER = (25,   90, 200)

    def to_hex(rgb):
        return '#{:02x}{:02x}{:02x}'.format(*rgb)

    BORDER_W = 8
    N_COLS   = 5
    N_ROWS   = 2

    # ── matplotlib figure 布局 ────────────────────────────────────────
    LABEL_H_IN  = 1.0    # 标签行高度（英寸）
    IMG_H_IN    = h / 72 * SCALE * 0.5
    IMG_W_IN    = w / 72 * SCALE * 0.5

    fig_w = IMG_W_IN * N_COLS
    fig_h = (IMG_H_IN + LABEL_H_IN) * N_ROWS

    fig, axes = plt.subplots(
        N_ROWS * 2, N_COLS,
        figsize=(fig_w, fig_h),
        gridspec_kw={'height_ratios': [IMG_H_IN, LABEL_H_IN,
                                        IMG_H_IN, LABEL_H_IN],
                     'hspace': 0.02,
                     'wspace': 0.02}
    )

    row2_images = [gt_images[0]] + list(rec_images)

    for ci in range(N_COLS):
        # ── 第一行图片 ────────────────────────────────────────────────
        ax_img = axes[0][ci]
        pil_img = to_pil(gt_images[ci]).resize((sw, sh), Image.LANCZOS)
        ax_img.imshow(pil_img)
        ax_img.set_xticks([])
        ax_img.set_yticks([])
        for spine in ax_img.spines.values():
            spine.set_edgecolor(to_hex(C_GT_BORDER))
            spine.set_linewidth(BORDER_W * 0.5)

        # ── 第一行标签 ────────────────────────────────────────────────
        ax_lbl = axes[1][ci]
        ax_lbl.axis('off')
        ax_lbl.text(
            0.5, 0.5, row1_labels[ci],
            ha='center', va='center',
            fontsize=20,
            color=to_hex(C_GT_BORDER),
            transform=ax_lbl.transAxes
        )

        # ── 第二行图片 ────────────────────────────────────────────────
        border_color = C_GT_BORDER if ci == 0 else C_REC_BORDER
        ax_img2 = axes[2][ci]
        pil_img2 = to_pil(row2_images[ci]).resize((sw, sh), Image.LANCZOS)
        ax_img2.imshow(pil_img2)
        ax_img2.set_xticks([])
        ax_img2.set_yticks([])
        for spine in ax_img2.spines.values():
            spine.set_edgecolor(to_hex(border_color))
            spine.set_linewidth(BORDER_W * 0.5)

        # ── 第二行标签 ────────────────────────────────────────────────
        ax_lbl2 = axes[3][ci]
        ax_lbl2.axis('off')
        ax_lbl2.text(
            0.5, 0.5, row2_labels[ci],
            ha='center', va='center',
            fontsize=20,
            color=to_hex(border_color),
            transform=ax_lbl2.transAxes
        )

    plt.savefig(path, dpi=100, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.close(fig)




def visualize_eval_lam_reconstruction_paper(
    gt_images,
    rec_images,
    latent_motion_ids,
    path
):
    import matplotlib.pyplot as plt
    import matplotlib as mpl
    from matplotlib import rcParams

    FONT_SIZE = 60

    rcParams['text.usetex']      = False
    rcParams['mathtext.fontset'] = 'stix'
    rcParams['font.size']        = FONT_SIZE
    rcParams['mathtext.default'] = 'regular'

    c, h, w = gt_images[0].shape
    to_pil = T.ToPILImage()

    SCALE = 4
    sh, sw = h * SCALE, w * SCALE

    row1_labels = [
    r"Source: $\mathit{O_a}$",
    r"$\mathit{O_b}$",
    r"$\mathit{O_c}$",
    r"$\mathit{O_c}$",
    ]
    row2_labels = [
    r"Target: $\mathit{O_{a}}$",
    r"$\widehat{\mathit{O}}^{\,\mathit{b}}_{\mathit{a}}$",
    r"$\widehat{\mathit{O}}^{\,\mathit{c}}_{\mathit{a}}$",
    r"$\widehat{\mathit{O}}^{\,\mathit{b}}_{\mathit{a}} + \widehat{\mathit{O}}^{\,\mathit{c}}_{\mathit{a}}$",
    ]


    C_GT_BORDER  = (34,  139,  34)
    C_REC_BORDER = (25,   90, 200)

    def to_hex(rgb):
        return '#{:02x}{:02x}{:02x}'.format(*rgb)

    BORDER_W = 8
    N_COLS   = 4
    N_ROWS   = 2

    DPI       = 20
    IMG_W_IN  = w * SCALE / DPI
    IMG_H_IN  = h * SCALE / DPI

    # ── 标签行高度紧贴字体 ────────────────────────────────────────────
    # 1 pt = 1/72 inch，hat符号有上标，留1.6倍刚好不裁切
    LINE_HEIGHT_IN = FONT_SIZE / 72.0
    LABEL_H_IN     = LINE_HEIGHT_IN * 1.6   # ← 从2.5收紧到1.6

    fig_w = IMG_W_IN * N_COLS
    fig_h = (IMG_H_IN + LABEL_H_IN) * N_ROWS

    fig, axes = plt.subplots(
        N_ROWS * 2, N_COLS,
        figsize=(fig_w, fig_h),
        gridspec_kw={
            'height_ratios': [IMG_H_IN, LABEL_H_IN,
                              IMG_H_IN, LABEL_H_IN],
            'hspace': 0.0,
            'wspace': 0.0,
        }
    )

    fig.subplots_adjust(left=0, right=1, top=1, bottom=0)

    row1_images = [gt_images[0], gt_images[1], gt_images[3], gt_images[4]]
    row2_images = [rec_images[0], rec_images[1], rec_images[3], rec_images[4]]

    for ci in range(N_COLS):

        # ── Row 1 image ───────────────────────────────────────────────
        ax_img = axes[0][ci]
        pil_img = to_pil(row1_images[ci]).resize((sw, sh), Image.LANCZOS)
        ax_img.imshow(pil_img, aspect='auto')
        ax_img.set_xticks([])
        ax_img.set_yticks([])
        for spine in ax_img.spines.values():
            spine.set_edgecolor(to_hex(C_GT_BORDER))
            spine.set_linewidth(BORDER_W * 0.5)

        # ── Row 1 label ───────────────────────────────────────────────
        ax_lbl = axes[1][ci]
        ax_lbl.axis('off')
        ax_lbl.set_xlim(0, 1)
        ax_lbl.set_ylim(0, 1)
        ax_lbl.text(
            0.5, 0.5, row1_labels[ci],
            ha='center', va='center',
            fontsize=FONT_SIZE,
            color=to_hex(C_GT_BORDER),
            transform=ax_lbl.transAxes,
            linespacing=1.0,   # ← 行间距收紧
        )

        # ── Row 2 image ───────────────────────────────────────────────
        ax_img2 = axes[2][ci]
        pil_img2 = to_pil(row2_images[ci]).resize((sw, sh), Image.LANCZOS)
        ax_img2.imshow(pil_img2, aspect='auto')
        ax_img2.set_xticks([])
        ax_img2.set_yticks([])
        for spine in ax_img2.spines.values():
            spine.set_edgecolor(to_hex(C_REC_BORDER))
            spine.set_linewidth(BORDER_W * 0.5)

        # ── Row 2 label ───────────────────────────────────────────────
        ax_lbl2 = axes[3][ci]
        ax_lbl2.axis('off')
        ax_lbl2.set_xlim(0, 1)
        ax_lbl2.set_ylim(0, 1)
        ax_lbl2.text(
            0.5, 0.5, row2_labels[ci],
            ha='center', va='center',
            fontsize=FONT_SIZE,
            color=to_hex(C_REC_BORDER),
            transform=ax_lbl2.transAxes,
            linespacing=1.0,   # ← 行间距收紧
        )

    plt.savefig(path, dpi=DPI, bbox_inches='tight',
                facecolor='white', edgecolor='none',
                pad_inches=0.05)   # ← 极小留白防止边缘裁切
    plt.close(fig)






def calculate_psl_metrics(img1, img2, device='cuda', log_path=None, tag=""):
    """
    img1, img2: torch.Tensor, shape [B, 3, H, W], range [0, 1]
    """
    img1 = img1.to(device)
    img2 = img2.to(device)

    # 1. PSNR (Higher is better)
    psnr_val = peak_signal_noise_ratio(img1, img2, data_range=1.0)

    # 2. SSIM (Higher is better, range 0~1)
    ssim_val = structural_similarity_index_measure(img1, img2, data_range=1.0)

    # 3. LPIPS (Lower is better, perceptual similarity)
    loss_fn_alex = lpips.LPIPS(net='alex').to(device)
    img1_norm = img1 * 2 - 1
    img2_norm = img2 * 2 - 1
    lpips_val = loss_fn_alex(img1_norm, img2_norm).mean()

    print(f"PSNR======: {psnr_val.item():.4f} dB")
    print(f"SSIM======: {ssim_val.item():.4f}")
    print(f"LPIPS=====: {lpips_val.item():.4f}")

    # 写入日志
    if log_path is not None:
        with open(log_path, "a") as f:
            f.write(f"\n[PSL Metrics] {tag}\n")
            f.write(f"  PSNR  : {psnr_val.item():.4f} dB\n")
            f.write(f"  SSIM  : {ssim_val.item():.4f}\n")
            f.write(f"  LPIPS : {lpips_val.item():.4f}\n")

    return {
        "PSNR": psnr_val.item(),
        "SSIM": ssim_val.item(),
        "LPIPS": lpips_val.item()
    }

def calculate_psl_metrics_dict(gt, recon, log_path=None, tag=""):
    """
    计算 PSNR / SSIM / LPIPS 三个指标。

    Parameters
    ----------
    gt    : Tensor (B, 3, H, W)  值域 [0, 1]
    recon : Tensor (B, 3, H, W)  值域 [0, 1]

    Returns
    -------
    dict  {"psnr": float, "ssim": float, "lpips": float}
    """
    from torchmetrics.image import (
        PeakSignalNoiseRatio,
        StructuralSimilarityIndexMeasure,
    )
    from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    psnr_fn  = PeakSignalNoiseRatio(data_range=1.0).to(device)
    ssim_fn  = StructuralSimilarityIndexMeasure(data_range=1.0).to(device)
    lpips_fn = LearnedPerceptualImagePatchSimilarity(net_type="alex").to(device)

    gt    = gt.to(device).float().clamp(0.0, 1.0)
    recon = recon.to(device).float().clamp(0.0, 1.0)

    psnr_val  = psnr_fn(recon, gt).item()
    ssim_val  = ssim_fn(recon, gt).item()
    # lpips 需要 [-1, 1]
    lpips_val = lpips_fn(
        recon * 2.0 - 1.0,
        gt    * 2.0 - 1.0,
    ).item()

    line = (
        f"[{tag}]  "
        f"PSNR={psnr_val:7.3f} dB  "
        f"SSIM={ssim_val:.4f}  "
        f"LPIPS={lpips_val:.4f}"
    )
    print(line)

    if log_path is not None:
        with open(log_path, "a") as f:
            f.write(line + "\n")

    return {
        "psnr":  psnr_val,
        "ssim":  ssim_val,
        "lpips": lpips_val,
    }



