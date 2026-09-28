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



def visualize_eval_lam_reconstruction(
    gt_images,          #  [gt0, gt1, gt2, gt3, gt4]
    rec_images,         #  [rec1, rec2, rec3, rec4]
    latent_motion_ids,  #  [id_for_rec1, id_for_rec2, id_for_rec3, id_for_rec4]
    path
):
    c, h, w = gt_images[0].shape

    row1_padding = 30
    row2_padding = 45
    
    to_pil = T.ToPILImage()

    ids_list = None
    if latent_motion_ids is not None:
        if torch.is_tensor(latent_motion_ids):
            ids_list = latent_motion_ids.cpu().numpy().tolist()
        else:
            ids_list = list(latent_motion_ids)
   
    canvas_w = 5 * w
    canvas_h = h + row1_padding + h + row2_padding
    compare_img = Image.new('RGB', size=(canvas_w, canvas_h), color=(0, 0, 0))
    draw = ImageDraw.Draw(compare_img)

    try:
        font_path = os.path.join(cv2.__path__[0], 'qt', 'fonts', 'DejaVuSans.ttf')
        font = ImageFont.truetype(font_path, size=12)
    except:
        font = ImageFont.load_default()

    green = (0, 255, 0)

    # --- 第一行绘制 (Ground Truth) ---
    row1_labels = [
        "ground_truth image 0", 
        "ground_truth image 1", 
        "ground_truth image 2", 
        "ground_truth image 2",
        "ground_truth image 2_additivity"
    ]
    
    for i in range(5):
        x = i * w
        y = 0
        compare_img.paste(to_pil(gt_images[i]), box=(x, y))
        draw.text((x + 5, h + 5), row1_labels[i], font=font, fill=green)

    row2_images = [gt_images[0]] + rec_images
    row2_labels = [
        "ground_truth image 0", 
        "input_gt image 0-add_phys_a_01", 
        "input_gt image 0-add_phys_a_02", 
        "input_gt image 1-add_phys_a_12", 
        "input_gt_image 0-add_phys_a_(01_add_12)"
    ]
    
    row2_y_start = h + row1_padding 
    
    for i in range(5):
        x = i * w
        y = row2_y_start
        compare_img.paste(to_pil(row2_images[i]), box=(x, y))
        label_y = y + h + 5
        draw.text((x + 5, label_y), row2_labels[i], font=font, fill=green)

        if i > 0 and ids_list is not None:
            if (i-1) < len(ids_list):
                current_id = ids_list[i-1] 
                ids_y = label_y + 15
                draw.text((x + 5, ids_y), f"ID: {current_id}", font=font, fill=green)
        

    compare_img.save(path)



def calculate_psl_metrics(img1, img2, device='cuda'):
    """
    img1, img2: torch.Tensor, shape [B, 3, H, W], range [0, 1]
    """
    img1 = img1.to(device)
    img2 = img2.to(device)

    # 1. PSNR (Higher is better)
    # data_range=1.0 表示输入是 [0, 1]
    psnr_val = peak_signal_noise_ratio(img1, img2, data_range=1.0)

    # 2. SSIM (Higher is better, range 0~1)
    ssim_val = structural_similarity_index_measure(img1, img2, data_range=1.0)

    # 3. LPIPS (Lower is better, perceptual similarity)
    # 注意：LPIPS 默认期望输入范围是 [-1, 1]
    # 我们需要将 [0, 1] 转换为 [-1, 1]
    loss_fn_alex = lpips.LPIPS(net='alex').to(device) # 或者用 'vgg'
    
    img1_norm = img1 * 2 - 1
    img2_norm = img2 * 2 - 1
    
    # lpips 返回的是每个样本的距离，取平均值
    lpips_val = loss_fn_alex(img1_norm, img2_norm).mean()
    print(f"PSNR======: {psnr_val.item():.4f} dB")
    print(f"SSIM======: {ssim_val.item():.4f}")
    print(f"LPIPS=====: {lpips_val.item():.4f}")
    return {
        "PSNR": psnr_val.item(),
        "SSIM": ssim_val.item(),
        "LPIPS": lpips_val.item()
    }



