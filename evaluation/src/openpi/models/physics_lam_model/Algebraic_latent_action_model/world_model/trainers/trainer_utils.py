import torch.nn.functional as F
import math
import cv2
from PIL import Image, ImageFont, ImageDraw
import os
import torchvision.transforms as T
import numpy as np

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