import torch.nn.functional as F
import math
import cv2
from PIL import Image, ImageFont, ImageDraw
import os
import torchvision.transforms as T
import numpy as np

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
