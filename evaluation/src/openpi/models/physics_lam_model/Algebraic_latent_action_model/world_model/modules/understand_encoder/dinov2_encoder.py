# Copyright (c) 2023, Zexin He
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import torch
import torch.nn as nn
from transformers import AutoImageProcessor, AutoModel
from transformers import Dinov2Backbone
from torchvision import transforms
from einops import rearrange, repeat
from IPython import embed

class DinoV2Encoder(nn.Module):
    """
    Dino v2 wrapper using huggingface transformer implementation.
    """
    def __init__(self, pretrained_model_name_or_path: str):
        super().__init__()
        self.model = self._build_dino(pretrained_model_name_or_path)

    def forward(self, image):
        # image: [B, C, H, W]
        # RGB image with [0,1] scale and properly sized
        outputs = self.model(image)
        image_embeds = outputs.last_hidden_state
        ## torch.Size([16, 257, 1536])
        return image_embeds

    @staticmethod
    def _build_dino(model_name: str, proxy_error_retries: int = 3, proxy_error_cooldown: int = 5):
        import requests
        try:
            model = AutoModel.from_pretrained(model_name)
            return model
        except requests.exceptions.ProxyError as err:
            if proxy_error_retries > 0:
                print(f"Huggingface ProxyError: Retrying in {proxy_error_cooldown} seconds...")
                import time
                time.sleep(proxy_error_cooldown)
                return DinoV2Wrapper._build_dino(model_name, proxy_error_retries - 1, proxy_error_cooldown)
            else:
                raise err
