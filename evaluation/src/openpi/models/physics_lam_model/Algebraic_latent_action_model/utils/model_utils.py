import omegaconf
import hydra
import os
import sys
import torch
from transformers import AutoTokenizer
from transformers.utils import FEATURE_EXTRACTOR_NAME, get_file_from_repo
import json


def load_model(pretrained_path):
    config_path = os.path.join(pretrained_path, "config.yaml")
    checkpoint_path = os.path.join(pretrained_path, "pytorch_model.bin")

    config = omegaconf.OmegaConf.load(config_path)
    model = hydra.utils.instantiate(config)
    model.config = config

    missing_keys, unexpected_keys = model.load_state_dict(torch.load(checkpoint_path), strict=False)
    missing_root_keys = set([k.split(".")[0] for k in missing_keys])
    print('load ', checkpoint_path, '\nmissing ', missing_root_keys, '\nunexpected ', unexpected_keys)

    return model
