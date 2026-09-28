import pyrootutils
pyrootutils.setup_root(__file__, indicator='.project-root', pythonpath=True, dotenv=True)
import omegaconf
import hydra
import os
import sys
import torch
from uni_world_model.robot_manipulation_benchmarks.uni_world_policy_wraper import UniWorld_PolicyWraper
from transformers import AutoTokenizer
from transformers.utils import FEATURE_EXTRACTOR_NAME, get_file_from_repo
from uni_world_model.data.img_utils import get_model_vision_basic_config
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

def load_moto_gpt_policy(args):
    print(f"loading UniWorld from {args.moto_gpt_path} ...")
    moto_gpt = load_model(args.moto_gpt_path)
    moto_gpt.mask_latent_motion_probability = args.mask_latent_motion_probability
    moto_gpt.config['mask_latent_motion_probability'] = args.mask_latent_motion_probability
    moto_gpt_config = moto_gpt.config

    lang_tokenizer = AutoTokenizer.from_pretrained(moto_gpt_config['world_model']['model_lang']['pretrained_model_name_or_path'])
    # vision_processor_config = json.load(open(get_file_from_repo(moto_gpt_config['model_vision']['pretrained_model_name_or_path'], FEATURE_EXTRACTOR_NAME)))
    # rgb_shape = [vision_processor_config['size'], vision_processor_config['size']]
    # rgb_mean = vision_processor_config['image_mean']
    # rgb_std = vision_processor_config['image_std']
    model_vision_basic_config = get_model_vision_basic_config(moto_gpt_config['world_model']['model_vision']['pretrained_model_name_or_path'])


    variant = {
        'test_chunk_size': args.test_chunk_size,
        'is_gripper_binary': args.is_gripper_binary,
        'use_temporal_ensemble': args.use_temporal_ensemble,

        # 'rgb_shape': rgb_shape,
        # 'rgb_mean': rgb_mean,
        # 'rgb_std': rgb_mean,

        'num_timestep_buckets': moto_gpt_config['num_timestep_buckets'],
        'act_dim': moto_gpt_config['act_dim'],
        'seq_len': moto_gpt_config['world_model']['sequence_length'],
        'chunk_size': moto_gpt_config['world_model']['chunk_size'],
        'mask_latent_motion_probability': 0,
        'latent_action_pred': moto_gpt_config['world_model']['latent_action_pred'],
        'per_latent_action_len': moto_gpt_config['world_model']['per_latent_action_len'],
        'pred_discrete_arm_action': moto_gpt_config.get('pred_discrete_arm_action', False) # NOTE 2024/12/17: predict discrete arm actions for berkeley_fanuc_manipulation
    }
    variant.update(model_vision_basic_config)

    latent_motion_decoding_kwargs = {
        'temperature': args.temperature, 
        'sample': args.sample, 
        'top_k': args.top_k, 
        'top_p': args.top_p,
        'beam_size': args.beam_size, 
        'parallel': args.parallel
    }

    eva = UniWorld_PolicyWraper(
        policy=moto_gpt,
        variant=variant,
        latent_motion_decoding_kwargs=latent_motion_decoding_kwargs,
        lang_tokenizer=lang_tokenizer
    )

    return eva