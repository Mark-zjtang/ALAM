#!/usr/bin/env python3
import subprocess

# 卸载 apex 库（如果已安装）
subprocess.run(["pip", "uninstall", "-y", "apex"])

import argparse
import json
from torch.utils.data import DataLoader
import omegaconf
import hydra
from functools import partial
from transformers import AutoTokenizer, AutoProcessor
from uni_world_model.world_model.trainers.world_model_trainer import WorldModel_Trainer
from torch.utils.data import DataLoader
from functools import partial
from uni_world_model.data.data_utils import load_dataset
from uni_world_model.data.img_utils import get_rgb_preprocessor
from uni_world_model.utils.model_utils import load_model
from IPython import embed

def main(cfg):
    # Prepare WorldModel
    world_model_config = cfg['world_model_config']
    world_model = hydra.utils.instantiate(world_model_config)
    world_model.config = world_model_config

    # Prepare lang_tokenizer and rgb_processor
    lang_tokenizer = AutoTokenizer.from_pretrained(world_model_config['model_lang']['pretrained_model_name_or_path'])
    rgb_preprocessor = get_rgb_preprocessor(**cfg['rgb_preprocessor_config'])

    # Prepare Latent Action Tokenizer
    if world_model_config['latent_action_pred']:
        latent_action_tokenizer_path = cfg['latent_action_tokenizer_path']
        assert latent_action_tokenizer_path is not None
        print(f"loading Latent Action Tokenizer from {latent_action_tokenizer_path} ...")
        latent_action_tokenizer = load_model(latent_action_tokenizer_path)
    else:
        latent_action_tokenizer = None

    vq_remap = cfg.get('vq_remap', None)
    unknown_index = cfg.get('unknown_index', 'closest')
    if vq_remap is not None:
        latent_action_tokenizer.vector_quantizer.setup_remap(remap=vq_remap, unknown_index=unknown_index)

    # Preprepare Dataloaders
    dataset_config_path = cfg['dataset_config']
    
    extra_data_config = {
        'sequence_length': world_model_config['sequence_length'],
        'chunk_size': world_model_config['chunk_size'],
        'do_extract_future_frames': world_model_config['latent_action_pred'],
    }

    train_dataset, eval_dataset = load_dataset(dataset_config_path, extra_data_config)
    dataloader_cls = partial(
        DataLoader, 
        pin_memory=True, # Accelerate data reading
        shuffle=True,
        persistent_workers=True,
        num_workers=cfg['dataloader_config']['workers_per_gpu'],
        batch_size=cfg['dataloader_config']['bs_per_gpu'],
        prefetch_factor= cfg['dataloader_config']['prefetch_factor']
    )
    train_dataloader = dataloader_cls(train_dataset)
    eval_dataloader = dataloader_cls(eval_dataset)
    
    # Prepare Trainer
    trainer = WorldModel_Trainer(
        world_model=world_model,
        world_model_config=world_model_config,
        latent_action_tokenizer=latent_action_tokenizer,
        rgb_preprocessor=rgb_preprocessor,
        lang_tokenizer=lang_tokenizer,
        train_dataloader=train_dataloader,
        eval_dataloader=eval_dataloader,
        bs_per_gpu=cfg['dataloader_config']['bs_per_gpu'],
        **cfg['training_config']
    )

    # Start Training
    trainer.train()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_path', type=str, default="/group/40101/milkcychen/Moto/moto_gpt/configs/train/data_calvin-model_actPredTrue_actionPredTrue_visionMaeLarge_seq2_chunk5_maskProb0.5-train_lr0.0002_bs512-aug_shiftTrue_resizedCropFalse-resume_from_predLatentOnly_calvin_Epoch10.yaml")
    args = parser.parse_args()

    cfg = omegaconf.OmegaConf.load(args.config_path)
    main(cfg)

    


