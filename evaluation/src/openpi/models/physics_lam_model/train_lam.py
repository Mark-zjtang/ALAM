
import argparse
import json
from torch.utils.data import DataLoader
import omegaconf
import hydra
from functools import partial
from transformers import AutoTokenizer
# from uni_world_model.latent_action_model.trainers.latent_action_tokenizer_trainer import LatentActionTokenizerTrainer
# from uni_world_model.latent_action_model.trainers.latent_action_tokenizer_trainer_eval import LatentActionTokenizerTrainer
# from uni_world_model.latent_action_model.trainers.ctl_latent_action_tokenizer_trainer import LatentActionTokenizerTrainer
# from uni_world_model.latent_action_model.trainers.ctl_latent_action_continous_tokenizer_trainer import LatentActionTokenizerTrainer
# from uni_world_model.latent_action_model.trainers.ctl_latent_action_tokenizer_trainer_eval import LatentActionTokenizerTrainer
from uni_world_model.latent_action_model.trainers.ctl_latent_action_continous_tokenizer_trainer_eval import LatentActionTokenizerTrainer


from torch.utils.data import DataLoader
from functools import partial
from uni_world_model.data.data_utils import load_dataset
from uni_world_model.data.img_utils import get_rgb_preprocessor

from IPython import embed

def main(cfg):
    # Prepare Latent Action Tokenizer
    latent_action_tokenizer_config = cfg['latent_action_model_config']
    latent_action_tokenizer = hydra.utils.instantiate(latent_action_tokenizer_config)
    print(f"Initializing Latent Motion Tokenizer ...")
    latent_action_tokenizer.config = latent_action_tokenizer_config
    
    # Prepare rgb_processor
    rgb_preprocessor = get_rgb_preprocessor(**cfg['rgb_preprocessor_config'])

    # Preprepare Dataloaders
    dataset_config_path = cfg['dataset_config']
    extra_data_config = {
        'sequence_length': 1,
        'do_extract_future_frames': True,
        'do_extract_action': False
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
    trainer = LatentActionTokenizerTrainer(
        latent_action_tokenizer=latent_action_tokenizer,
        rgb_preprocessor=rgb_preprocessor,
        train_dataloader=train_dataloader,
        eval_dataloader=eval_dataloader,
        bs_per_gpu=cfg['dataloader_config']['bs_per_gpu'],
        **cfg['training_config']
    )

    # Start Training
    trainer.train()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_path', type=str, default="configs/lam/lam_calvin.yaml")
    args = parser.parse_args()

    cfg = omegaconf.OmegaConf.load(args.config_path)
    main(cfg)



