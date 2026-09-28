#!/usr/bin/env python3
"""
Precompute latent-world vectors (w_true) or latent token ids for a dataset.

Usage:
  python scripts/precompute_w.py --config configs/...yaml --out_dir /path/to/out --mode latent_vectors --batch_size 8

Modes:
  token_ids      - save discrete latent action token ids returned by the tokenizer
  latent_vectors - run the world model (with token ids) to obtain continuous latent_action_feature

Outputs: per-sample .npz files saved under `out_dir` containing keys:
  - w: float32 array (if mode latent_vectors)
  - latent_ids: int array (if mode token_ids or latent_vectors)
  - actions: optional action array from dataset
  - idx: dataset index

"""
import os
import argparse
import omegaconf
import hydra
import torch
from torch.utils.data import DataLoader
from functools import partial
import numpy as np
from transformers import AutoTokenizer
from uni_world_model.data.data_utils import load_dataset
from uni_world_model.data.img_utils import get_rgb_preprocessor
from uni_world_model.utils.model_utils import load_model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True)
    parser.add_argument('--out_dir', type=str, required=True)
    parser.add_argument('--mode', type=str, default='latent_vectors', choices=['token_ids', 'latent_vectors'])
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--max_samples', type=int, default=None)
    args = parser.parse_args()

    cfg = omegaconf.OmegaConf.load(args.config)
    diffusion_policy_config = cfg['diffusion_policy_config']

    os.makedirs(args.out_dir, exist_ok=True)

    # instantiate diffusion_policy to get world_model and tokenizer settings
    diffusion_policy = hydra.utils.instantiate(diffusion_policy_config)
    diffusion_policy.config = diffusion_policy_config
    world_model = diffusion_policy.world_model

    # tokenizer used to obtain latent action token ids from frame pairs
    latent_action_tokenizer = None
    if diffusion_policy_config['world_model']['latent_action_pred']:
        latent_action_tokenizer_path = cfg.get('latent_action_tokenizer_path', None)
        if latent_action_tokenizer_path is not None:
            latent_action_tokenizer = load_model(latent_action_tokenizer_path)
        else:
            # if tokenizer not provided, fall back to vector-quantizer in world_model if available
            latent_action_tokenizer = None

    # language tokenizer for world_model language encoder
    lang_tokenizer = AutoTokenizer.from_pretrained(diffusion_policy_config['world_model']['model_lang']['pretrained_model_name_or_path'])

    # rgb preprocessor
    rgb_preprocessor = get_rgb_preprocessor(**cfg['rgb_preprocessor_config'])

    # prepare dataset
    extra_data_config = {
        'sequence_length': diffusion_policy_config['world_model']['sequence_length'],
        'act_dim': diffusion_policy_config['act_dim'],
        'chunk_size': diffusion_policy_config['world_model']['chunk_size'],
        'do_extract_future_frames': diffusion_policy_config['world_model']['latent_action_pred'],
        'do_extract_action': diffusion_policy_config['act_pred']
    }

    train_dataset, eval_dataset = load_dataset(cfg['dataset_config'], extra_data_config)
    dataset = train_dataset

    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=cfg.get('dataloader_config', {}).get('workers_per_gpu', 4), collate_fn=None)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    world_model = world_model.to(device)
    world_model.eval()
    if latent_action_tokenizer is not None:
        latent_action_tokenizer = latent_action_tokenizer.to(device)
        latent_action_tokenizer.eval()

    sample_count = 0
    global_idx = 0
    with torch.no_grad():
        for batch in dataloader:
            # dataset items expected to contain 'rgb_initial' (1,c,h,w) and 'rgb_future' (T,c,h,w)
            # and optionally 'actions' and 'lang'
            # batch may be a list of dicts when default collate_fn is None
            if isinstance(batch, list):
                # collate manually
                batch_size = len(batch)
                lang_list = [b.get('lang', '') for b in batch]
                rgb_initial = torch.stack([b['rgb_initial'][0] for b in batch], dim=0)  # (B, c, h, w)
                rgb_future = torch.stack([b['rgb_future'] for b in batch], dim=0)  # (B, T, c, h, w)
                actions = torch.stack([b['actions'] for b in batch], dim=0) if 'actions' in batch[0] else None
                latent_mask = torch.stack([b['latent_mask'] for b in batch], dim=0) if 'latent_mask' in batch[0] else None
                idxs = [b.get('idx', i+global_idx) for i, b in enumerate(batch)]
            else:
                # if DataLoader used default collate, it may already be tensors
                lang_list = batch.get('lang', [])
                rgb_initial = batch['rgb_initial'][:,0]
                rgb_future = batch['rgb_future']
                actions = batch.get('actions', None)
                latent_mask = batch.get('latent_mask', None)
                idxs = batch.get('idx', list(range(global_idx, global_idx + rgb_initial.shape[0])))

            B = rgb_initial.shape[0]
            # prepare rgb_seq like trainer: initial + first future frame
            if rgb_future is not None:
                rgb_seq = torch.cat([rgb_initial.unsqueeze(1), rgb_future[:, :1]], dim=1)  # (B, 2, c, h, w)
            else:
                rgb_seq = rgb_initial.unsqueeze(1)

            # preprocess images
            rgb_seq_proc = []
            for b in range(B):
                frames = [rgb_seq[b, i] for i in range(rgb_seq.shape[1])]
                frames = torch.stack([rgb_preprocessor(f, train=False) if callable(rgb_preprocessor) else f for f in frames], dim=0)
                rgb_seq_proc.append(frames)
            rgb_seq_proc = torch.stack(rgb_seq_proc, dim=0)  # (B, 2, c, h, w)

            # tokenise language
            lang_inputs = lang_tokenizer(lang_list, return_tensors='pt', padding=True)
            lang_input_ids = lang_inputs.input_ids

            # prepare cond/target images as numpy arrays expected by tokenizer
            c, h, w = rgb_seq_proc.shape[2:]
            cond_pixels = rgb_seq_proc[:, :-1].reshape(-1, c, h, w)
            target_pixels = rgb_seq_proc[:, 1:].reshape(-1, c, h, w)

            # move to device if needed
            cond_pixels = cond_pixels.to(device)
            target_pixels = target_pixels.to(device)
            lang_input_ids = lang_input_ids.to(device)

            # obtain latent_action token ids
            if latent_action_tokenizer is not None:
                # tokenizer API used in training: return_action_token_ids_only=True
                token_ids = latent_action_tokenizer(
                    cond_pixel_values=cond_pixels,
                    target_pixel_values=target_pixels,
                    return_action_token_ids_only=True
                ).reshape(B, -1)
            else:
                token_ids = None

            # if mode latent_vectors, run world_model forward to get latent_action_feature
            w_feats = None
            if args.mode == 'latent_vectors':
                if token_ids is None:
                    raise RuntimeError('latent_action_tokenizer is required for latent_vectors mode')
                # run world_model: it expects rgb (B,1,c,h,w) and latent_action_ids
                rgb_initial_proc = rgb_seq_proc[:, :1].to(device)
                out = world_model(rgb=rgb_initial_proc, language=lang_input_ids, latent_action_ids=token_ids, train=False)
                w_feats = out['latent_action_feature'].cpu().numpy()

            # save per-sample
            for i in range(B):
                sample_idx = idxs[i] if isinstance(idxs, (list, tuple)) else int(global_idx + i)
                fname = os.path.join(args.out_dir, f'sample_{sample_idx}.npz')
                tosave = {}
                if token_ids is not None:
                    tosave['latent_ids'] = token_ids[i].cpu().numpy()
                if w_feats is not None:
                    tosave['w'] = w_feats[i]
                if actions is not None:
                    tosave['actions'] = actions[i].cpu().numpy()
                if latent_mask is not None:
                    tosave['latent_mask'] = latent_mask[i].cpu().numpy()
                np.savez_compressed(fname, **tosave)

            sample_count += B
            global_idx += B
            if args.max_samples is not None and sample_count >= args.max_samples:
                break

    print(f'Wrote {sample_count} samples to {args.out_dir}')

if __name__ == '__main__':
    main()
