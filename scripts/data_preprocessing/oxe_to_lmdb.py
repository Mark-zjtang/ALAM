import pyrootutils
pyrootutils.setup_root(__file__, indicator='.project-root', pythonpath=True, dotenv=True)
import os
import io
import argparse
import lmdb
from pickle import dumps, loads
import numpy as np
import torch
from torchvision.transforms.functional import resize
from torchvision.io import encode_jpeg
import tensorflow_datasets as tfds
from tqdm import tqdm
from data_preprocessing.oxe_dataset_configs import OXE_DATASET_CONFIGS


def get_dataset_path(parent_dir, dataset_name):
    if dataset_name in ['robo_net', 'cmu_playing_with_food', 'bridge_dataset', 'droid']:
        version = '1.0.0'
    elif dataset_name == 'language_table':
        version = '0.0.1'
    elif dataset_name[:-1] == 'uiuc_d3field' or dataset_name[:-1] == 'stanford_robocook_converted_externally_to_rlds':
        dataset_name = dataset_name[:-1]
        version = '0.1.0'
    else:
        version = '0.1.0'
    return os.path.join(parent_dir, dataset_name, version)


def save_to_lmdb(output_dir, input_dir, dataset_name):
    oxe_dataset_config = OXE_DATASET_CONFIGS[dataset_name]
    image_obs_keys = [v for k, v in oxe_dataset_config['image_obs_keys'].items() if v is not None]
    depth_obs_keys = [v for k, v in oxe_dataset_config['depth_obs_keys'].items() if v is not None]
    exclude_obs_keys = image_obs_keys + depth_obs_keys + ['natural_language_embedding']

    ds = tfds.builder_from_directory(builder_dir=get_dataset_path(input_dir, dataset_name)).as_dataset()
    for split in ds:
        print(f"processing {dataset_name}-{split} ...")
        target_dir = os.path.join(output_dir, dataset_name, split)
        if not os.path.exists(target_dir):
            os.makedirs(target_dir)

        env = lmdb.open(target_dir, map_size=int(3e12), readonly=False, lock=False) # maximum size of memory map is 3TB
        with env.begin(write=True) as txn:
            if txn.get('cur_step'.encode()) is not None:
                cur_step = loads(txn.get('cur_step'.encode())) + 1
                cur_episode = loads(txn.get(f'cur_episode_{cur_step - 1}'.encode())) + 1
            else:
                cur_step = 0
                cur_episode = 0

            i = 0
            for episode in tqdm(iter(ds[split]), total=len(ds[split])):
                for j, step in enumerate(episode["steps"].as_numpy_iterator()):
                    if i < cur_step:
                        i += 1
                        print(f"skipping step-{i}!!")
                        continue
                    else:
                        i += 1

                    for k in exclude_obs_keys:
                        if k in step['observation']:
                            step['observation'].pop(k)
                    
                    if dataset_name == "berkeley_fanuc_manipulation":
                        step['observation']['natural_language_instruction'] = step['language_instruction'].decode().strip('. ').lower().encode()
                        step['observation']['action'] = step['action']
                    

                    txn.put('cur_step'.encode(), dumps(cur_step))
                    txn.put(f'cur_episode_{cur_step}'.encode(), dumps(cur_episode))
                    txn.put(f'local_step_{cur_step}'.encode(), dumps(j))
                    txn.put(f'feature_dict_{cur_step}'.encode(), dumps(step))
                    cur_step += 1
                cur_episode += 1
        env.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", default='/group/40101/public_datasets/open-x-embodiment/tensorflow_datasets', type=str, help="Original root dataset directory.")
    parser.add_argument("--output_dir", default='/group/40101/public_datasets/open-x-embodiment/lmdb_datasets', type=str, help="Output root dataset directory.")
    parser.add_argument("--dataset_name", default='fractal20220817_data', type=str, help="Dataset name.")
    args = parser.parse_args()
    save_to_lmdb(args.output_dir, args.input_dir, args.dataset_name)
