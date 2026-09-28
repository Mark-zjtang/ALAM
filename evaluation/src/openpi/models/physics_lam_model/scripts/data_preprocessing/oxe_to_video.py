import pyrootutils
pyrootutils.setup_root(__file__, indicator='.project-root', pythonpath=True, dotenv=True)

import numpy as np
import os
from tqdm import tqdm
import argparse
import tensorflow_datasets as tfds

from data_preprocessing.oxe_utils import DISPLAY_KEY
import cv2
import concurrent.futures
import json

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


def write_video(frames, output_file, fps=30, convert_rgb=True):
    try:
        print(f"Processing {output_file}")
        
        height, width, _ = frames[0].shape
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_file, fourcc, fps, (width, height))
        
        for frame in frames:
            if convert_rgb:
                out.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
            else:
                out.write(frame)
        
        out.release()
        print(f"Finished {output_file} (num_frames = {len(frames)})")
    except Exception as e:
        print(f"Error processing {output_file}: {e} (num_frames = {len(frames)})")



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_name', type=str, default='fractal20220817_data')
    parser.add_argument('--input_path', type=str, default='/group/40101/public_datasets/open-x-embodiment/tensorflow_datasets')
    parser.add_argument('--output_path', type=str, default='/group/40101/public_datasets/open-x-embodiment/video_datasets')
    parser.add_argument('--max_num_episodes', default=-1, type=int)
    parser.add_argument('--fps', type=int, default=30)
    parser.add_argument('--convert_rgb', type=int, default=1)

    args = parser.parse_args()

    dataset_name = args.dataset_name
    ds = tfds.builder_from_directory(builder_dir=get_dataset_path(args.input_path, dataset_name)).as_dataset()
    display_key = DISPLAY_KEY.get(dataset_name, 'image')
    root_path = os.path.join(args.output_path, dataset_name, display_key)
    os.makedirs(root_path, exist_ok=True)

    num_episodes = 0
    video_metadata = {}
    with concurrent.futures.ProcessPoolExecutor() as executor:
        for key in ds:
            print(key)
            videos = []
            total_frames = 0

            bar = tqdm(enumerate(ds[key]), total=len(ds[key]))
            for i, episode in bar:
                output_file = os.path.join(os.path.join(root_path, f'{key}_eps_{i:08d}.mp4'))
                try:
                    frames = np.array([step['observation'][display_key] for step in episode['steps']])

                    episode_len = len(frames)
                    if episode_len >= 2:
                        total_frames += episode_len
                        videos.append((f'{key}_eps_{i:08d}.mp4', episode_len))

                    bar.set_postfix(epslen=episode_len)
                    executor.submit(write_video, frames, output_file, args.fps, args.convert_rgb)
                    num_episodes += 1
                    if (args.max_num_episodes>0) and (num_episodes>=args.max_num_episodes):
                        break
                except Exception as e:
                    print(e)
                    continue

            video_metadata[key] = {
                'total_frames': total_frames,
                'videos': videos
            }

with open(os.path.join(root_path, 'video_metadata.json'), 'w') as f:
    json.dump(video_metadata, f)


print("All tfds data have been converted to video.")