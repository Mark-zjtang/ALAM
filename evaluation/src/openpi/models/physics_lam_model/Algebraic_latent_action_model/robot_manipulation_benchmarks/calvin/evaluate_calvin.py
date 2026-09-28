# MIT License

# Copyright (c) 2021 Oier Mees
# Copyright (c) 2024 Bytedance Ltd. and/or its affiliates

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

"""Code to evaluate Calvin."""
import pyrootutils
import os
import sys
pyrootutils.setup_root(__file__, indicator='.project-root', pythonpath=True, dotenv=True)
from uni_world_model.robot_manipulation_benchmarks.model_utils import load_moto_gpt_policy

from omegaconf import OmegaConf
import hydra
import argparse
import json
import numpy as np
import logging
from pathlib import Path
import time
import copy
from moviepy.editor import ImageSequenceClip
from accelerate import Accelerator
from datetime import timedelta
from accelerate.utils import InitProcessGroupKwargs


from calvin_agent.evaluation.multistep_sequences import get_sequences
from calvin_agent.evaluation.utils import (
    count_success,
    get_env_state_for_initial_condition,
    get_log_dir,
)
from calvin_utils import print_and_save

from termcolor import colored
import torch
from tqdm.auto import tqdm
from transformers import set_seed

from PIL import Image


logger = logging.getLogger(__name__)

os.environ["FFMPEG_BINARY"] = "auto-detect"
CALVIN_ROOT = os.environ['CALVIN_ROOT']

def make_env(dataset_path, observation_space, device):
    val_folder = Path(dataset_path) / "validation"
    from calvin_env_wrapper_raw import CalvinEnvWrapperRaw
    env = CalvinEnvWrapperRaw(val_folder, observation_space, device)
    return env


def evaluate_policy(model, env, eval_sr_path, eval_result_path, ep_len, num_sequences, num_procs, procs_id, eval_dir=None, debug=False):
    conf_dir = Path(f"{CALVIN_ROOT}/calvin_models") / "conf"
    task_cfg = OmegaConf.load(conf_dir / "callbacks/rollout/tasks/new_playtable_tasks.yaml")
    task_oracle = hydra.utils.instantiate(task_cfg)
    val_annotations = OmegaConf.load(conf_dir / "annotations/new_playtable_validation.yaml")
    eval_dir = get_log_dir(eval_dir)
    eval_sequences = get_sequences(num_sequences)
    num_seq_per_procs = num_sequences // num_procs
    eval_sequences = eval_sequences[num_seq_per_procs*procs_id:num_seq_per_procs*(procs_id+1)]

    import random
    # 设置随机种子（例如 seed=42）
    random.seed(42)  
    # 从 0-999 中随机选取 100 个不重复的整数
    sequence_list = random.sample(range(1000), 100)

    # 使用这些索引从 eval_sequences 中提取对应的元素
    eval_sequences = [eval_sequences[i] for i in sequence_list]

    results = []
    if not debug:
        eval_sequences = tqdm(eval_sequences, position=0, leave=True)

    sequence_i = 0
    for initial_state, eval_sequence in eval_sequences:
        result = evaluate_sequence(env, model, task_oracle, initial_state, eval_sequence, val_annotations, debug, eval_dir, sequence_i, ep_len)
        results.append(result)
        if not debug:
            success_list = count_success(results)
            with open(eval_sr_path, 'a') as f:
                line =f"{sequence_i}/{num_sequences}: "
                for sr in success_list:
                    line += f"{sr:.3f} | "
                sequence_i += 1
                line += "\n"
                f.write(line)
            eval_sequences.set_description(
                " ".join([f"{i + 1}/5 : {v * 100:.1f}% |" for i, v in enumerate(success_list)]) + "|"
            )
        else:
            sequence_i += 1
    print_and_save(results, eval_sequences, eval_result_path, None)
    return results


def evaluate_sequence(env, model, task_checker, initial_state, eval_sequence, val_annotations, debug, eval_dir, sequence_i, ep_len):
    robot_obs, scene_obs = get_env_state_for_initial_condition(initial_state)
    env.reset(robot_obs=robot_obs, scene_obs=scene_obs)
    success_counter = 0
    if debug:
        time.sleep(1)
        print()
        print()
        print(f"Evaluating sequence: {' -> '.join(eval_sequence)}")
        print("Subtask: ", end="")
    for subtask_i, subtask in enumerate(eval_sequence):
        success = rollout(env, model, task_checker, subtask, val_annotations, debug, eval_dir, subtask_i, sequence_i, ep_len)
        if success:
            success_counter += 1
        else:
            return success_counter
    return success_counter


def rollout(env, model, task_oracle, subtask, val_annotations, debug, eval_dir, subtask_i, sequence_i, ep_len):
    if debug:
        print(f"{subtask} ", end="")
        time.sleep(0.5)
    obs = env.get_obs()
    lang_annotation = val_annotations[subtask][0]
    model.reset()
    start_info = env.get_info()
    if debug:
        img_list = []
    unfinished = 0
    for step in range(ep_len):
        if unfinished == 0:
            action = model.step(obs, lang_annotation)
            unfinished = action.shape[0]
        obs, _, _, current_info = env.step(action[-unfinished])
        unfinished -= 1
        if debug:
            img_copy = copy.deepcopy(obs['rgb_obs']['rgb_static'])
            img_list.append(img_copy)
        # check if current step solves a task
        current_task_info = task_oracle.get_task_info_for_set(start_info, current_info, {subtask})
        if len(current_task_info) > 0:
            if debug:
                print(colored("success", "green"), end=" ")
                # 保存 GIF
                gif_filename = f'{sequence_i}-{subtask_i}-{subtask}-succ.gif'
                gif_path = os.path.join(eval_dir, gif_filename)
                clip = ImageSequenceClip(img_list, fps=30)
                clip.write_gif(gif_path, fps=30)
                
                # # 同时创建一个与 GIF 同名（仅去除扩展名）的文件夹保存图片
                # img_folder = os.path.join(eval_dir, f'{sequence_i}-{subtask_i}-{subtask}-succ')
                # os.makedirs(img_folder, exist_ok=True)
                # # 每隔 5 帧保存一张图片
                # for idx, img in enumerate(img_list):
                #     if idx % 5 == 0:
                #         # 如果 img 是 numpy array，则转换为 PIL.Image 对象
                #         pil_img = Image.fromarray(img) if not isinstance(img, Image.Image) else img
                #         image_filename = f'image_{idx:05d}.png'
                #         pil_img.save(os.path.join(img_folder, image_filename))
            return True
    if debug:
        print(colored("fail", "red"), end=" ")
        # 保存失败 GIF
        gif_filename = f'{sequence_i}-{subtask_i}-{subtask}-fail.gif'
        gif_path = os.path.join(eval_dir, gif_filename)
        clip = ImageSequenceClip(img_list, fps=30)
        clip.write_gif(gif_path, fps=30)
        
        # # 同时创建同名图片文件夹保存每隔 5 帧的图片
        # img_folder = os.path.join(eval_dir, f'{sequence_i}-{subtask_i}-{subtask}-fail')
        # os.makedirs(img_folder, exist_ok=True)
        # for idx, img in enumerate(img_list):
        #     if idx % 5 == 0:
        #         pil_img = Image.fromarray(img) if not isinstance(img, Image.Image) else img
        #         image_filename = f'image_{idx:05d}.png'
        #         pil_img.save(os.path.join(img_folder, image_filename))
    return False

def main(args):
    print(args)
    kwargs = InitProcessGroupKwargs(timeout=timedelta(seconds=3600))
    acc = Accelerator(kwargs_handlers=[kwargs])
    device = acc.device
    
    args.is_gripper_binary = True
    eva = load_moto_gpt_policy(args)
    eva.policy = acc.prepare(eva.policy, device_placement=[True])
    eva.policy.eval()


    # Prepare CALVIN Environment
    observation_space = {
        'rgb_obs': ['rgb_static'], 
        'depth_obs': [], 
        'state_obs': ['robot_obs'], 
        'actions': ['rel_actions'], 
        'language': ['language']}

    try:
        eval_dir = os.path.join(args.eval_dir, f'eval{torch.cuda.current_device()}/')
    except:
        eval_dir = os.path.join(args.eval_dir, 'eval')
    os.makedirs(eval_dir, exist_ok=True)
    env = make_env('fake_dataset', observation_space, device)
    acc.print(f"initialize CALVIN environment")

    # Evaluation
    avg_reward = torch.tensor(evaluate_policy(
        eva, 
        env,
        os.path.join(args.eval_dir,'success_rate.txt'),
        os.path.join(args.eval_dir,'result.txt'),
        args.ep_len,
        args.num_sequences,
        acc.num_processes,
        acc.process_index,
        eval_dir,
        debug=args.record_evaluation_video,
    )).float().mean().to(device)
    acc.wait_for_everyone()
    avg_reward = acc.gather_for_metrics(avg_reward).mean()
    if acc.is_main_process:
        print('average success rate ', avg_reward)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--moto_gpt_path', type=str, required=True)
    parser.add_argument('--mask_latent_motion_probability', type=float, default=1.0)
    
    parser.add_argument('--temperature', type=float, default=1.0)
    parser.add_argument('--sample', type=json.loads, default='true')
    parser.add_argument('--top_k', type=int, default=0)
    parser.add_argument('--top_p', type=float, default=1.0)
    parser.add_argument('--beam_size', type=int, default=5)
    parser.add_argument('--parallel', type=json.loads, default='false')

    parser.add_argument('--test_chunk_size', type=int, default=10)
    parser.add_argument('--use_temporal_ensemble', type=json.loads, default='false')

    parser.add_argument('--num_sequences', type=int, default=1000)
    parser.add_argument('--ep_len', type=int, default=360)
    parser.add_argument('--eval_dir', type=str, required=True)
    parser.add_argument('--record_evaluation_video', action='store_true')

    args = parser.parse_args()
    set_seed(12345)
    main(args)

    