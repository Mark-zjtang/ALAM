from pathlib import Path
import os
import mujoco
import argparse
import logging
import metaworld
import numpy as np
import torch
import imageio
os.environ["MUJOCO_GL"] = "egl"

from openpi_client import websocket_client_policy as _websocket_client_policy
from datetime import datetime
from tqdm import tqdm
from mt50_eval_instruction import INSTRUCTION


TASK_DIFFICULTY = {
    'easy': [
        'button-press-v3', 'button-press-topdown-v3', 'button-press-topdown-wall-v3', 'button-press-wall-v3',
        'coffee-button-v3', 'dial-turn-v3', 'door-close-v3', 'door-lock-v3',
        'door-open-v3', 'door-unlock-v3', 'drawer-close-v3', 'drawer-open-v3',
        'faucet-close-v3', 'faucet-open-v3', 'handle-press-v3', 'handle-press-side-v3',
        'handle-pull-v3', 'handle-pull-side-v3', 'lever-pull-v3', 'plate-slide-v3',
        'plate-slide-back-v3', 'plate-slide-back-side-v3', 'plate-slide-side-v3', 'reach-v3', 'reach-wall-v3',
        'window-close-v3', 'window-open-v3', 'peg-unplug-side-v3'
    ],
    'medium': [
        'basketball-v3', 'bin-picking-v3', 'box-close-v3', 'coffee-pull-v3',
        'coffee-push-v3', 'hammer-v3', 'peg-insert-side-v3', 'push-wall-v3',
        'soccer-v3', 'sweep-v3', 'sweep-into-v3'
    ],
    'hard': [
        'assembly-v3', 'hand-insert-v3', 'pick-out-of-hole-v3', 'pick-place-v3', 'push-v3', 'push-back-v3'
    ],
    'very_hard': [
        'shelf-place-v3', 'disassemble-v3', 'stick-pull-v3', 'stick-push-v3', 'pick-place-wall-v3'
    ]
}


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate policy on MetaWorld tasks")
    parser.add_argument("--device", type=str, default="cuda", choices=["cuda", "cpu"])
    parser.add_argument("--policy_type", type=str, default="pi0_wflow", choices=["pi0_wflow"])
    parser.add_argument("--pretrained_policy_path", type=str, default="/home/fortress/new_storage/TZJ/openpi_worldflow/checkpoints/metaworld_finetune/metaworld_pushv2_worldflow_h_5_1GPU_10_22_4_pool/14999", required=False)
    parser.add_argument("--env_name", type=str, default="push-v3")
    parser.add_argument("--single_task", action="store_true")
    parser.add_argument("--max_episodes", type=int, default=10)
    parser.add_argument("--cam_pos_x", type=float, default=0.71)
    parser.add_argument("--cam_pos_y", type=float, default=0.075)
    parser.add_argument("--cam_pos_z", type=float, default=0.7)
    parser.add_argument("--seed", type=int, default=10)
    parser.add_argument("--output_dir", type=str, default="examples/metaworld/eval_results")
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--chunk_size", type=int, default=10)
    parser.add_argument("--action_horzion", type=int, default=10)
    parser.add_argument("--eval_max_steps", type=int, default=200)
    parser.add_argument("--model_name", type=str, default="worldflow_light_h_20_1stage_training_11_19_chunk_size10")
    return parser.parse_args()


def get_task_difficulty(env_name):
    for difficulty, tasks in TASK_DIFFICULTY.items():
        if env_name in tasks:
            return difficulty
    return 'unknown'


def main():
    args = parse_args()
    mt50 = metaworld.MT50()
    output_log_directory = Path(args.output_dir)/f"{args.model_name}"/f"{args.policy_type}_policy_49_task_evaluate_logging_{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
    output_log_directory.mkdir(parents=True, exist_ok=True)

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        filename=output_log_directory/f'{args.model_name}_episodes_{args.max_episodes}.log',
        filemode='a'
    )
    logger = logging.getLogger(__name__)
    logger.info(f"Evaluating policy on {args.model_name}")

    results_by_difficulty = {
        'easy':      {'success_count': 0, 'total_episodes': 0, 'tasks_evaluated': set()},
        'medium':    {'success_count': 0, 'total_episodes': 0, 'tasks_evaluated': set()},
        'hard':      {'success_count': 0, 'total_episodes': 0, 'tasks_evaluated': set()},
        'very_hard': {'success_count': 0, 'total_episodes': 0, 'tasks_evaluated': set()},
        'unknown':   {'success_count': 0, 'total_episodes': 0, 'tasks_evaluated': set()}
    }

    if args.policy_type == "pi0_wflow":
        policy = _websocket_client_policy.WebsocketClientPolicy(
            host=args.host,
            port=args.port,
        )
    else:
        raise ValueError("Please select one true policy")


    for task_id, task in tqdm(enumerate(INSTRUCTION.items()), desc="Processing Tasks", total=len(INSTRUCTION)):

        if args.single_task:
            print("single_task")
            env_name = args.env_name
            goal_task = [INSTRUCTION[env_name]] 
        else:
            print("multi_task")
            env_name = task[0]
            goal_task = [task[1]]                

        print(f"task_id_{task_id} env_name: {env_name}")
        print(f"chunk_size_{args.chunk_size}")

        difficulty = get_task_difficulty(env_name)
        print(f"Task difficulty: {difficulty}")

        env = mt50.train_classes[env_name](render_mode="rgb_array", camera_name="corner2")

        all_task_variants = [t for t in mt50.train_tasks if t.env_name == env_name]
        print(f"Found {len(all_task_variants)} variants for {env_name}")


        output_directory = Path(args.output_dir)/f"{args.model_name}"/f"{env_name}"/f"{env_name}_task_{args.policy_type}_policy_{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
        output_directory.mkdir(parents=True, exist_ok=True)

        print(f"start evaluate task: {goal_task[0]}")
        successes = 0
        chunk_size = args.chunk_size

        for episode in tqdm(range(args.max_episodes), desc="Evaluating"):
            policy.reset()
            eval_rewards = 0   

            task_variant = all_task_variants[episode % len(all_task_variants)]
            env.set_task(task_variant)

            # set_task后设置camera
            env.model.cam_pos[2] = [args.cam_pos_x, args.cam_pos_y, args.cam_pos_z]
            for obj in (env, getattr(env, "unwrapped", None)):
                fn = getattr(obj, "iterate_goal_position", None)
                if callable(fn):
                    try:
                        fn()
                    except Exception:
                        pass
                    break

            numpy_observation, info = env.reset(seed=args.seed + task_id * 1000 + episode)

            image_array = env.render()  

            truncated, terminated = False, False
            frames = []
            fps = env.metadata["render_fps"]
            successes_episode_flag = False
            action_buffer = []
            current_chunk_step = 0

            for step in tqdm(range(args.eval_max_steps), desc="steps"):  
                image_array = env.render()                                    
                rotated_image2 = np.rot90(image_array.copy(), k=2, axes=(0,1))
                frames.append(rotated_image2)                               

                if len(action_buffer) == 0 or current_chunk_step >= len(action_buffer):   
                    image = torch.tensor(rotated_image2.copy(), dtype=torch.float32)
                    state = torch.tensor(numpy_observation, dtype=torch.float32)[:4]

                    image = image / 255
                    image = image.permute(2, 0, 1)
                    image = image.to(args.device, non_blocking=True)
                    state = state.to(args.device, non_blocking=True)
                    image = image.unsqueeze(0)

                    observation = {
                        "observation/state": state.detach().cpu().numpy(),
                        "observation/image": image.detach().cpu().numpy(),
                        "prompt": goal_task[0],
                    }

                    with torch.inference_mode():
                        sample_actions = policy.infer(observation)["actions"][:, :4]
                        if sample_actions.shape[0] > chunk_size:
                            sample_actions = sample_actions[:chunk_size]

                    action_buffer = [act for act in sample_actions]
                    current_chunk_step = 0

                numpy_action = action_buffer[current_chunk_step]
                current_chunk_step += 1

                # ========== 新增：clip动作到合法范围 ==========
                numpy_action = np.clip(
                    numpy_action,
                    env.action_space.low,
                    env.action_space.high
                )
                # ==============================================

                next_numpy_observation, reward, truncated, terminated, info = env.step(numpy_action)
                numpy_observation = next_numpy_observation
                eval_rewards += reward


                if not successes_episode_flag and info.get("success", False):
                    successes_episode_flag = True
                    break

                if truncated or terminated:
                    break

            if successes_episode_flag:
                successes += 1
                results_by_difficulty[difficulty]['success_count'] += 1

            results_by_difficulty[difficulty]['total_episodes'] += 1
            results_by_difficulty[difficulty]['tasks_evaluated'].add(env_name)

            video_path = output_directory / f"rollout_episode_{episode}.mp4"
            imageio.mimsave(str(video_path), np.stack(frames), fps=fps)
            print("Video of the evaluation is available in '{}'".format(video_path))

            logger.info(f"Episode {episode}: success={successes_episode_flag}, reward={eval_rewards:.4f}")   # ← 新增
            logger.info(f"port: {args.port}, replan_steps: {args.chunk_size}")  # 

        env.close()
        success_rate = successes / args.max_episodes

        logger.info(f"evaluate_model: {args.model_name}")
        logger.info(f"Task difficulty: {difficulty}")
        logger.info(f"{env_name} Success rate: {success_rate:.2%}")
        logger.info("=====================================================\n")

        if args.single_task:
            break


    print("\n" + "="*60)
    print("SUMMARY BY TASK DIFFICULTY")
    print("="*60)

    for difficulty, stats in results_by_difficulty.items():
        if stats['total_episodes'] > 0:
            success_rate = stats['success_count'] / stats['total_episodes']
            tasks_count = len(stats['tasks_evaluated'])
            print(f"{difficulty.upper():<12} | Success Rate: {success_rate:.2%} | Tasks: {tasks_count} | Episodes: {stats['total_episodes']}")
            logger.info(f"{difficulty.upper():<12} | Success Rate: {success_rate:.2%} | Tasks: {tasks_count} | Episodes: {stats['total_episodes']}")

    print("="*60)


if __name__ == "__main__":
    main()
