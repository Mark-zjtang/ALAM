import os
import math
import pathlib
import logging
import dataclasses
import numpy as np
import torch
import imageio
import metaworld
from tqdm import tqdm
from datetime import datetime

from training import config as _config
from policies import policy_config
from shared import download
from mt50_eval_instruction import INSTRUCTION

os.environ["MUJOCO_GL"] = "egl"

#######################################################################################################
# 配置参数（类似 LIBERO 中的 Args）
#######################################################################################################
@dataclasses.dataclass
class Args:
    ###################################################################################################
    # 模型参数
    ###################################################################################################
    device: str = "cuda"
    policy_type: str = "pi0_wflow"
    pretrained_policy_path: str = (
        "/home/fortress/new_storage/TZJ/openpi_worldflow/checkpoints/metaworld_finetune/"
        "metaworld_pushv2_worldflow_h_5_1GPU_10_22_4_pool/14999"
    )

    ###################################################################################################
    # 环境参数
    ###################################################################################################
    env_name: str = "push-v3"
    single_task: bool = False
    max_episodes: int = 10
    cam_pos_x: float = 0.75
    cam_pos_y: float = 0.075
    cam_pos_z: float = 0.7
    seed: int = 5

    ###################################################################################################
    # 输出与日志
    ###################################################################################################
    output_dir: str = "metaworld_outputs/eval"


#######################################################################################################
# 评估函数
#######################################################################################################
def eval_metaworld(args: Args):
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    config = _config.get_config("metaworld_finetune")
    mt50 = metaworld.MT50()

    # 日志输出路径
    output_log_dir = (
        pathlib.Path(args.output_dir)
        / f"{args.policy_type}_policy_mt50_eval_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}"
    )
    output_log_dir.mkdir(parents=True, exist_ok=True)

    total_successes, total_episodes = 0, 0
    total_rewards = 0.0

    ###################################################################################################
    # 遍历任务
    ###################################################################################################
    for task_id, task in tqdm(enumerate(INSTRUCTION.items()), total=len(INSTRUCTION), desc="Evaluating Tasks"):
        env_name = args.env_name if args.single_task else task[0]
        goal_task = [task[1]]

        logging.info(f"Initializing environment: {env_name}")

        env = mt50.train_classes[env_name](render_mode="rgb_array", camera_name="corner2")
        task_instance = [t for t in mt50.train_tasks if t.env_name == env_name][0]
        env.set_task(task_instance)

        # 设置相机
        env.model.cam_pos[2] = [args.cam_pos_x, args.cam_pos_y, args.cam_pos_z]

        ###################################################################################################
        # 加载策略
        ###################################################################################################
        if args.policy_type == "diffusion":
            policy = DiffusionPolicy.from_pretrained(args.pretrained_policy_path, map_location=args.device)
        elif args.policy_type == "act":
            policy = ACTPolicy.from_pretrained(args.pretrained_policy_path, map_location=args.device)
        elif args.policy_type == "pi0":
            policy = PI0Policy.from_pretrained(args.pretrained_policy_path, map_location=args.device)
        elif args.policy_type == "pi0_wflow":
            policy = policy_config.create_trained_policy(
                config, download.maybe_download(args.pretrained_policy_path)
            )
        else:
            raise ValueError(f"Unknown policy type: {args.policy_type}")

        logging.info(f"Using policy: {args.policy_type}")

        ###################################################################################################
        # 任务输出路径
        ###################################################################################################
        task_output_dir = (
            pathlib.Path(args.output_dir)
            / env_name
            / f"{env_name}_{args.policy_type}_eval_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}"
        )
        task_output_dir.mkdir(parents=True, exist_ok=True)

        ###################################################################################################
        # 运行评估 Episode
        ###################################################################################################
        successes = 0
        total_reward_task = 0.0
        fps = env.metadata.get("render_fps", 10)

        for episode_idx in tqdm(range(args.max_episodes), desc=f"{env_name} Episodes"):
            policy.reset()
            obs, info = env.reset(seed=args.seed)
            img = env.render()
            frames = []
            success_flag = False
            done = False
            step_count = 0

            while not done and step_count < 500:
                rotated_img = np.rot90(img.copy(), k=2, axes=(0, 1))

                image_tensor = (
                    torch.tensor(rotated_img.copy(), dtype=torch.float32).permute(2, 0, 1).unsqueeze(0).to(args.device)
                    / 255.0
                )
                state_tensor = torch.tensor(obs, dtype=torch.float32)[:4].unsqueeze(0).to(args.device)

                observation = {
                    "observation/state": state_tensor,
                    "observation/images/head": image_tensor,  # 用 head 表示主视角图像
                    "observation/images/hand": image_tensor,  # 暂时同一图像替代
                    "prompt": goal_task,
                }

                with torch.inference_mode():
                    action = policy.infer(observation)["actions"]

                numpy_action = action.squeeze(0).cpu().numpy()
                obs, reward, truncated, terminated, info = env.step(numpy_action)
                total_reward_task += reward
                img = env.render()
                frames.append(rotated_img)
                step_count += 1

                done = truncated or terminated
                if info.get("success", False):
                    success_flag = True

            # 保存每个 episode 视频
            video_path = task_output_dir / f"rollout_ep_{episode_idx}.mp4"
            imageio.mimsave(str(video_path), np.stack(frames), fps=fps)

            if success_flag:
                successes += 1
            total_episodes += 1
            total_successes += int(success_flag)
            logging.info(f"[{env_name}] Episode {episode_idx}: {'SUCCESS' if success_flag else 'FAILURE'}")

        ###################################################################################################
        # 日志统计
        ###################################################################################################
        avg_reward = total_reward_task / args.max_episodes
        success_rate = successes / args.max_episodes

        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s - %(levelname)s - %(message)s",
            filename=output_log_dir / f"{env_name}_episodes_{args.max_episodes}.log",
            filemode="a",
        )
        logger = logging.getLogger(__name__)
        logger.info(f"{env_name} Success rate: {success_rate:.2%}")
        logger.info(f"{env_name} Average reward: {avg_reward:.3f}")
        logger.info("=====================================================\n")

    env.close()
    logging.info(f"Overall Success Rate: {total_successes / total_episodes:.2%}")
    logging.info(f"Total Episodes: {total_episodes}")


#######################################################################################################
# 启动入口
#######################################################################################################
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    args = Args()
    eval_metaworld(args)
