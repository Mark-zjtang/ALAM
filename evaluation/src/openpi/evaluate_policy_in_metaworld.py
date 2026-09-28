from pathlib import Path
# import gymnasium as gym
import os
import mujoco
import argparse
import logging
import metaworld
os.environ["MUJOCO_GL"] = "egl"

from datetime import datetime
from tqdm import tqdm
from mt50_eval_instruction import INSTRUCTION

from training import config as _config
from policies import policy_config
from shared import download


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate policy on MetaWorld tasks")
    parser.add_argument("--device", type=str, default="cuda", choices=["cuda", "cpu"], help="Device to use (cuda or cpu)")
    parser.add_argument("--policy_type", type=str, default="pi0_wflow", 
                       choices=["diffusion", "act", "pi0_wflow"], help="Type of policy to evaluate")
    parser.add_argument("--pretrained_policy_path", type=str, default="/home/fortress/new_storage/TZJ/openpi_worldflow/checkpoints/metaworld_finetune/metaworld_pushv2_worldflow_h_5_1GPU_10_22_4_pool/14999", required=False,
                       help="Path to pretrained policy checkpoint")
    parser.add_argument("--env_name", type=str, default="push-v3",
                       help="Name of the environment to evaluate on")
    parser.add_argument("--single_task", action="store_true",
                       help="Whether to evaluate on a single task")
    parser.add_argument("--max_episodes", type=int, default=10,
                       help="Number of episodes to evaluate")
    parser.add_argument("--cam_pos_x", type=float, default=0.75,
                       help="Camera X position")
    parser.add_argument("--cam_pos_y", type=float, default=0.075,
                       help="Camera Y position")
    parser.add_argument("--cam_pos_z", type=float, default=0.7,
                       help="Camera Z position-parse setting is [0.8, 0.1, 0.7] on 25/06/16")
    parser.add_argument("--seed", type=int, default=5,
                       help="Random seed for environment")
    parser.add_argument("--output_dir", type=str, default="metaworld_outputs/eval",
                       help="Directory to save evaluation results")
    return parser.parse_args()


def main():

    config = _config.get_config("metaworld_finetune")
   

    args = parse_args()
    mt50 = metaworld.MT50()
    output_log_directory = Path(args.output_dir)/f"{args.policy_type}_policy_49_task_evaluate_logging_{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
    output_log_directory.mkdir(parents=True, exist_ok=True)
    
    for task_id , task in tqdm(enumerate(INSTRUCTION.items()), desc="Processing Tasks",total=len(INSTRUCTION)):

        # Select one task
        # if args.single_task:
        #     env_name = args.env_name
        # else:
        #     env_name = list(mt50.train_classes.keys())[task[0]]  # such as 'pick-place-v2'
        if args.single_task:
            print("single_task")
            env_name =  args.env_name
        else:
            print("multi_task")
            env_name = task[0]

        print(f"task_id_{task_id} env_name: {env_name}")
        env = mt50.train_classes[env_name](render_mode="rgb_array",camera_name="corner2")
        # finding match task
        task2 = [t for t in mt50.train_tasks if t.env_name == env_name][0]
        print("setting_task:", task2)
        env.set_task(task2)
        env.model.cam_pos[2] = [args.cam_pos_x, args.cam_pos_y, args.cam_pos_z]  # 调整 XYZ 坐标

        goal_task = [task[1]]
        print("goal_task", goal_task[0])

        if args.policy_type == "diffusion":
            policy = DiffusionPolicy.from_pretrained(args.pretrained_policy_path, map_location=args.device)
        elif args.policy_type == "act":
            policy = ACTPolicy.from_pretrained(args.pretrained_policy_path, map_location=args.device)
        elif args.policy_type == "pi0":
            policy = PI0Policy.from_pretrained(args.pretrained_policy_path, map_location=args.device)
        elif args.policy_type == "pi0_wflow":
            # Create a trained policy.
            policy = policy_config.create_trained_policy(config,  download.maybe_download(args.pretrained_policy_path))
        else:
            raise ValueError("Please select one policy")
        print(f"policy: {args.policy_type}")
        # Create a directory to store the video of the evaluation
        output_directory = Path(args.output_dir)/f"{env_name}"/f"{env_name}_task_{args.policy_type}_policy_{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
        output_directory.mkdir(parents=True, exist_ok=True)

        # We can verify that the shapes of the features expected by the policy match the ones from the observations
        # produced by the environment
        # print(policy.config.input_features)
        # print(env.observation_space)

        # Similarly, we can check that the actions produced by the policy will match the actions expected by the
        # environment
        # print(policy.config.output_features)
        # print(env.action_space)

        print("start evaluate")
        successes = 0
        eval_rewards = 0
        for episode in tqdm(range(args.max_episodes), desc="Evaluating"):
            # Reset the policy and environments to prepare for rollout
            # print(f"================episode_{episode}====================")
            policy.reset()
            numpy_observation, info = env.reset(seed=args.seed)
            image_array = env.render()
            truncated, terminated = False, False
            frames = []
            step = 0
            fps = env.metadata["render_fps"]
            successes_episode_flag = False

            for step in tqdm(range(500), desc="steps"):
                # print("image_array.shape", image_array.shape)
                # rotated_image = np.rot90(image_array.copy(), k=2)
                rotated_image2 = np.rot90(image_array.copy(), k=2, axes=(0,1))

                image = torch.tensor(rotated_image2.copy(), dtype=torch.float32)
                state = torch.tensor(numpy_observation, dtype=torch.float32)[:4]
                # print(f"================step_{step}====================")

                image = image / 255
                image = image.permute(2, 0, 1)  

                image = image.to(args.device, non_blocking=True)  
                state = state.to(args.device, non_blocking=True)

                state = state.unsqueeze(0)
                image = image.unsqueeze(0)

                # print("state", state.shape)
                # print("image", image.shape)

                # Create the policy input dictionary
        
                # observation = {
                #     "observation.state": state,
                #     "observation.image": image,
                #     "task": goal_task
                # }


                observation = {
                    "observation/state": state,
                    "observation/images/head": head_img,
                    "observation/images/hand": hand_img, 
                    "prompt": goal_task,
                }

                # Predict the next action with respect to the current observation
                with torch.inference_mode():
                    action = policy.infer(obeservation)("actions")
                    print('policy_select_action', action.shape)

                # Prepare the action for the environment
                numpy_action = action.squeeze(0).to("cpu").numpy()
                # print(env.action_space)

                # Step through the environment and receive a new observation
                next_numpy_observation, reward, truncated, terminated, info = env.step(numpy_action)
                numpy_observation = next_numpy_observation
                frames.append(rotated_image2)
                eval_rewards += reward
                step += 1
                image_array = env.render()

                if not successes_episode_flag and info.get("success", False):
                    successes_episode_flag = True

                if truncated or terminated:
                    break
            
            if successes_episode_flag:
                successes += 1

            video_path = output_directory / f"rollout_episode_{episode}.mp4"
            imageio.mimsave(str(video_path), numpy.stack(frames), fps=fps)
            print("Video of the evaluation is available in '{}'".format(video_path))

        env.close()
        final_rewards = eval_rewards / args.max_episodes
        success_rate = successes / args.max_episodes
        logging.basicConfig(
                level=logging.INFO,
                format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                filename=output_log_directory/f'{env_name}_episodes_{args.max_episodes}.log',  # 直接指定日志文件
                filemode='a'         # 'a' 表示追加模式，'w' 表示覆盖模式
            )

        logger = logging.getLogger(__name__)
        logger.info(f"{env_name} Success rate: {success_rate:.2%}")
        logger.info(f"{env_name} Final_rewards: {final_rewards}")
        logger.info("=====================================================\n")



if __name__ =="__main__":
    main()


