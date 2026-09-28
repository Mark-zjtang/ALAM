import os
import math
import pathlib
import logging
import dataclasses
import numpy as np
import imageio
import metaworld
from tqdm import tqdm
from datetime import datetime

# 移除本地策略导入，只保留WebSocket客户端相关
from openpi_client import action_chunk_broker
from openpi_client import websocket_client_policy as _websocket_client_policy
from openpi_client.runtime import runtime as _runtime
from openpi_client.runtime.agents import policy_agent as _policy_agent
import tyro

from mt50_eval_instruction import INSTRUCTION

os.environ["MUJOCO_GL"] = "egl"

#######################################################################################################
# 配置参数
#######################################################################################################
@dataclasses.dataclass
class Args:
    ###################################################################################################
    # 输出与日志
    ###################################################################################################
    out_dir: pathlib.Path = pathlib.Path("metaworld_outputs/eval")
    
    ###################################################################################################
    # 环境参数
    ###################################################################################################
    task: str = "push-v3"
    single_task: bool = False
    max_episodes: int = 10
    cam_pos_x: float = 0.75
    cam_pos_y: float = 0.075
    cam_pos_z: float = 0.7
    seed: int = 5

    ###################################################################################################
    # WebSocket客户端参数
    ###################################################################################################
    host: str = "0.0.0.0"  # 策略服务器地址
    port: int = 8000       # 策略服务器端口
    action_horizon: int = 10

    ###################################################################################################
    # 显示选项
    ###################################################################################################
    display: bool = False


#######################################################################################################
# MetaWorld环境包装器 - 适配OpenPI运行时接口
#######################################################################################################
class MetaWorldEnvironment:
    def __init__(self, task: str, seed: int, cam_pos_x: float = 0.75, cam_pos_y: float = 0.075, cam_pos_z: float = 0.7):
        self.task = task
        self.seed = seed
        self.cam_pos_x = cam_pos_x
        self.cam_pos_y = cam_pos_y
        self.cam_pos_z = cam_pos_z
        
        self.mt50 = metaworld.MT50()
        self.env = None
        self.current_task_name = None
        self.current_goal = None
        self.current_obs = None
        self._setup_environment(task)
        
    def _setup_environment(self, task_name: str):
        """设置环境"""
        if task_name not in self.mt50.train_classes:
            raise ValueError(f"Task {task_name} not found in MT50")
            
        self.env = self.mt50.train_classes[task_name](render_mode="rgb_array", camera_name="corner2")
        task_instance = [t for t in self.mt50.train_tasks if t.env_name == task_name][0]
        self.env.set_task(task_instance)
        
        # 设置相机位置
        self.env.model.cam_pos[2] = [self.cam_pos_x, self.cam_pos_y, self.cam_pos_z]
        
        self.current_task_name = task_name
        self.current_goal = INSTRUCTION.get(task_name, "Unknown task")
        
    def reset(self):
        """重置环境并返回观察"""
        obs, info = self.env.reset(seed=self.seed)
        img = self.env.render()
        rotated_img = np.rot90(img.copy(), k=2, axes=(0, 1))
        
        self.current_obs = {
            "observation": obs,
            "image": rotated_img,
            "info": info,
            "goal": self.current_goal
        }
        
        return self.current_obs
    
    def step(self, action):
        """执行一步动作"""
        obs, reward, truncated, terminated, info = self.env.step(action)
        img = self.env.render()
        rotated_img = np.rot90(img.copy(), k=2, axes=(0, 1))
        
        done = truncated or terminated
        success = info.get("success", False)
        
        self.current_obs = {
            "observation": obs,
            "image": rotated_img,
            "reward": reward,
            "done": done,
            "truncated": truncated,
            "terminated": terminated,
            "info": info,
            "success": success
        }
        
        return self.current_obs
    
    def get_observation(self):
        """获取当前观察"""
        return self.current_obs
    
    def render(self):
        """渲染当前帧"""
        img = self.env.render()
        return np.rot90(img.copy(), k=2, axes=(0, 1))
    
    def close(self):
        """关闭环境"""
        if self.env:
            self.env.close()


#######################################################################################################
# 视频保存器
#######################################################################################################
class VideoSaver:
    def __init__(self, output_dir: pathlib.Path):
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.frames = []
        self.episode_count = 0
        
    def __call__(self, observation, action, reward, done, info):
        """保存观察帧，在episode结束时保存视频"""
        if observation and "image" in observation:
            self.frames.append(observation["image"])
        
        # 如果episode结束，保存视频
        if done and self.frames:
            video_path = self.output_dir / f"episode_{self.episode_count}.mp4"
            try:
                # 确保所有帧的形状一致
                frames_array = np.stack(self.frames)
                imageio.mimsave(str(video_path), frames_array, fps=20)
                logging.info(f"Saved video: {video_path}")
            except Exception as e:
                logging.warning(f"Failed to save video: {e}")
            
            self.frames = []
            self.episode_count += 1


#######################################################################################################
# 评估运行器
#######################################################################################################
class MetaWorldEvaluator:
    def __init__(self, args: Args):
        self.args = args
        self.output_dir = pathlib.Path(args.out_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # 设置日志
        self._setup_logging()
        
    def _setup_logging(self):
        """设置日志"""
        log_file = self.output_dir / f"metaworld_eval_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.log"
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s - %(levelname)s - %(message)s",
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
        
    def create_runtime(self, task_name: str, goal_text: str):
        """为特定任务创建runtime"""
        # 创建环境实例
        env = MetaWorldEnvironment(
            task=task_name,
            seed=self.args.seed,
            cam_pos_x=self.args.cam_pos_x,
            cam_pos_y=self.args.cam_pos_y,
            cam_pos_z=self.args.cam_pos_z
        )
        
        # 创建视频保存器
        task_output_dir = self.output_dir / task_name
        video_saver = VideoSaver(task_output_dir)
        
        # 创建WebSocket策略代理
        runtime = _runtime.Runtime(
            environment=env,
            agent=_policy_agent.PolicyAgent(
                policy=action_chunk_broker.ActionChunkBroker(
                    policy=_websocket_client_policy.WebsocketClientPolicy(
                        host=self.args.host,
                        port=self.args.port,
                    ),
                    action_horizon=self.args.action_horizon,
                )
            ),
            subscribers=[video_saver],
            max_hz=50,
        )
        
        return runtime, env
    
    def run_evaluation(self):
        """运行评估"""
        tasks_to_evaluate = []
        
        if self.args.single_task:
            # 单任务评估
            if self.args.task not in INSTRUCTION:
                self.logger.warning(f"Task {self.args.task} not found in INSTRUCTION, using default goal")
                goal = f"Complete {self.args.task}"
            else:
                goal = INSTRUCTION[self.args.task]
            tasks_to_evaluate.append((self.args.task, goal))
        else:
            # 多任务评估（MT50）
            tasks_to_evaluate = list(INSTRUCTION.items())
        
        total_successes = 0
        total_episodes = 0
        
        for task_name, goal in tqdm(tasks_to_evaluate, desc="Evaluating Tasks"):
            self.logger.info(f"Evaluating task: {task_name}")
            self.logger.info(f"Task goal: {goal}")
            
            try:
                runtime, env = self.create_runtime(task_name, goal)
                task_successes = 0
                
                for episode_idx in range(self.args.max_episodes):
                    self.logger.info(f"Starting episode {episode_idx + 1}/{self.args.max_episodes}")
                    
                    # 重置环境
                    obs = env.reset()
                    
                    # 运行episode - 使用runtime的run方法
                    # 这里假设runtime.run()会自动运行一个完整的episode
                    # 如果需要更细粒度的控制，可以手动步进
                    try:
                        # 运行一个episode
                        # 注意：这里需要根据runtime的实际接口进行调整
                        runtime.run()
                        
                        # 检查是否成功（这里需要根据实际环境接口调整）
                        # 可能需要从环境的最后状态获取成功信息
                        final_obs = env.get_observation()
                        success = final_obs.get("success", False)
                        
                        if success:
                            task_successes += 1
                            total_successes += 1
                            self.logger.info(f"Episode {episode_idx + 1}: SUCCESS")
                        else:
                            self.logger.info(f"Episode {episode_idx + 1}: FAILURE")
                            
                    except Exception as e:
                        self.logger.error(f"Error during episode {episode_idx + 1}: {e}")
                        success = False
                        self.logger.info(f"Episode {episode_idx + 1}: FAILURE (Error)")
                    
                    total_episodes += 1
                
                task_success_rate = task_successes / self.args.max_episodes
                self.logger.info(f"Task {task_name} success rate: {task_success_rate:.2%}")
                
                env.close()
                
            except Exception as e:
                self.logger.error(f"Error evaluating task {task_name}: {e}")
                continue
        
        overall_success_rate = total_successes / total_episodes if total_episodes > 0 else 0
        self.logger.info(f"Overall evaluation completed:")
        self.logger.info(f"Total episodes: {total_episodes}")
        self.logger.info(f"Total successes: {total_successes}")
        self.logger.info(f"Overall success rate: {overall_success_rate:.2%}")


#######################################################################################################
# 替代的手动评估方法（如果runtime.run()不适用）
#######################################################################################################
def manual_evaluation(args: Args):
    """手动控制评估循环"""
    output_dir = pathlib.Path(args.out_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    tasks_to_evaluate = list(INSTRUCTION.items()) if not args.single_task else [(args.task, INSTRUCTION.get(args.task, "Unknown task"))]
    
    total_successes = 0
    total_episodes = 0
    
    for task_name, goal in tqdm(tasks_to_evaluate, desc="Evaluating Tasks"):
        logging.info(f"Evaluating task: {task_name}")
        
        # 创建环境
        env = MetaWorldEnvironment(
            task=task_name,
            seed=args.seed,
            cam_pos_x=args.cam_pos_x,
            cam_pos_y=args.cam_pos_y,
            cam_pos_z=args.cam_pos_z
        )
        
        # 创建WebSocket策略
        policy = _policy_agent.PolicyAgent(
            policy=action_chunk_broker.ActionChunkBroker(
                policy=_websocket_client_policy.WebsocketClientPolicy(
                    host=args.host,
                    port=args.port,
                ),
                action_horizon=args.action_horizon,
            )
        )
        
        # 视频保存器
        task_output_dir = output_dir / task_name
        video_saver = VideoSaver(task_output_dir)
        
        task_successes = 0
        
        for episode_idx in range(args.max_episodes):
            logging.info(f"Starting episode {episode_idx + 1}/{args.max_episodes}")
            
            obs = env.reset()
            done = False
            steps = 0
            success = False
            
            while not done and steps < 500:  # 最大步数限制
                # 通过WebSocket策略获取动作
                try:
                    action = policy.act(obs)
                    
                    # 执行动作
                    obs = env.step(action)
                    done = obs["done"]
                    success = obs.get("success", False)
                    
                    # 保存视频帧
                    video_saver(obs, action, obs.get("reward", 0), done, obs.get("info", {}))
                    
                    steps += 1
                    
                except Exception as e:
                    logging.error(f"Error during step {steps}: {e}")
                    break
            
            if success:
                task_successes += 1
                total_successes += 1
                logging.info(f"Episode {episode_idx + 1}: SUCCESS")
            else:
                logging.info(f"Episode {episode_idx + 1}: FAILURE")
            
            total_episodes += 1
        
        task_success_rate = task_successes / args.max_episodes
        logging.info(f"Task {task_name} success rate: {task_success_rate:.2%}")
        
        env.close()
    
    overall_success_rate = total_successes / total_episodes if total_episodes > 0 else 0
    logging.info(f"Overall evaluation completed:")
    logging.info(f"Total episodes: {total_episodes}")
    logging.info(f"Total successes: {total_successes}")
    logging.info(f"Overall success rate: {overall_success_rate:.2%}")


#######################################################################################################
# 主函数
#######################################################################################################
def main(args: Args) -> None:
    """主评估函数"""
    # 使用手动评估方法，更可控
    manual_evaluation(args)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, force=True)
    tyro.cli(main)