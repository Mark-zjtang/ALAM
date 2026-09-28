import os
from time import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import get_cosine_schedule_with_warmup
from accelerate import Accelerator
from accelerate.utils import DistributedDataParallelKwargs
from torch.utils.tensorboard import SummaryWriter
import torch
from uni_world_model.data.datasets import DataPrefetcher, JointFlowDataPrefetcher
import omegaconf
from glob import glob
import shutil
from collections import defaultdict
from torch.distributions import Beta
from uni_world_model.diffusion_policy.trainers.trainer_utils import cross_entropy, masked_loss, visualize_latent_action_gen, WorldActionCouplingLoss

class DP_Trainer(nn.Module):
    def __init__(
        self,
        diffusion_policy,
        diffusion_policy_config,
        latent_action_tokenizer,
        rgb_preprocessor,
        lang_tokenizer,
        train_dataloader,
        eval_dataloader,
        save_path,
        save_epochs=1,
        save_steps=10000,
        num_epochs=20,
        print_steps=100,
        lr_max=0.0001,
        weight_decay=0.0001,
        num_warmup_epochs=1,
        gradient_accumulation_steps=4,
        resume_ckpt_path=None,
        bs_per_gpu=32,
        max_epoch=None,
        pred_binary_gripper_action=True,
    ):
        super(DP_Trainer, self).__init__()
        ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
        accelerator= Accelerator(
            gradient_accumulation_steps=gradient_accumulation_steps,
            kwargs_handlers=[ddp_kwargs]
        )
        self.accelerator = accelerator

        if resume_ckpt_path is not None:
            self.print(f"resuming WorldModel from {resume_ckpt_path} ...")

            current_model_dict = diffusion_policy.state_dict()
            resume_model_dict = torch.load(os.path.join(resume_ckpt_path, 'pytorch_model.bin'), map_location='cpu')

            mismatched_param_names = []
            filtered_state_dict = {}

            for name, param in resume_model_dict.items():
                if name in current_model_dict and current_model_dict[name].shape != param.shape:
                    mismatched_param_names.append(name)
                else:
                    filtered_state_dict[name] = param

            missing_keys, unexpected_keys = diffusion_policy.load_state_dict(filtered_state_dict, strict=False)
            missing_root_keys = set([k.split(".")[0] for k in missing_keys])
            self.print('load ', resume_ckpt_path, '\nmissing ', missing_root_keys, '\nunexpected ', unexpected_keys, '\nmismatched ', mismatched_param_names)
        
        optimizer = torch.optim.AdamW(diffusion_policy.parameters(), lr=lr_max, weight_decay=weight_decay, fused=True)
        total_prints_per_epoch = len(train_dataloader.dataset) // (print_steps * bs_per_gpu * accelerator.num_processes)
        scheduler = get_cosine_schedule_with_warmup(
            optimizer, 
            num_warmup_steps=min(num_warmup_epochs*total_prints_per_epoch, 5000000 // (print_steps * bs_per_gpu * accelerator.num_processes)),
            num_training_steps=num_epochs*total_prints_per_epoch,
        )
        diffusion_policy, optimizer, train_dataloader, eval_dataloader = accelerator.prepare(
            diffusion_policy, optimizer, train_dataloader, eval_dataloader, 
            device_placement=[True, True, False, False]
        )
        if latent_action_tokenizer is not None:
            print("loading latent action tokenizer from pretrained model ...")
            latent_action_tokenizer = latent_action_tokenizer.to(accelerator.device)
            latent_action_tokenizer.eval()

        self.writer = SummaryWriter(os.path.join(save_path, 'logs'))
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.total_prints_per_epoch = total_prints_per_epoch
        self.diffusion_policy = diffusion_policy
        self.diffusion_policy_config = diffusion_policy_config
        self.latent_action_tokenizer = latent_action_tokenizer
        self.optimizer = optimizer
        self.train_prefetcher = JointFlowDataPrefetcher(train_dataloader, self.device, lang_tokenizer=lang_tokenizer, debug_mode=True)
        self.eval_prefetcher = JointFlowDataPrefetcher(eval_dataloader, self.device, lang_tokenizer=lang_tokenizer, debug_mode=False)
        self.rgb_preprocessor = rgb_preprocessor.to(self.device)
        self.lang_tokenizer = lang_tokenizer
        self.save_path = save_path
        self.save_epochs = save_epochs
        self.save_steps = save_steps
        self.max_epoch = max_epoch
        self.num_epochs = num_epochs
        self.print_steps = print_steps
        self.bs_per_gpu = bs_per_gpu
        self.pred_binary_gripper_action = pred_binary_gripper_action

        self.beta_dist = Beta(self.diffusion_policy_config.noise_beta_alpha, self.diffusion_policy_config.noise_beta_beta)
        self.num_timestep_buckets = self.diffusion_policy_config.num_timestep_buckets

        # ★ 新增：耦合损失模块（需要知道 action_dim 和 latent_dim）
        _act_dim  = self.diffusion_policy_config.act_dim
        _latent_dim  = getattr(
            self.diffusion_policy_config, 'latent_dim',
            diffusion_policy.module.latent_dim   # accelerate 包装后用 .module 访问
            if hasattr(diffusion_policy, 'module')
            else diffusion_policy.latent_dim
        )
        self.coupling_loss_fn = WorldActionCouplingLoss(
            action_dim=_act_dim,
            latent_dim=_latent_dim,
        ).to(self.device)

        # ★ 新增：损失权重（可在 config 中覆盖）
        self.lambda_w   = getattr(self.diffusion_policy_config, 'lambda_w',   1.0)
        self.lambda_a   = getattr(self.diffusion_policy_config, 'lambda_a',   200.0)
        self.lambda_la  = getattr(self.diffusion_policy_config, 'lambda_la',  1.0)
        self.lambda_c   = getattr(self.diffusion_policy_config, 'lambda_c',   0.5)

        print("DP_Trainer has initialized")
    @property
    def device(self):
        return self.accelerator.device

    @property
    def is_main(self):
        return self.accelerator.is_main_process

    @property
    def process_index(self):
        return self.accelerator.process_index

    def print(self, *args, **kwargs):
        self.accelerator.print(*args, **kwargs)

    def save_checkpoint(self, save_dir):
        unwrapped_diffusion_policy = self.accelerator.unwrap_model(self.diffusion_policy)
        state_dict = unwrapped_diffusion_policy.get_state_dict_to_save()
        
        torch.save(state_dict, os.path.join(save_dir, "pytorch_model.bin"))
        omegaconf.OmegaConf.save(unwrapped_diffusion_policy.config, os.path.join(save_dir, "config.yaml"))
        self.print(f"A new model checkpoint is saved to {save_dir}!!!")

    def sample_time(self, batch_size, device, dtype):
        sample = self.beta_dist.sample([batch_size]).to(device, dtype=dtype)
        return (self.diffusion_policy_config.noise_s - sample) / self.diffusion_policy_config.noise_s
        
    def train(self):
        step = 0
        print("Start Training ...")
        for epoch in range(self.num_epochs+1):
            if epoch != 0:
                self.accelerator.wait_for_everyone()
                save_dir = os.path.join(self.save_path, f'saved_epoch_{epoch}_step_{step}')

                if self.is_main:
                    os.makedirs(save_dir, exist_ok=True)
                    self.save_checkpoint(save_dir)

                if self.diffusion_policy_config.world_model.latent_action_pred:
                    visualization_dir = os.path.join(save_dir, 'visualization')
                    self.eval_latent_action_gen(visualization_dir)

                if epoch == self.num_epochs:
                    break
                if (self.max_epoch is not None) and (epoch >= self.max_epoch):
                    break

            # ★ 改动：新增 world_loss / coupling_loss 字段
            log_loss = {
                'latent_action_loss': torch.tensor(0).float().to(self.device),
                'action_loss':        torch.tensor(0).float().to(self.device),
                'world_loss':         torch.tensor(0).float().to(self.device),  # ★ 新增
                'coupling_loss':      torch.tensor(0).float().to(self.device),  # ★ 新增
                'total_loss':         torch.tensor(0).float().to(self.device),
            }
            eval_log_loss = {
                'latent_action_loss': torch.tensor(0).float().to(self.device),
                'action_loss':        torch.tensor(0).float().to(self.device),
                'world_loss':         torch.tensor(0).float().to(self.device),  # ★ 新增
                'coupling_loss':      torch.tensor(0).float().to(self.device),  # ★ 新增
                'total_loss':         torch.tensor(0).float().to(self.device),
            }
            
            cum_load_time = 0 
            clock = time()
            batch_idx = 0
            print("before batch")
            batch, load_time = self.train_prefetcher.next()
            print("after loading batch")
            
            while batch is not None:
                with self.accelerator.accumulate(self.diffusion_policy):

                    self.diffusion_policy.train()
                    self.optimizer.zero_grad()
                    loss = self.calculate_loss(batch, train=True)
                    self.accelerator.backward(loss['total_loss'])
                    self.optimizer.step()

                    for key in log_loss:
                        log_loss[key] += loss[key].detach() / self.print_steps
                    cum_load_time += load_time / self.print_steps

                if (batch_idx+1) % self.print_steps == 0:

                    with torch.no_grad():
                        self.diffusion_policy.eval()
                        batch, _ = self.eval_prefetcher.next_without_none()
                        loss = self.calculate_loss(batch, train=True)
                        for key in eval_log_loss:
                            eval_log_loss[key] = loss[key].detach()

                    self.log(log_loss, eval_log_loss, cum_load_time, clock, epoch, batch_idx, step)
                    for key in log_loss:
                        log_loss[key] = torch.tensor(0).float().to(self.device)
                    for key in eval_log_loss:
                        eval_log_loss[key] = torch.tensor(0).float().to(self.device)

                    cum_load_time = 0
                    clock = time()
                    self.scheduler.step()

                if batch_idx  % self.save_steps == 0: #(batch_idx+1)  % self.save_steps == 0:
                    self.accelerator.wait_for_everyone()
                    save_dir = os.path.join(self.save_path, f'temp_epoch_{epoch}_step_{step}')

                    if self.is_main:
                        existing_ckpt_dirs = glob(os.path.join(self.save_path, f'temp_epoch_*_step_*'))
                        for existing_ckpt_dir in existing_ckpt_dirs:
                            if existing_ckpt_dir != save_dir:
                                shutil.rmtree(existing_ckpt_dir)
                        os.makedirs(save_dir, exist_ok=True)
                        self.save_checkpoint(save_dir)

                    if self.diffusion_policy_config.world_model.latent_action_pred:
                        visualization_dir = os.path.join(save_dir, 'visualization')
                        self.eval_latent_action_gen(visualization_dir)
        

                batch_idx += 1
                step += 1
                batch, load_time = self.train_prefetcher.next()



    @torch.no_grad()
    def eval_latent_action_gen(self, visualization_dir):
        pass


    def calculate_loss(self, batch, train):

        # ══════════════════════════════════════════════════════════
        # Step 1: 图像预处理
        # ══════════════════════════════════════════════════════════
        # batch['rgb_initial']: (B, 1, 3, H, W)
        # batch['rgb_future']:  (B, C, 3, H, W)  C = chunk_size
        rgb_initial = self.rgb_preprocessor(batch['rgb_initial'], train=train)
        # (B, 1, 3, H, W) float

        # rgb_future 预处理（逐帧）
        B, C, ch, H, W = batch['rgb_future'].shape
        rgb_future_flat = batch['rgb_future'].reshape(B * C, ch, H, W)
        # 统一调用 preprocessor
        rgb_future_proc = self.rgb_preprocessor(
            rgb_future_flat.unsqueeze(1), train=train
        ).squeeze(1)                                    # (B*C, ch, H, W)
        rgb_future = rgb_future_proc.reshape(B, C, ch, H, W)

        # ══════════════════════════════════════════════════════════
        # Step 2: LAM 提取 w_true（时序对齐）
        #
        #   rgb_all = [O_t, O_t+1, ..., O_t+C]   (C+1 帧)
        #   w_true[i] = LAM.encode(rgb_all[i], rgb_all[i+1])
        #   → w_true shape: (B, C, num_codes * latent_dim)
        # ══════════════════════════════════════════════════════════
        latent_action_ids = None
        w_true            = None

        if self.latent_action_tokenizer is not None:
            # rgb_all: (B, C+1, ch, H, W)
            rgb_all = torch.cat([rgb_initial, rgb_future], dim=1)

            # 对每一步 i=0..C-1 做 LAM 编码
            # 批量处理：把 B*C 对帧一次性送入 LAM
            cond_frames   = rgb_all[:, :-1]           # (B, C, ch, H, W)
            target_frames = rgb_all[:, 1:]            # (B, C, ch, H, W)

            cond_flat   = cond_frames.reshape(B * C, ch, H, W)
            target_flat = target_frames.reshape(B * C, ch, H, W)

            lam_out = self.latent_action_tokenizer.encode_latent_action(
                cond_pixel_values=cond_flat,     # (B*C, ch, H, W)
                target_pixel_values=target_flat, # (B*C, ch, H, W)
            )
            # lam_out['token_ids']:              (B*C, 1, num_codes)
            # lam_out['physical_latent_action']: (B*C, 1, num_codes, latent_dim)

            # 离散 id：取第一步用于辅助损失
            # (只用 O_t → O_t+1 的 id 做 world_model 条件)
            token_ids_all = lam_out['token_ids'].reshape(B, C, -1)
            latent_action_ids = token_ids_all[:, 0]  # (B, num_codes)

            # 连续 z_e：所有 C 步，作为联合流的 w_true
            phys_a = lam_out['physical_latent_action']
            # (B*C, 1, num_codes, latent_dim)
            _, _, num_codes, latent_dim_ = phys_a.shape
            w_true = phys_a.reshape(B, C, num_codes * latent_dim_)
            # (B, C, num_codes * latent_dim)  ← 与 actions(B,C,act_dim) 对齐 ✅

        # ══════════════════════════════════════════════════════════
        # Step 3: 动作真值（LeRobot 格式已是 (B, C, act_dim)）
        # ══════════════════════════════════════════════════════════
        actions = batch['actions']                    # (B, C, act_dim)
        B, C, act_dim = actions.shape

        # ══════════════════════════════════════════════════════════
        # Step 4: 联合 Flow Matching 噪声采样（共享同一个 t）
        # ══════════════════════════════════════════════════════════
        t_cont = self.sample_time(B, device=actions.device, dtype=actions.dtype)
        # t_cont: (B,)

        t_bc  = t_cont[:, None, None]               # (B, 1, 1) 广播用

        # ── 动作噪声轨迹 ─────────────────────────────────────────
        noise_a          = torch.randn_like(actions)
        noisy_trajectory = (1 - t_bc) * noise_a + t_bc * actions
        v_a_target       = actions - noise_a         # (B, C, act_dim)

        # ── 世界噪声轨迹 ─────────────────────────────────────────
        noisy_world = None
        v_w_target  = None
        if w_true is not None:
            noise_w     = torch.randn_like(w_true)
            noisy_world = (1 - t_bc) * noise_w + t_bc * w_true
            v_w_target  = w_true - noise_w           # (B, C, num_codes*latent_dim)

        t_discretized = (t_cont * self.num_timestep_buckets).long()

        # ══════════════════════════════════════════════════════════
        # Step 5: 模型前向
        # ══════════════════════════════════════════════════════════
        pred = self.diffusion_policy(
            rgb=rgb_initial,
            language=batch['lang_input_ids'],
            latent_action_ids=latent_action_ids,
            noisy_trajectory=noisy_trajectory,       # (B, C, act_dim)
            t_discretized=t_discretized,
            noisy_world=noisy_world,                 # (B, C, num_codes*latent_dim)
            train=True,
        )

        pred_v = pred['pred_v']                      # (B, C, act_dim)
        pred_w = pred.get('pred_w', None)            # (B, C, num_codes*latent_dim)
        latent_action_preds = pred['latent_action_preds']

        # ══════════════════════════════════════════════════════════
        # Step 6: 损失计算
        #
        #   mask        : (B, C)  动作有效掩码
        #   latent_mask : (B, C)  帧有效掩码
        # ══════════════════════════════════════════════════════════
        mask        = batch['mask']                  # (B, C)
        latent_mask = batch['latent_mask']           # (B, C)
        loss        = {}

        # ── 辅助分类损失（只对第 0 步）────────────────────────────
        loss['latent_action_loss'] = (
            masked_loss(
                latent_action_preds,
                latent_action_ids,
                latent_mask[:, 0],   # 只用第 0 步的 mask
                0, cross_entropy
            )
            if latent_action_preds is not None
            else torch.tensor(0.0, device=actions.device)
        )

        # ── 动作流损失（所有 C 步）────────────────────────────────
        # masked_loss 期望 (B, 1, C, act_dim) 格式
        loss['action_loss'] = F.mse_loss(
            pred_v * mask.unsqueeze(-1),             # (B, C, act_dim)
            v_a_target * mask.unsqueeze(-1),
        )

        # ── 世界流损失（所有 C 步）────────────────────────────────
        if pred_w is not None and v_w_target is not None:
            loss['world_loss'] = F.mse_loss(
                pred_w * latent_mask.unsqueeze(-1),  # (B, C, latent_total)
                v_w_target * latent_mask.unsqueeze(-1),
            )
        else:
            loss['world_loss'] = torch.tensor(0.0, device=actions.device)

        # ── 耦合一致性损失 ────────────────────────────────────────
        if pred_w is not None and pred_v is not None:
            loss['coupling_loss'] = self.coupling_loss_fn(
                pred_w=pred_w,                       # (B, C, latent_total)
                pred_v=pred_v,                       # (B, C, act_dim)
                mask=mask[:, 0],                     # (B,) 用第0步代表整体有效性
            )
        else:
            loss['coupling_loss'] = torch.tensor(0.0, device=actions.device)

        # ── 加权总损失 ────────────────────────────────────────────
        loss['total_loss'] = (
            self.lambda_la * loss['latent_action_loss']
            + self.lambda_a  * loss['action_loss']
            + self.lambda_w  * loss['world_loss']
            + self.lambda_c  * loss['coupling_loss']
        )

        return loss

    def log(self, log_loss, eval_log_loss, cum_load_time, clock, epoch, batch_idx, step):
        for key in log_loss:
            log_loss[key] = self.accelerator.gather_for_metrics(log_loss[key]).mean()
        for key in eval_log_loss:
            eval_log_loss[key] = self.accelerator.gather_for_metrics(eval_log_loss[key]).mean()
        load_pecnt = torch.tensor(cum_load_time / (time()-clock)).to(self.device)
        load_pecnt = self.accelerator.gather_for_metrics(load_pecnt).mean()
        fps = (self.bs_per_gpu*self.print_steps*(self.diffusion_policy_config.world_model.sequence_length+1)) / (time()-clock)
        fps = self.accelerator.gather_for_metrics(torch.tensor(fps).to(self.device)).sum()

        text = 'Train Epoch: {} [{}/{} ({:.0f}%)] FPS:{:.5f} Load Pertentage:{:.5f} LR:{}'.format(
            epoch, 
            batch_idx * self.bs_per_gpu * self.accelerator.num_processes, 
            len(self.train_prefetcher), 
            100. * batch_idx * self.bs_per_gpu * self.accelerator.num_processes / len(self.train_prefetcher),
            fps,
            load_pecnt,
            self.scheduler.get_last_lr()[0],
        )
        for key in log_loss:
            text = text + ' {}_loss: {:.5f}'.format(key, log_loss[key])
        for key in eval_log_loss:
            text = text + ' eval_{}_loss: {:.5f}'.format(key, eval_log_loss[key])
        self.print(text)
        if self.is_main:
            for key in log_loss:
                self.writer.add_scalar(key+'_loss', log_loss[key], step)
            for key in eval_log_loss:
                self.writer.add_scalar('eval_'+key+'_loss', eval_log_loss[key], step)
            self.writer.add_scalar("learning rate", self.scheduler.get_last_lr()[0], step)
            self.writer.add_scalar("FPS", fps, step)
            self.writer.add_scalar("loading time in total time", load_pecnt, step)
