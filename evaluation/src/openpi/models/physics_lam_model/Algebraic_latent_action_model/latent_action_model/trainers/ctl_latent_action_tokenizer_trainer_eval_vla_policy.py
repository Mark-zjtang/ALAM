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
from uni_world_model.data.datasets import DataPrefetcher
import omegaconf
from glob import glob
import shutil
from collections import defaultdict
from uni_world_model.latent_action_model.trainers.trainer_utils import visualize_latent_motion_reconstruction, visualize_eval_lam_reconstruction, calculate_psl_metrics
from uni_world_model.latent_action_model.trainers.optimizer import get_optimizer, LinearWarmup_CosineAnnealing
from contextlib import contextmanager
from IPython import embed

class LatentActionTokenizerTrainer(nn.Module):
    def __init__(
        self,
        latent_action_tokenizer,
        rgb_preprocessor,
        train_dataloader,
        eval_dataloader,
        save_path,
        save_epochs=1,
        save_steps=10000,
        num_epochs=20,
        print_steps=100,
        lr_max=0.0001,
        weight_decay=0.,
        num_warmup_epochs=1,
        gradient_accumulation_steps=4,
        resume_ckpt_path=None,
        bs_per_gpu=32,
        max_epoch=None,
    ):
        super(LatentActionTokenizerTrainer, self).__init__()
        if resume_ckpt_path is not None:
            print(f"resuming Latent Motion Tokenizer from {resume_ckpt_path} ...")
            missing_keys, unexpected_keys = latent_action_tokenizer.load_state_dict(torch.load(os.path.join(resume_ckpt_path, 'pytorch_model.bin'), map_location='cpu'), strict=False)
            missing_root_keys = set([k.split(".")[0] for k in missing_keys])
            print('load ', resume_ckpt_path, '\nmissing ', missing_root_keys, '\nunexpected ', unexpected_keys)
        
        ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True) # default setting fasle
        accelerator= Accelerator(
            gradient_accumulation_steps=gradient_accumulation_steps,
            kwargs_handlers=[ddp_kwargs]
        )
        self.gradient_accumulation_steps = gradient_accumulation_steps

        total_prints_per_epoch = len(train_dataloader.dataset) // (print_steps * bs_per_gpu * accelerator.num_processes)

        optimizer = get_optimizer(
                        [p for n, p in latent_action_tokenizer.named_parameters() if p.requires_grad], 
                        lr=lr_max, 
                        wd=weight_decay
                    )
        
        linear_warmup_total_iters = min(num_warmup_epochs*total_prints_per_epoch, 5000000 // (print_steps * bs_per_gpu * accelerator.num_processes))
        scheduler = LinearWarmup_CosineAnnealing(
                        optimizer=optimizer,
                        linear_warmup_start_factor=0.5,
                        linear_warmup_total_iters=linear_warmup_total_iters,
                        cosine_annealing_T_max=num_epochs*total_prints_per_epoch-linear_warmup_total_iters,
                        cosine_annealing_eta_min=5e-5
                    )
        # default setting
        latent_action_tokenizer, optimizer, train_dataloader, eval_dataloader = accelerator.prepare(
            latent_action_tokenizer, optimizer, train_dataloader, eval_dataloader, 
            device_placement=[True, True, False, False]
        )

        # latent_action_tokenizer, optimizer, train_dataloader, eval_dataloader = accelerator.prepare(
        #     latent_action_tokenizer, optimizer, train_dataloader, eval_dataloader, 
        # )
        
        self.writer = SummaryWriter(os.path.join(save_path, 'logs'))
        self.accelerator = accelerator
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.total_prints_per_epoch = total_prints_per_epoch
        self.latent_action_tokenizer = latent_action_tokenizer
        self.optimizer = optimizer
        self.train_prefetcher = DataPrefetcher(train_dataloader, self.device)
        self.eval_prefetcher = DataPrefetcher(eval_dataloader, self.device)
        self.rgb_preprocessor = rgb_preprocessor.to(self.device)
        self.save_path = save_path
        self.save_epochs = save_epochs
        self.save_steps = save_steps
        self.max_epoch = max_epoch
        self.num_epochs = num_epochs
        self.print_steps = print_steps
        self.bs_per_gpu = bs_per_gpu


    @property
    def device(self):
        return self.accelerator.device

    @property
    def is_main(self):
        return self.accelerator.is_main_process

    @property
    def process_index(self):
        return self.accelerator.process_index

    def print(self, msg):
        self.accelerator.print(msg)

    def save_checkpoint(self, save_dir):
        unwrapped_latent_action_tokenizer = self.accelerator.unwrap_model(self.latent_action_tokenizer)
        state_dict = unwrapped_latent_action_tokenizer.get_state_dict_to_save()
        
        torch.save(state_dict, os.path.join(save_dir, "pytorch_model.bin"))
        omegaconf.OmegaConf.save(unwrapped_latent_action_tokenizer.config, os.path.join(save_dir, "config.yaml"))

        self.print(f"A new model checkpoint is saved to {save_dir}!!!")
        
    def train(self):
        eval_loss_steps = len(self.train_prefetcher) // len(self.eval_prefetcher)
        step = 0
        
        for epoch in range(self.num_epochs+1):
            if epoch != 0:
                self.accelerator.wait_for_everyone()
                save_dir = os.path.join(self.save_path, f'saved_epoch_{epoch}_step_{step}')

                if self.is_main:
                    os.makedirs(save_dir, exist_ok=True)
                    self.save_checkpoint(save_dir)
                    
                visualization_dir = os.path.join(save_dir, 'visualization')
                self.eval_latent_motion_reconstruction(visualization_dir)

                if epoch == self.num_epochs:
                    break
                if (self.max_epoch is not None) and (epoch >= self.max_epoch):
                    break

            log_loss = {}
            eval_log_loss = {}
            
            cum_load_time = 0 
            clock = time()
            batch_idx = 0
            batch, load_time = self.train_prefetcher.next()
            print("batch keys: ", batch.keys())
            
            while batch is not None:
                with self.accelerator.accumulate(self.latent_action_tokenizer):

                    self.latent_action_tokenizer.train()
                    self.optimizer.zero_grad()
                    loss = self.calculate_loss(batch, train=True)
                    self.accelerator.backward(loss['loss'])
                    self.optimizer.step()

                    for key in loss:
                        if key not in log_loss:
                            log_loss[key] = 0.0
                        log_loss[key] += loss[key].detach() / self.print_steps

                    cum_load_time += load_time / self.print_steps

                if (batch_idx+1) % self.print_steps == 0:

                    with torch.no_grad():
                        batch, _ = self.eval_prefetcher.next_without_none()
                        self.latent_action_tokenizer.eval()
                        loss = self.calculate_loss(batch, train=True)
                        for key in loss:
                            eval_log_loss[key] = loss[key].detach()

                    self.log(log_loss, eval_log_loss, cum_load_time, clock, epoch, batch_idx, step)
                    log_loss = {}
                    eval_log_loss = {}

                    cum_load_time = 0
                    clock = time()
                    self.scheduler.step()

                if step % self.save_steps == 0:
                    self.accelerator.wait_for_everyone()
                    save_dir = os.path.join(self.save_path, f'temp_epoch_{epoch}_step_{step}')
                    if self.is_main:
                        existing_ckpt_dirs = glob(os.path.join(self.save_path, f'temp_epoch_*_step_*'))
                        for existing_ckpt_dir in existing_ckpt_dirs:
                            if existing_ckpt_dir != save_dir:
                                shutil.rmtree(existing_ckpt_dir)
                        os.makedirs(save_dir, exist_ok=True)
                        self.save_checkpoint(save_dir)

                    visualization_dir = os.path.join(save_dir, 'visualization')
                    self.eval_latent_motion_reconstruction(visualization_dir)

                batch_idx += 1
                step += 1
                batch, load_time = self.train_prefetcher.next()



    # @torch.no_grad()
    # def eval_latent_motion_reconstruction(self, visualization_dir):
    #     os.makedirs(visualization_dir, exist_ok=True)
    #     self.print(f"Saving visualization results to {visualization_dir} ...")
    #     batch, _ = self.eval_prefetcher.next_without_none()
        

    #     orig_rgb_seq = torch.cat([batch['rgb_initial'], batch['rgb_future']], dim=1) # (b, 2, c, h, w)
    #     rgb_seq = self.rgb_preprocessor(orig_rgb_seq, train=True)

    #     self.latent_action_tokenizer.eval()
    #     ## default lam/lam_dino calvin series methods
    #     ## ctl lam/lam_dino calvin series methods (need continues three frames)
  
    #     outputs = self.latent_action_tokenizer(
    #         cond_pixel_values=rgb_seq[:,0],
    #         target_pixel_values=rgb_seq[:,1],
    #         next_target_pixel_values=rgb_seq[:,2],
    #         return_recons_only=True
    #     )


        
    #     #####################  key argument return_recons_only : if continue is not an argument(removed) else is true argument
            
    #     recons_rgb_future_01 = self.rgb_preprocessor.post_process(outputs["recon_pixel_values_01"]).detach().cpu()  # (b, c, h, w)
    #     gt_latent_motion_ids_01 = outputs["indices_01"].detach().cpu() # (b, per_latent_motion_len)
        
    #     recons_rgb_future_12 = self.rgb_preprocessor.post_process(outputs["recon_pixel_values_12"]).detach().cpu()  # (b, c, h, w)
    #     gt_latent_motion_ids_12 = outputs["indices_12"].detach().cpu() # (b, per_latent_motion_len)
        
    #     recons_rgb_future_02 = self.rgb_preprocessor.post_process(outputs["recon_pixel_values_02"]).detach().cpu()  # (b, c, h, w)
    #     gt_latent_motion_ids_02 = outputs["indices_02"].detach().cpu() # (b, per_latent_motion_len)
        
        
    #     # orig_rgb_seq = orig_rgb_seq.detach().cpu()
    #     orig_rgb_seq = self.rgb_preprocessor.post_process(rgb_seq).detach().cpu()

    #     for i in range(orig_rgb_seq.shape[0]):
    #         visualize_eval_lam_reconstruction(
    #             gt_images=[orig_rgb_seq[i,0], orig_rgb_seq[i,1], orig_rgb_seq[i,2], orig_rgb_seq[i,2]],
    #             rec_images=[recons_rgb_future_01[i], recons_rgb_future_12[i], recons_rgb_future_02[i]],
    #             latent_motion_ids=[gt_latent_motion_ids_01[i], gt_latent_motion_ids_12[i], gt_latent_motion_ids_02[i]],
    #             path=os.path.join(visualization_dir, f"{self.process_index}-{i}-eval.png")
    #         )
    
    @torch.no_grad()
    def eval_latent_motion_reconstruction(self, visualization_dir):
        os.makedirs(visualization_dir, exist_ok=True)
        self.print(f"Saving visualization results to {visualization_dir} ...")
        batch, _ = self.eval_prefetcher.next_without_none()
        

        orig_rgb_seq = torch.cat([batch['rgb_initial'], batch['rgb_future']], dim=1) # (b, 2, c, h, w)
        rgb_seq = self.rgb_preprocessor(orig_rgb_seq, train=True)

        self.latent_action_tokenizer.eval()
        ## default lam/lam_dino calvin series methods
        ## ctl lam/lam_dino calvin series methods (need continues three frames)
  
        outputs = self.latent_action_tokenizer(
            cond_pixel_values=rgb_seq[:,0],
            target_pixel_values=rgb_seq[:,1],
            next_target_pixel_values=rgb_seq[:,2],
            return_recons_only=True
        )

        #####################  key argument return_recons_only : if continue is not an argument(removed) else is true argument
            
        recons_rgb_future_01 = self.rgb_preprocessor.post_process(outputs["recon_pixel_values_01"]).detach().cpu()  # (b, c, h, w)
        
        gt_latent_motion_ids_01 = outputs["indices_01"].detach().cpu() # (b, per_latent_motion_len)
        pred_phys_a_01 = outputs["phys_a_01"].detach().cpu() # (b, physical_a_dim)
        
        recons_rgb_future_12 = self.rgb_preprocessor.post_process(outputs["recon_pixel_values_12"]).detach().cpu()  # (b, c, h, w)
        gt_latent_motion_ids_12 = outputs["indices_12"].detach().cpu() # (b, per_latent_motion_len)
        pred_phys_a_12 = outputs["phys_a_12"].detach().cpu() # (b, physical_a_dim)

        recons_rgb_future_02 = self.rgb_preprocessor.post_process(outputs["recon_pixel_values_02"]).detach().cpu()  # (b, c, h, w)
        gt_latent_motion_ids_02 = outputs["indices_02"].detach().cpu() # (b, per_latent_motion_len)
        pred_phys_a_02 = outputs["phys_a_02"].detach().cpu() # (b, physical_a_dim)

        pred_phys_a_add_02 = self.latent_action_tokenizer.decode_additivity_lam(pred_phys_a_01.to(self.device), pred_phys_a_12.to(self.device))
        gt_latent_motion_ids_02_additivity = pred_phys_a_add_02["indices"].detach().cpu()

        orig_rgb_seq = self.rgb_preprocessor.post_process(rgb_seq).detach().cpu()

        recons_rgb_future_01_metrics = []
        recons_rgb_future_12_metrics = []
        recons_rgb_future_02_metrics = []
        recons_rgb_future_02_additivity_metrics = []

        for i in range(orig_rgb_seq.shape[0]):
            recons_rgb_future_01 = self.latent_action_tokenizer.decode_image(orig_rgb_seq[i,0].unsqueeze(0).to(self.device), gt_latent_motion_ids_01[i].to(self.device))["recon_pixel_values"][0]
            recons_rgb_future_01_metrics.append(recons_rgb_future_01)
            recons_rgb_future_12 = self.latent_action_tokenizer.decode_image(orig_rgb_seq[i,1].unsqueeze(0).to(self.device), gt_latent_motion_ids_12[i].to(self.device))["recon_pixel_values"][0]
            recons_rgb_future_12_metrics.append(recons_rgb_future_12) 
            recons_rgb_future_02 = self.latent_action_tokenizer.decode_image(orig_rgb_seq[i,0].unsqueeze(0).to(self.device), gt_latent_motion_ids_02[i].to(self.device))["recon_pixel_values"][0]
            recons_rgb_future_02_metrics.append(recons_rgb_future_02)
            recons_rgb_future_02_additivity = self.latent_action_tokenizer.decode_image(orig_rgb_seq[i,0].unsqueeze(0).to(self.device), gt_latent_motion_ids_02_additivity[i].to(self.device))["recon_pixel_values"][0]
            recons_rgb_future_02_additivity_metrics.append(recons_rgb_future_02_additivity)
            
            visualize_eval_lam_reconstruction(
                gt_images=[orig_rgb_seq[i,0], orig_rgb_seq[i,1], orig_rgb_seq[i,2], orig_rgb_seq[i,2], orig_rgb_seq[i,2]],
                rec_images=[recons_rgb_future_01, recons_rgb_future_12, recons_rgb_future_02, recons_rgb_future_02_additivity],
                latent_motion_ids=[gt_latent_motion_ids_01[i], gt_latent_motion_ids_12[i], gt_latent_motion_ids_02[i], gt_latent_motion_ids_02_additivity[i]],
                path=os.path.join(visualization_dir, f"{self.process_index}-{i}-eval.png")
            )
        
        recon_pixel_values_01_metrics = torch.stack(recons_rgb_future_01_metrics, dim=0)
        print("recon_pixel_values_01_metrics shape: ", recon_pixel_values_01_metrics.shape)
        recon_pixel_values_12_metrics = torch.stack(recons_rgb_future_12_metrics, dim=0)
        print("recon_pixel_values_12_metrics shape:", recon_pixel_values_12_metrics.shape)
        recon_pixel_values_02_metrics = torch.stack(recons_rgb_future_02_metrics, dim=0)
        print("recon_pixel_values_02_metrics shape:", recon_pixel_values_02_metrics.shape)
        recon_pixel_values_02_additivity_metrics = torch.stack(recons_rgb_future_02_additivity_metrics, dim=0)
        print("recon_pixel_values_02_additivity_metrics shape:", recon_pixel_values_02_additivity_metrics.shape)
        calculate_psl_metrics(orig_rgb_seq[:,1], recon_pixel_values_01_metrics)
        calculate_psl_metrics(orig_rgb_seq[:,2], recon_pixel_values_12_metrics)
        calculate_psl_metrics(orig_rgb_seq[:,2], recon_pixel_values_02_metrics)
        calculate_psl_metrics(orig_rgb_seq[:,2], recon_pixel_values_02_additivity_metrics)


    def calculate_loss(self, batch, train):
        # image preprocessing
        rgb_seq = torch.cat([batch['rgb_initial'], batch['rgb_future']], dim=1)
        rgb_seq = self.rgb_preprocessor(rgb_seq, train=train)

        # compute loss
        outputs = self.latent_action_tokenizer(
            cond_pixel_values=rgb_seq[:,0],
            target_pixel_values=rgb_seq[:,1],
            next_target_pixel_values=rgb_seq[:,2],
            return_recons_only=False)

        return outputs

    def log(self, log_loss, eval_log_loss, cum_load_time, clock, epoch, batch_idx, step):
        for key in log_loss:
            log_loss[key] = self.accelerator.gather_for_metrics(log_loss[key]).mean()
        for key in eval_log_loss:
            eval_log_loss[key] = self.accelerator.gather_for_metrics(eval_log_loss[key]).mean()
        load_pecnt = torch.tensor(cum_load_time / (time()-clock)).to(self.device)
        load_pecnt = self.accelerator.gather_for_metrics(load_pecnt).mean()
        fps = (self.bs_per_gpu*self.print_steps*2) / (time()-clock)
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
            text = text + ' {}: {:.5f}'.format(key, log_loss[key])
        for key in eval_log_loss:
            text = text + ' eval_{}: {:.5f}'.format(key, eval_log_loss[key])
        self.print(text)
        if self.is_main:
            for key in log_loss:
                self.writer.add_scalar(key, log_loss[key], step)
            for key in eval_log_loss:
                self.writer.add_scalar('eval_'+key, eval_log_loss[key], step)
            self.writer.add_scalar("learning rate", self.scheduler.get_last_lr()[0], step)
            self.writer.add_scalar("FPS", fps, step)
            self.writer.add_scalar("loading time in total time", load_pecnt, step)
