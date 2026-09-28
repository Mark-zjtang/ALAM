import torch
import torch.nn as nn
import torch.nn.functional as F
from uni_world_model.diffusion_policy.modules.action import ActionEncoder, ActionDecoder


class DiffusionPolicyModel(nn.Module):
    def __init__(
            self,
            world_model,
            diffusion_head,
            freeze_world_model,
            pretrained_model_path,
            dp_hidden_size,
            input_embedding_dim,
            noise_beta_alpha,
            noise_beta_beta,
            noise_s,
            num_inference_timesteps,
            num_timestep_buckets,
            act_pred,
            act_dim,
            add_pos_embed,
            latent_dim=None,          # LAM z_e 的维度，None 则自动从 world_model 读取
            use_joint_flow=True,      # 是否启用联合流
            **kwargs
    ):
        super(DiffusionPolicyModel, self).__init__()
        # ──────────────────────────────────────────────────
        # (不变) 原有属性赋值
        # ──────────────────────────────────────────────────
        self.world_model = world_model
        self.diffusion_head = diffusion_head
        self.freeze_world_model = freeze_world_model
        self.pretrained_model_path = pretrained_model_path
        self.dp_hidden_size = dp_hidden_size
        self.input_embedding_dim = input_embedding_dim
        self.noise_beta_alpha = noise_beta_alpha
        self.noise_beta_beta = noise_beta_beta
        self.noise_s = noise_s
        self.num_inference_timesteps = num_inference_timesteps
        self.num_timestep_buckets = num_timestep_buckets
        self.act_pred = act_pred
        self.act_dim = act_dim
        self.add_pos_embed = add_pos_embed

        # ──────────────────────────────────────────────────
        # (不变) 动作 encoder / decoder
        # ──────────────────────────────────────────────────
        self.action_encoder = ActionEncoder(
            action_dim=self.act_dim,
            hidden_size=self.input_embedding_dim,
        )
        self.action_decoder = ActionDecoder(
            input_dim=self.dp_hidden_size,
            hidden_dim=self.dp_hidden_size,
            output_dim=self.act_dim,
        )

        # ──────────────────────────────────────────────────
        # ★ 新增：联合流 world encoder / decoder
        # ──────────────────────────────────────────────────
        self.use_joint_flow = use_joint_flow
        if use_joint_flow:
            assert latent_dim is not None 
            # latent_dim 优先使用显式传入值，否则从 world_model 读取
            self.latent_dim = latent_dim 

            # world_encoder 与 action_encoder 结构相同，
            # 保证 w 和 a 进入 DiffusionHead 前处于同一嵌入空间
            self.world_encoder = ActionEncoder(
                action_dim=self.latent_dim,
                hidden_size=self.input_embedding_dim,
            )

            # world_decoder：DiffusionHead 隐层 → latent_dim
            self.world_decoder = nn.Sequential(
                nn.Linear(self.dp_hidden_size, self.dp_hidden_size),
                nn.SiLU(),
                nn.Linear(self.dp_hidden_size, self.latent_dim),
            )

        # ──────────────────────────────────────────────────
        # (不变) 位置编码
        # ──────────────────────────────────────────────────
        if self.add_pos_embed:
            max_seq_len = 1024
            self.position_embedding = nn.Embedding(max_seq_len, self.input_embedding_dim)
            nn.init.normal_(self.position_embedding.weight, mean=0.0, std=0.02)

        # ──────────────────────────────────────────────────
        # (不变) 加载预训练 world model
        # ──────────────────────────────────────────────────
        if self.pretrained_model_path:
            missing_keys, unexpected_keys = self.world_model.load_state_dict(
                torch.load(self.pretrained_model_path), strict=False
            )
            missing_root_keys = set([k.split(".")[0] for k in missing_keys])
            print('load ', self.pretrained_model_path,
                  '\nmissing ', missing_root_keys,
                  '\nunexpected ', unexpected_keys)

            if self.freeze_world_model:
                self.world_model.eval()
                for k, p in self.world_model.named_parameters():
                    p.requires_grad_(False)

    # ──────────────────────────────────────────────────────────
    # ★ 修改：forward() 新增 noisy_world 参数走联合流分支
    # ──────────────────────────────────────────────────────────
    def forward(
            self,
            rgb,                  # (B, 1, C, H, W)
            language,
            latent_action_ids,    # (B, H)  离散 id，给 world_model 做条件
            noisy_trajectory,     # (B, H, action_dim)
            t_discretized,        # (B,)
            noisy_world=None,     # ★ 新增: (B, H, latent_dim)  w 的噪声轨迹
            train=True,
            **kwargs
    ):
        # ──────────────────────────────────────────────
        # Step A: 通过 world_model 获取条件特征
        #         （逻辑不变，只做条件提取，不预测 w）
        # ──────────────────────────────────────────────
        if train:
            latent_action_pred = self.world_model(
                rgb, language, latent_action_ids, train=train
            )
        else:
            latent_action_pred = latent_action_ids   # inference 时直接传入 dict

        latent_action_feature = latent_action_pred['latent_action_feature']
        # (B, H, hidden)
        cond_input_feature    = latent_action_pred['cond_input_feature']
        # (B, n_cond, hidden)
        latent_action_preds   = latent_action_pred['latent_action_preds']
        # (B, H, codebook_size)  辅助分类损失用

        # ──────────────────────────────────────────────
        # Step B: 编码噪声轨迹
        # ──────────────────────────────────────────────
        action_features = self.action_encoder(noisy_trajectory, t_discretized)
        # (B, H, input_embedding_dim)

        # ★ 新增：编码噪声世界轨迹
        world_features = None
        if self.use_joint_flow and noisy_world is not None:
            world_features = self.world_encoder(noisy_world, t_discretized)
            # (B, H, input_embedding_dim)

        # ──────────────────────────────────────────────
        # Step C: 位置编码（原逻辑，仅作用在 action 上）
        # ──────────────────────────────────────────────
        if self.add_pos_embed:
            pos_ids  = torch.arange(
                action_features.shape[1],
                dtype=torch.long,
                device=action_features.device
            )
            pos_embs = self.position_embedding(pos_ids).unsqueeze(0)
            action_features = action_features + pos_embs

        # ──────────────────────────────────────────────
        # Step D: 拼接联合序列
        #
        #   joint_feat = [world_features(B,H,h) ‖ action_features(B,H,h)]
        #              → (B, 2H, h)
        #
        #   ⚠️  world 在前，action 在后，split 时依 world_len 切割
        # ──────────────────────────────────────────────
        if world_features is not None:
            sa_embs = torch.cat([world_features, action_features], dim=1)
            # (B, 2H, input_embedding_dim)
        else:
            sa_embs = action_features
            # (B, H, input_embedding_dim)

        # 条件序列（cross-attn key/value）
        vl_embs = torch.cat([latent_action_feature, cond_input_feature], dim=1)
        # (B, H+n_cond, hidden)

        # ──────────────────────────────────────────────
        # Step E: DiffusionHead（结构不变）
        # ──────────────────────────────────────────────
        model_output = self.diffusion_head(
            hidden_states=sa_embs,
            encoder_hidden_states=vl_embs,
            encoder_attention_mask=None,
            timestep=t_discretized,
        )
        # model_output: (B, 2H, dp_hidden_size) 或 (B, H, dp_hidden_size)

        # ──────────────────────────────────────────────
        # Step F: ★ 按 world_len 切割输出，分别解码
        # ──────────────────────────────────────────────
        pred_w = None
        if world_features is not None:
            world_len   = world_features.shape[1]           # H
            world_part  = model_output[:, :world_len]       # (B, H, dp_hidden_size)
            action_part = model_output[:, world_len:]       # (B, H, dp_hidden_size)

            # ★ world_decoder: (B, H, dp_hidden_size) → (B, H, latent_dim)
            pred_w = self.world_decoder(world_part)
        else:
            action_part = model_output

        # (不变) action_decoder
        pred   = self.action_decoder(action_part)
        pred_v = pred[:, -noisy_trajectory.shape[1]:]       # (B, H, action_dim)

        # ──────────────────────────────────────────────
        # Step G: 组装返回值
        # ──────────────────────────────────────────────
        res = {
            'latent_action_preds':   latent_action_preds,   # 辅助分类损失
            'pred_v':                pred_v,                 # 动作速度场预测
            'latent_action_feature': latent_action_feature,
        }
        if pred_w is not None:
            res['pred_w'] = pred_w                          # ★ 世界速度场预测

        return res

    # ──────────────────────────────────────────────────────────
    # (不变) 工具方法
    # ──────────────────────────────────────────────────────────
    @property
    def device(self):
        return next(self.parameters()).device

    def get_state_dict_to_save(self):
        modules_to_exclude = ['model_lang', 'model_vision']
        return {
            k: v for k, v in self.state_dict().items()
            if not any(m in k for m in modules_to_exclude)
        }
