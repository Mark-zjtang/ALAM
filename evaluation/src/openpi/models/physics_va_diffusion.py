import dataclasses
import logging
import math

import einops
import flax.nnx as nnx
import flax.nnx.bridge as nnx_bridge
import jax
import jax.numpy as jnp
from typing_extensions import override

from openpi.models import model as _model
import openpi.models.gemma as _gemma
import openpi.models.siglip as _siglip
from openpi.shared import array_typing as at
import openpi.shared.nnx_utils as nnx_utils

import torch
from functools import partial

logger = logging.getLogger("openpi")
# from diffusers.models import AutoencoderKL
import jax.nn as jnn

from openpi.models.pytorch_lam_wrapper import get_or_create_lam, _LAM_REGISTRY

from flax import nnx
import jax.numpy as jnp
import jax
import einops
import numpy as np


# ===========================================================================
# Cosine schedule 工具函数（diffusion 版专属）
# ===========================================================================
def _cosine_alpha_bar(t, s: float = 0.008):
    """Cosine schedule 的 alpha_bar(t)，t ∈ [0, 1]。

    满足 alpha_bar(0)=1（纯信号）, alpha_bar(1)=0（纯噪声）。
    Reference: Nichol & Dhariwal, "Improved DDPM", ICML 2021.
    """
    f = jnp.cos((t + s) / (1.0 + s) * math.pi * 0.5) ** 2
    f0 = math.cos(s / (1.0 + s) * math.pi * 0.5) ** 2
    return jnp.clip(f / f0, 1e-8, 1.0)


def _alpha_sigma(t, s: float = 0.008):
    """从连续时间 t 得到 (alpha_t, sigma_t)，满足 alpha^2 + sigma^2 = 1（VP-SDE）。"""
    ab = _cosine_alpha_bar(t, s=s)
    return jnp.sqrt(ab), jnp.sqrt(1.0 - ab)


def make_attn_mask(input_mask, mask_ar):
    """Adapted from big_vision.

    Tokens can attend to valid inputs tokens which have a cumulative mask_ar
    smaller or equal to theirs. This way `mask_ar` bool[?B, N] can be used to
    setup several types of attention, for example:

      [[1 1 1 1 1 1]]: pure causal attention.

      [[0 0 0 1 1 1]]: prefix-lm attention. The first 3 tokens can attend between
          themselves and the last 3 tokens have a causal attention. The first
          entry could also be a 1 without changing behaviour.

      [[1 0 1 0 1 0 0 1 0 0]]: causal attention between 4 blocks. Tokens of a
          block can attend all previous blocks and all tokens on the same block.

    Args:
      input_mask: bool[B, N] true if its part of the input, false if padding.
      mask_ar: bool[?B, N] mask that's true where previous tokens cannot depend on
        it and false where it shares the same attention mask as the previous token.
    """
    mask_ar = jnp.broadcast_to(mask_ar, input_mask.shape)
    cumsum = jnp.cumsum(mask_ar, axis=1)
    attn_mask = cumsum[:, None, :] <= cumsum[:, :, None]
    valid_mask = input_mask[:, None, :] * input_mask[:, :, None]
    return jnp.logical_and(attn_mask, valid_mask)


@at.typecheck
def posemb_sincos(
        pos: at.Real[at.Array, " b"], embedding_dim: int, min_period: float, max_period: float
) -> at.Float[at.Array, "b {embedding_dim}"]:
    """Computes sine-cosine positional embedding vectors for scalar positions."""
    if embedding_dim % 2 != 0:
        raise ValueError(f"embedding_dim ({embedding_dim}) must be divisible by 2")

    fraction = jnp.linspace(0.0, 1.0, embedding_dim // 2)
    period = min_period * (max_period / min_period) ** fraction
    sinusoid_input = jnp.einsum(
        "i,j->ij",
        pos,
        1.0 / period * 2 * jnp.pi,
        precision=jax.lax.Precision.HIGHEST,
    )
    return jnp.concatenate([jnp.sin(sinusoid_input), jnp.cos(sinusoid_input)], axis=-1)


@dataclasses.dataclass(frozen=True)
class Pi0Config(_model.BaseModelConfig):
    dtype: str = "bfloat16"
    paligemma_variant: _gemma.Variant = "gemma_2b"
    action_expert_variant: _gemma.Variant = "gemma_300m"

    # Set the model specific defaults.
    action_dim: int = 32  # default=32
    action_horizon: int = 6  # default=50
    max_token_len: int = 48  # default=48

    latent_action_pred: bool = False
    vq_remap: bool | None = None
    unknown_index: int | None = None
    phy_lam_ckpt: str | None = None

    # ------- diffusion 专属：cosine schedule 的 offset（有默认值，不影响已有 config） -------
    schedule_s: float = 0.008

    @property
    @override
    def model_type(self) -> _model.ModelType:
        return _model.ModelType.PI0

    @override
    def create(self, rng: at.KeyArrayLike) -> "Pi0":
        return Pi0(self, rngs=nnx.Rngs(rng))

    @override
    def inputs_spec(self, *, batch_size: int = 1) -> tuple[_model.Observation, _model.Actions]:
        image_spec = jax.ShapeDtypeStruct([batch_size, self.action_horizon, *_model.IMAGE_RESOLUTION, 3], jnp.float32)
        image_mask_spec = jax.ShapeDtypeStruct([batch_size, self.action_horizon], jnp.bool_)

        with at.disable_typechecking():
            observation_spec = _model.Observation(
                images={
                    "base_0_rgb": image_spec,
                    "left_wrist_0_rgb": image_spec,
                    "right_wrist_0_rgb": image_spec,
                },
                image_masks={
                    "base_0_rgb": image_mask_spec,
                    "left_wrist_0_rgb": image_mask_spec,
                    "right_wrist_0_rgb": image_mask_spec,
                },
                state=jax.ShapeDtypeStruct([batch_size, self.action_horizon, self.action_dim], jnp.float32),
                tokenized_prompt=jax.ShapeDtypeStruct([batch_size, self.action_horizon, self.max_token_len], jnp.int32),
                tokenized_prompt_mask=jax.ShapeDtypeStruct([batch_size, self.action_horizon, self.max_token_len], bool),
            )
        action_spec = jax.ShapeDtypeStruct([batch_size, self.action_horizon, self.action_dim], jnp.float32)

        return observation_spec, action_spec

    def get_freeze_filter(self) -> nnx.filterlib.Filter:
        """Returns the freeze filter based on the model config."""
        filters = []
        has_lora = False
        gemma_params_filter = nnx_utils.PathRegex(".*llm.*")
        action_expert_params_filter = nnx_utils.PathRegex(".*llm.*_1.*")
        if "lora" in self.paligemma_variant:
            filters.append(
                gemma_params_filter,
            )
            if "lora" not in self.action_expert_variant:
                filters.append(
                    nnx.Not(action_expert_params_filter),
                )
            has_lora = True
        elif "lora" in self.action_expert_variant:
            filters.append(
                action_expert_params_filter,
            )
            has_lora = True

        if has_lora:
            filters.append(
                nnx.Not(nnx_utils.PathRegex(".*lora.*")),
            )

        if not filters:
            return nnx.Nothing
        return nnx.All(*filters)


class Pi0(_model.BaseModel):
    """Pi0 policy (diffusion 版：v-prediction + DDIM + cosine schedule)。

    【接口不变】
      - compute_loss(rng, observation, actions, *, train, lam_out) -> (action_loss, rgb_loss, wrist_loss)
      - sample_actions(rng, observation, *, num_steps) -> actions

    【内部实现差异（相对 flow matching 版）】
      - 加噪：x_t = alpha_t * x + sigma_t * eps    （原：线性插值）
      - 目标：v = alpha_t * eps - sigma_t * x      （原：u = eps - x）
      - 采样：DDIM 单步反解（原：Euler 积分）

    网络结构、token 排布、KV cache、embed_prefix/embed_suffix 完全保留。
    RGB / Wrist / Action 三路全部同步切换到 diffusion。
    """

    def __init__(self, config: Pi0Config, rngs: nnx.Rngs):
        super().__init__(config.action_dim, config.action_horizon, config.max_token_len)
        self._schedule_s = config.schedule_s  # 供 _alpha_sigma 使用
        paligemma_config = _gemma.get_config(config.paligemma_variant)
        action_expert_config = _gemma.get_config(config.action_expert_variant)
        # TODO: rewrite gemma in NNX. For now, use bridge.
        llm = nnx_bridge.ToNNX(
            _gemma.Module(
                configs=[paligemma_config, action_expert_config],
                embed_dtype=config.dtype,
            )
        )
        llm.lazy_init(rngs=rngs, method="init")
        img = nnx_bridge.ToNNX(
            _siglip.Module(
                num_classes=paligemma_config.width,
                variant="So400m/14",
                pool_type="none",
                scan=True,
                dtype_mm=config.dtype,
            )
        )
        img.lazy_init(next(iter(config.fake_obs().images.values())), train=False, rngs=rngs)
        self.PaliGemma = nnx.Dict(llm=llm, img=img)
        # ── LAM（直接挂载 PyTorch 模型）──────────────
        if config.latent_action_pred:
            phy_lam = get_or_create_lam(pretrained_path=config.phy_lam_ckpt, device="cuda",)
            print("config.phy_lam_ckpt", config.phy_lam_ckpt)
            vq_remap = getattr(config, 'vq_remap', None)
            unknown_index = getattr(config, 'unknown_index', 'closest')
            if vq_remap is not None:
                phy_lam.setup_remap(vq_remap, unknown_index)

            lam_input_dim = (
                phy_lam.config.action_num_codes *
                phy_lam.config.latent_dim
            )  # 4 * 128 = 512
            lam_output_dim = action_expert_config.width * 2  # 2048
            self.lam_proj = nnx.Linear(
                in_features=lam_input_dim,
                out_features=lam_output_dim,
                rngs=rngs,
            )
        else:
            phy_lam = None

        self.latent_action_tokenizer = phy_lam

        self.state_proj = nnx.Linear(config.action_dim, action_expert_config.width, rngs=rngs)
        self.action_in_proj = nnx.Linear(config.action_dim, action_expert_config.width, rngs=rngs)
        self.action_time_mlp_in = nnx.Linear(2 * action_expert_config.width, action_expert_config.width, rngs=rngs)
        self.action_time_mlp_out = nnx.Linear(action_expert_config.width, action_expert_config.width, rngs=rngs)
        self.action_out_proj = nnx.Linear(action_expert_config.width, config.action_dim, rngs=rngs)

        new_rngs = nnx.Rngs(42)
        self.compress_dim = 2048
        self.rgb_in_proj = nnx.Linear(self.compress_dim, action_expert_config.width, rngs=new_rngs)
        self.rgb_time_mlp_in = nnx.Linear(2 * action_expert_config.width, action_expert_config.width, rngs=new_rngs)
        self.rgb_time_mlp_out = nnx.Linear(action_expert_config.width, action_expert_config.width, rngs=new_rngs)
        self.rgb_out_proj = nnx.Linear(action_expert_config.width, self.compress_dim, rngs=new_rngs)

        self.wrist_in_proj = nnx.Linear(self.compress_dim, action_expert_config.width, rngs=new_rngs)
        self.wrist_time_mlp_in = nnx.Linear(2 * action_expert_config.width, action_expert_config.width, rngs=new_rngs)
        self.wrist_time_mlp_out = nnx.Linear(action_expert_config.width, action_expert_config.width, rngs=new_rngs)
        self.wrist_out_proj = nnx.Linear(action_expert_config.width, self.compress_dim, rngs=new_rngs)

        self.image_chunk = 1
        self.action_seq_horizon = self.action_horizon - 1
        self.img_seq_horizon = (self.action_horizon - 1) * self.image_chunk  # default = self.action_horizon-1

    # ======================================================================
    # embed_prefix / embed_suffix：与 flow matching 版**完全一致**，不改
    # ======================================================================
    @at.typecheck
    def embed_prefix(
            self, obs: _model.Observation, lam_out: dict | None = None
    ) -> tuple[at.Float[at.Array, "b s emb"], at.Bool[at.Array, "b s"], at.Bool[at.Array, " s"], at.Float[
        at.Array, "..."]]:
        input_mask = []
        ar_mask = []
        current_tokens = []
        predict_tokens = []

        # embed images
        for name in obs.images:
            if name == "right_wrist_0_rgb":
                continue

            images = obs.images[name]
            B, T, H, W, C = images.shape
            print(f"{name}.shape", images.shape)

            if self.latent_action_tokenizer is not None and lam_out is not None:
                phys = lam_out[name]['physical_latent_action']
                print(f"Using precomputed LAM output for {name}, shape: {phys.shape}")
                phys = phys.squeeze(-1)
                w_seq = phys.reshape(B, T - 1, -1)
                w_seq = w_seq.astype(jnp.bfloat16)

                phy_lam_out = self.lam_proj(w_seq)
                print(f"phy_lam_out.shape for {name}", phy_lam_out.shape)
            else:
                phy_lam_out = jnp.zeros((B, 0, 2048), dtype=jnp.bfloat16)

            current_image_tokens, _ = self.PaliGemma.img(obs.images[name][:, :1], train=False)
            current_image_tokens = current_image_tokens.squeeze(1)
            print("current_image_tokens.shape", current_image_tokens.shape)
            predict_phy_lam_tokens = phy_lam_out
            print("predict_phy_lam_tokens.shape", predict_phy_lam_tokens.shape)

            current_tokens.append(current_image_tokens)
            predict_tokens.append(predict_phy_lam_tokens)
            input_mask.append(
                einops.repeat(
                    obs.image_masks[name],
                    "b -> b s",
                    s=current_image_tokens.shape[1],
                )
            )
            ar_mask += [False] * current_image_tokens.shape[1]

        if obs.tokenized_prompt is not None:
            tokenized_inputs = self.PaliGemma.llm(obs.tokenized_prompt, method="embed")
            print("tokenized_inputs.shape", tokenized_inputs.shape)
            current_tokens.append(tokenized_inputs)
            input_mask.append(obs.tokenized_prompt_mask)
            ar_mask += [False] * tokenized_inputs.shape[1]
        current_tokens = jnp.concatenate(current_tokens, axis=1)
        predict_tokens = jnp.concatenate(predict_tokens, axis=1)
        print("final_current_tokens.shape", current_tokens.shape)
        print("final_predict_tokens.shape", predict_tokens.shape)
        input_mask = jnp.concatenate(input_mask, axis=1)
        print("image_input_mask", input_mask.shape)
        ar_mask = jnp.array(ar_mask)
        return current_tokens, input_mask, ar_mask, predict_tokens

    @at.typecheck
    def embed_suffix(
            self, obs: _model.Observation, noisy_actions: _model.Actions, timestep: at.Float[at.Array, " b"],
            rgb_predict_tokens: at.Float[at.Array, "..."], wrist_predict_tokens: at.Float[at.Array, "..."]
    ) -> tuple[at.Float[at.Array, "..."], at.Bool[at.Array, "..."], at.Bool[at.Array, "..."]]:
        input_mask = []
        ar_mask = []
        tokens = []

        # add a single state token
        state_token = self.state_proj(obs.state)[:, None, :]
        tokens.append(state_token)
        input_mask.append(jnp.ones((obs.state.shape[0], 1), dtype=jnp.bool_))
        ar_mask += [True]

        time_emb = posemb_sincos(timestep, self.action_in_proj.out_features, min_period=4e-3, max_period=4.0)

        print("rgb_predict_tokens.shape", rgb_predict_tokens.shape)
        print("wrist_predict_tokens", wrist_predict_tokens.shape)
        print("nosie_actions.shape", noisy_actions.shape)

        rgb_tokens = self.rgb_in_proj(rgb_predict_tokens)
        wrist_tokens = self.wrist_in_proj(wrist_predict_tokens)
        action_tokens = self.action_in_proj(noisy_actions)

        time_tokens = einops.repeat(time_emb, "b emb -> b s emb", s=action_tokens.shape[1])
        time_tokens_images = einops.repeat(time_emb, "b emb -> b s emb", s=(action_tokens.shape[1]) * self.image_chunk)

        rgb_time_tokens = jnp.concatenate([rgb_tokens, time_tokens_images], axis=-1)
        rgb_time_tokens = self.rgb_time_mlp_in(rgb_time_tokens)
        rgb_time_tokens = nnx.swish(rgb_time_tokens)
        rgb_time_tokens = self.rgb_time_mlp_out(rgb_time_tokens)

        wrist_time_tokens = jnp.concatenate([wrist_tokens, time_tokens_images], axis=-1)
        wrist_time_tokens = self.wrist_time_mlp_in(wrist_time_tokens)
        wrist_time_tokens = nnx.swish(wrist_time_tokens)
        wrist_time_tokens = self.wrist_time_mlp_out(wrist_time_tokens)

        action_time_tokens = jnp.concatenate([action_tokens, time_tokens], axis=-1)
        action_time_tokens = self.action_time_mlp_in(action_time_tokens)
        action_time_tokens = nnx.swish(action_time_tokens)
        action_time_tokens = self.action_time_mlp_out(action_time_tokens)

        print("rgb_time_tokens.shape", rgb_time_tokens.shape)
        print("wrist_time_tokens.shape", wrist_time_tokens.shape)
        print("action_time_tokens", action_time_tokens.shape)

        batch_size = action_tokens.shape[0]
        time_steps = action_tokens.shape[1]

        for t in range(time_steps):
            tokens.append(rgb_time_tokens[:, t:t + 1, :])
            input_mask.append(jnp.ones((batch_size, 1), dtype=jnp.bool_))

            tokens.append(wrist_time_tokens[:, t:t + 1, :])
            input_mask.append(jnp.ones((batch_size, 1), dtype=jnp.bool_))

            tokens.append(action_time_tokens[:, t:t + 1, :])
            input_mask.append(jnp.ones((batch_size, 1), dtype=jnp.bool_))

        tokens = jnp.concatenate(tokens, axis=1)
        input_mask = jnp.concatenate(input_mask, axis=1)

        for t in range(time_steps):
            ar_mask.append(True)   # RGB
            ar_mask.append(False)  # Wrist
            ar_mask.append(True)   # Action

        print("final.tokens.shape", tokens.shape)
        print("final.input_mask.shape", input_mask.shape)
        ar_mask = jnp.array(ar_mask)
        print("final.ar_mask.shape", ar_mask.shape)

        return tokens, input_mask, ar_mask

    # ======================================================================
    # 训练 loss：v-prediction（diffusion 版）
    # ======================================================================
    @override
    def compute_loss(
            self, rng: at.KeyArrayLike, observation: _model.Observation, actions: _model.Actions, *, train: bool = False, lam_out: dict | None = None
    ) -> at.Float[at.Array, "*b ah"]:
        preprocess_rng, noise_rng, rgb_noise_rng, wrist_noise_rng, time_rng = jax.random.split(rng, 5)
        observation = _model.preprocess_observation(preprocess_rng, observation, train=train)
        print("===================================================compute_loss==========================================")
        actions = actions[:, :-1]
        batch_shape = actions.shape[:-2]
        noise = jax.random.normal(noise_rng, actions.shape)

        # ------------- diffusion: 采样时间 t ~ Beta(1.5, 1)（偏大的 t，配合 v-prediction 训练更稳）-------------
        # 保留原先的 Beta 采样习惯；也可以改成 uniform（见注释）
        time = jax.random.beta(time_rng, 1.5, 1, batch_shape) * 0.999 + 0.001  # (B,)
        # time = jax.random.uniform(time_rng, batch_shape, minval=0.001, maxval=0.999)

        # ------------- 构造 alpha_t, sigma_t ---------------------------------
        alpha_t_1d, sigma_t_1d = _alpha_sigma(time, s=self._schedule_s)     # (B,), (B,)
        alpha_t = alpha_t_1d[..., None, None].astype(actions.dtype)         # (B, 1, 1) for action
        sigma_t = sigma_t_1d[..., None, None].astype(actions.dtype)

        # ------------- action: 加噪 + v-target -------------------------------
        x_t = alpha_t * actions + sigma_t * noise
        u_t = alpha_t * noise - sigma_t * actions                            # v-prediction target

        prefix_tokens, prefix_mask, prefix_ar_mask, predict_tokens = self.embed_prefix(observation, lam_out=lam_out)
        print("prefix_tokens.shape", prefix_tokens.shape)

        rgb_pred = predict_tokens[:, : self.img_seq_horizon]
        wrist_pred = predict_tokens[:, self.img_seq_horizon: 2 * self.img_seq_horizon]

        rgb_pred_noise = jax.random.normal(rgb_noise_rng, rgb_pred.shape)
        wrist_pred_noise = jax.random.normal(wrist_noise_rng, wrist_pred.shape)

        # ------------- rgb: 加噪 + v-target ---------------------------------
        alpha_t_img = alpha_t_1d[..., None, None].astype(rgb_pred.dtype)     # (B, 1, 1) for rgb/wrist
        sigma_t_img = sigma_t_1d[..., None, None].astype(rgb_pred.dtype)

        xr_t = alpha_t_img * rgb_pred + sigma_t_img * rgb_pred_noise
        ur_t = alpha_t_img * rgb_pred_noise - sigma_t_img * rgb_pred

        xw_t = alpha_t_img * wrist_pred + sigma_t_img * wrist_pred_noise
        uw_t = alpha_t_img * wrist_pred_noise - sigma_t_img * wrist_pred

        # ------------- 组装 suffix 并前向 -------------------------------------
        suffix_tokens, suffix_mask, suffix_ar_mask = self.embed_suffix(observation, x_t, time, xr_t, xw_t)
        print("suffix_tokens.shape", suffix_tokens.shape)
        input_mask = jnp.concatenate([prefix_mask, suffix_mask], axis=1)
        ar_mask = jnp.concatenate([prefix_ar_mask, suffix_ar_mask], axis=0)
        attn_mask = make_attn_mask(input_mask, ar_mask)
        positions = jnp.cumsum(input_mask, axis=1) - 1

        (prefix_out, suffix_out), _ = self.PaliGemma.llm([prefix_tokens, suffix_tokens], mask=attn_mask, positions=positions)

        print("suffix_out_0.shape:", suffix_out.shape)

        # token order: [state, rgb_t1, wrist_t1, action_t1, rgb_t2, wrist_t2, action_t2 ...]
        state_idx = 0

        rgb_start_idx = 1
        rgb_indices = jnp.arange(rgb_start_idx, rgb_start_idx + 3 * self.action_seq_horizon, 3)

        wrist_start_idx = 2
        wrist_indices = jnp.arange(wrist_start_idx, wrist_start_idx + 3 * self.action_seq_horizon, 3)

        action_start_idx = 3
        action_indices = jnp.arange(action_start_idx, action_start_idx + 3 * self.action_seq_horizon, 3)

        vr_t = self.rgb_out_proj(suffix_out[:, rgb_indices])
        print("vr_t.shape", vr_t.shape)

        vw_t = self.wrist_out_proj(suffix_out[:, wrist_indices])
        print("vw_t.shape", vw_t.shape)

        v_t = self.action_out_proj(suffix_out[:, action_indices])
        print("v_t.shape", v_t.shape)

        # diffusion v-prediction L1 loss（和原先一样对最后一维取 mean，保持 loss 形状不变）
        return (
            jnp.mean(jnp.abs(v_t - u_t), axis=-1),
            jnp.mean(jnp.abs(vr_t - ur_t), axis=-1),
            jnp.mean(jnp.abs(vw_t - uw_t), axis=-1),
        )

    # ======================================================================
    # 推理采样：DDIM（diffusion 版）
    # ======================================================================
    @override
    def sample_actions(
            self,
            rng: at.KeyArrayLike,
            observation: _model.Observation,
            *,
            num_steps: int = 10,  # 必须是 Python int（静态展开）
    ) -> _model.Actions:
        """DDIM 确定性采样（接口与 flow matching 版完全一致）。

        三路（action / rgb / wrist）共享同一个时间 t 的 schedule，
        每一步：
          1) 网络预测 v_hat（三路各一个）
          2) 从 v_hat 反解 (x_0_hat, eps_hat)
          3) 按 t_next 的 alpha/sigma 重组得到 x_{t_next}
        """
        preprocess_rng, noise_rng, rgb_noise_rng, wrist_noise_rng, time_rng = jax.random.split(rng, 5)

        img_seq_horizon = self.action_seq_horizon * self.image_chunk  # default = self.action_horizon-1
        action_seq_horizon = self.action_seq_horizon
        observation = _model.preprocess_observation(None, observation, train=False)

        batch_size = observation.state.shape[0]
        print("=============================action_seq_horizon================================", self.action_seq_horizon)

        # ---- 三路都从纯噪声开始（t=1 对应 alpha=0, sigma=1，纯高斯）----------
        x_t = jax.random.normal(noise_rng, (batch_size, action_seq_horizon, self.action_dim))
        print("x_t.shape", x_t.shape)
        xr_t = jax.random.normal(rgb_noise_rng, (batch_size, img_seq_horizon, 2048))
        print("xr_t.shape", xr_t.shape)
        xw_t = jax.random.normal(wrist_noise_rng, (batch_size, img_seq_horizon, 2048))
        print("xw_t.shape", xw_t.shape)

        # ---- 预先跑一次 prefix，得到 kv_cache ------------------------------
        prefix_tokens, prefix_mask, prefix_ar_mask, predict_tokens = self.embed_prefix(observation)
        prefix_attn_mask = make_attn_mask(prefix_mask, prefix_ar_mask)
        positions = jnp.cumsum(prefix_mask, axis=1) - 1
        _, kv_cache = self.PaliGemma.llm([prefix_tokens, None], mask=prefix_attn_mask, positions=positions)

        # ---- DDIM 时间 grid：t = 1.0 -> 0.0，num_steps 步均匀切分 ------------
        t_grid = jnp.linspace(1.0, 0.0, num_steps + 1)  # (num_steps + 1,)

        # ---- 单步 DDIM：给定 (x_t, xr_t, xw_t, t_curr, t_next) 返回下一步 ----
        def _one_step(x_t, xr_t, xw_t, t_curr, t_next):
            time_for_inference = jnp.full((batch_size,), t_curr)  # (B,)

            suffix_tokens, suffix_mask, suffix_ar_mask = self.embed_suffix(
                observation, x_t, time_for_inference, xr_t, xw_t
            )

            suffix_attn_mask = make_attn_mask(suffix_mask, suffix_ar_mask)
            prefix_attn_mask_local = einops.repeat(prefix_mask, "b p -> b s p", s=suffix_tokens.shape[1])
            full_attn_mask = jnp.concatenate([prefix_attn_mask_local, suffix_attn_mask], axis=-1)
            assert full_attn_mask.shape == (
                batch_size,
                suffix_tokens.shape[1],
                prefix_tokens.shape[1] + suffix_tokens.shape[1],
            )
            positions_local = jnp.sum(prefix_mask, axis=-1)[:, None] + jnp.cumsum(suffix_mask, axis=-1) - 1

            (prefix_out, suffix_out), _ = self.PaliGemma.llm(
                [None, suffix_tokens],
                mask=full_attn_mask,
                positions=positions_local,
                kv_cache=kv_cache,
            )
            assert prefix_out is None

            # token order: [state, rgb_t1, wrist_t1, action_t1, ...]
            rgb_start_idx = 1
            rgb_indices = jnp.arange(rgb_start_idx, rgb_start_idx + 3 * self.action_seq_horizon, 3)

            wrist_start_idx = 2
            wrist_indices = jnp.arange(wrist_start_idx, wrist_start_idx + 3 * self.action_seq_horizon, 3)

            action_start_idx = 3
            action_indices = jnp.arange(action_start_idx, action_start_idx + 3 * self.action_seq_horizon, 3)

            # v-predictions for the three streams
            vr_t_pred = self.rgb_out_proj(suffix_out[:, rgb_indices])
            vw_t_pred = self.wrist_out_proj(suffix_out[:, wrist_indices])
            v_t_pred = self.action_out_proj(suffix_out[:, action_indices])

            # ---- DDIM: v -> (x0_hat, eps_hat) -> x_{next} -----------------
            #   x0_hat = alpha_t * x_t - sigma_t * v_hat
            #   eps_hat = sigma_t * x_t + alpha_t * v_hat
            #   x_next = alpha_next * x0_hat + sigma_next * eps_hat
            alpha_c, sigma_c = _alpha_sigma(t_curr, s=self._schedule_s)
            alpha_n, sigma_n = _alpha_sigma(t_next, s=self._schedule_s)

            def _ddim_update(x, v, dt_out_dtype):
                a_c = alpha_c.astype(dt_out_dtype)
                s_c = sigma_c.astype(dt_out_dtype)
                a_n = alpha_n.astype(dt_out_dtype)
                s_n = sigma_n.astype(dt_out_dtype)
                x0_hat = a_c * x - s_c * v
                eps_hat = s_c * x + a_c * v
                return a_n * x0_hat + s_n * eps_hat

            x_next = _ddim_update(x_t, v_t_pred, x_t.dtype)
            xr_next = _ddim_update(xr_t, vr_t_pred, xr_t.dtype)
            xw_next = _ddim_update(xw_t, vw_t_pred, xw_t.dtype)
            return x_next, xr_next, xw_next

        # ---- 静态展开 DDIM 循环（num_steps 是 Python int）-------------------
        for i in range(num_steps):
            t_curr = t_grid[i]
            t_next = t_grid[i + 1]
            x_t, xr_t, xw_t = _one_step(x_t, xr_t, xw_t, t_curr, t_next)

        return x_t