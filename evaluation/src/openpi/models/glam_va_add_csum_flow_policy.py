import dataclasses
import logging

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




# ========== SALI: Measured training cumsum std ==========
# key: dataset_name -> {"rgb": (H,), "wrist": (H,)}
# 采集方式：compute_loss 里对 >= 20 个 batch 统计 per-position std，取均值
# 注意：顶层用 np.array，避免 import 时触发 JAX backend 初始化（GPU OOM 时会失败）

SALI_MEASURED_STATS = {
    "metaworld": {
        "rgb":   np.array([0.00255, 0.00441, 0.00604, 0.00754, 0.00887], dtype=np.float32),
        "wrist": np.array([0.00342, 0.00685, 0.01027, 0.01370, 0.01712], dtype=np.float32),
    },
    "libero": {
        "rgb": np.array([
            0.0075, 0.0113, 0.0148, 0.0180, 0.0210,
            0.0238, 0.0265, 0.0290, 0.0312, 0.0332,
            0.0350, 0.0367, 0.0381, 0.0395, 0.0408,
            0.0420, 0.0430, 0.0438, 0.0445, 0.0452,
        ], dtype=np.float32),
        "wrist": np.array([
            0.0400, 0.0500, 0.0580, 0.0650, 0.0720,
            0.0780, 0.0830, 0.0880, 0.0920, 0.0960,
            0.1000, 0.1040, 0.1070, 0.1100, 0.1130,
            0.1150, 0.1170, 0.1190, 0.1210, 0.1220,
        ], dtype=np.float32),
    },
    "real_world": {
        "rgb":   None,
        "wrist": None,
    },
}

# =========================================================


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


def build_la_cumsum(la_single: jnp.ndarray) -> jnp.ndarray:
    """根据单步 latent 序列构造"累积 latent"。

    新规则（标准前缀和，无特殊位置）：
      la_cumsum[:, t] = sum(la_single[:, 0..t])  for all t >= 0

    示例（T=5）:
      [L0,  L0+L1,  L0+L1+L2,  L0+L1+L2+L3,  L0+L1+L2+L3+L4]
    """
    return jnp.cumsum(la_single, axis=1)

def build_la_linear_extrap(la_current: jnp.ndarray, T: int) -> jnp.ndarray:
    """Linear Extrapolation：假设未来 T 步的 latent 保持匀速 = la_current。
    走同一套 build_la_cumsum 以保证与训练结构对齐。

    Args:
      la_current: shape (B, D)，"上一帧 → 当前帧"的 LAM 单步 latent。
      T: 要铺开的时间步数。
    """
    repeated = einops.repeat(la_current, "b d -> b t d", t=T)
    return build_la_cumsum(repeated)


def build_la_historical_ema(la_current: jnp.ndarray, la_history_mean: jnp.ndarray, T: int) -> jnp.ndarray:
    """Historical EMA：用历史 latent 的 EMA 均值作为"单步幅度"的估计。

    Args:
      la_current: shape (B, D)，当前"上一帧→当前帧"的 LAM 单步 latent。
      la_history_mean: shape (B, D)，历史若干步 latent 的 EMA 平均。
      T: 要铺开的时间步数。
    """
    del la_current  # 目前仅使用 EMA 均值作为幅度基准，保留参数便于后续扩展
    repeated = einops.repeat(la_history_mean, "b d -> b t d", t=T)
    return build_la_cumsum(repeated)


@dataclasses.dataclass(frozen=True)
class Pi0Config(_model.BaseModelConfig):
    dtype: str = "bfloat16"
    paligemma_variant: _gemma.Variant = "gemma_2b"
    action_expert_variant: _gemma.Variant = "gemma_300m"

    # 新增：指定用哪个数据集的 SALI 统计量
    sali_dataset: str = "metaworld"

    # Set the model specific defaults.
    action_dim: int = 32  # default=32
    action_horizon: int = 6  # default=50
    max_token_len: int = 48  # default=48

    latent_action_pred: bool = False
    vq_remap: bool | None = None
    unknown_index: int | None = None
    phy_lam_ckpt: str | None = None

    # ── CII 推理策略开关（默认值，可在 sample_actions 调用处 override） ──
    latent_init_scheme: str = "virtual_cumsum"   # gauss / virtual_cumsum / historical_ema / linear_extrap
    latent_init_tau: float = 1.0                 # τ∗，默认从纯噪声端开始

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
    def __init__(self, config: Pi0Config, rngs: nnx.Rngs):
        super().__init__(config.action_dim, config.action_horizon, config.max_token_len)
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
        self.config = config
        # ── LAM（直接挂载 PyTorch 模型）──────────────
        if config.latent_action_pred:
            phy_lam = get_or_create_lam(pretrained_path=config.phy_lam_ckpt, device="cuda")
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

        # 默认推理策略（可在 sample_actions 调用处临时 override）
        self._default_latent_init_scheme = config.latent_init_scheme
        self._default_latent_init_tau = config.latent_init_tau

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
        self.img_seq_horizon = (self.action_horizon - 1) * self.image_chunk

    @at.typecheck
    def embed_prefix(
            self, obs: _model.Observation, lam_out: dict | None = None
    ) -> tuple[at.Float[at.Array, "b s emb"], at.Bool[at.Array, "b s"], at.Bool[at.Array, " s"], at.Float[
        at.Array, "..."]]:
        input_mask = []
        ar_mask = []
        current_tokens = []
        predict_tokens = []

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
            ar_mask.append(True)
            ar_mask.append(False)
            ar_mask.append(True)

        print("final.tokens.shape", tokens.shape)
        print("final.input_mask.shape", input_mask.shape)
        ar_mask = jnp.array(ar_mask)
        print("final.ar_mask.shape", ar_mask.shape)

        return tokens, input_mask, ar_mask

    @override
    def compute_loss(
            self, rng: at.KeyArrayLike, observation: _model.Observation, actions: _model.Actions, *,
            train: bool = False, lam_out: dict | None = None
    ) -> at.Float[at.Array, "*b ah"]:
        preprocess_rng, noise_rng, time_rng, rgb_noise_rng, wrist_noise_rng = jax.random.split(rng, 5)
        observation = _model.preprocess_observation(preprocess_rng, observation, train=train)
        print("===================================================compute_loss==========================================")

        # 多拆两个 rng 给 latent 分支的噪声
        preprocess_rng, noise_rng, time_rng, rgb_noise_rng, wrist_noise_rng = jax.random.split(rng, 5)
        observation = _model.preprocess_observation(preprocess_rng, observation, train=train)

        actions = actions[:, :-1]
        batch_shape = actions.shape[:-2]

        noise = jax.random.normal(noise_rng, actions.shape)

        time = jax.random.beta(time_rng, 1.5, 1, batch_shape) * 0.999 + 0.001
        time_expanded_action = time[..., None, None]

        # Action 分支 flow matching
        x_t = time_expanded_action * noise + (1 - time_expanded_action) * actions
        u_t = noise - actions

        prefix_tokens, prefix_mask, prefix_ar_mask, predict_tokens = self.embed_prefix(observation, lam_out=lam_out)

        rgb_pred = predict_tokens[:, : self.img_seq_horizon]
        wrist_pred = predict_tokens[:, self.img_seq_horizon: 2 * self.img_seq_horizon]

        # 累加成绝对 latent 序列，作为 latent 分支的"真值 data"
        rgb_cumsum = build_la_cumsum(rgb_pred)
        wrist_cumsum = build_la_cumsum(wrist_pred)

        # Latent 分支：各自独立的噪声
        rgb_lnoise = jax.random.normal(rgb_noise_rng, rgb_cumsum.shape)
        wrist_lnoise = jax.random.normal(wrist_noise_rng, wrist_cumsum.shape)

        # Latent 分支 flow matching（和 action 分支保持同一个 time）
        xr_t = time_expanded_action * rgb_lnoise + (1 - time_expanded_action) * rgb_cumsum
        xw_t = time_expanded_action * wrist_lnoise + (1 - time_expanded_action) * wrist_cumsum

        # 目标速度：noise - data
        ur_t = rgb_lnoise - rgb_cumsum
        uw_t = wrist_lnoise - wrist_cumsum

        suffix_tokens, suffix_mask, suffix_ar_mask = self.embed_suffix(observation, x_t, time, xr_t, xw_t)
        print("suffix_tokens.shape", suffix_tokens.shape)
        input_mask = jnp.concatenate([prefix_mask, suffix_mask], axis=1)
        ar_mask = jnp.concatenate([prefix_ar_mask, suffix_ar_mask], axis=0)
        attn_mask = make_attn_mask(input_mask, ar_mask)
        positions = jnp.cumsum(input_mask, axis=1) - 1

        (prefix_out, suffix_out), _ = self.PaliGemma.llm(
            [prefix_tokens, suffix_tokens], mask=attn_mask, positions=positions
        )

        print("suffix_out_0.shape:", suffix_out.shape)

        rgb_indices = jnp.arange(1, 1 + 3 * self.action_seq_horizon, 3)
        wrist_indices = jnp.arange(2, 2 + 3 * self.action_seq_horizon, 3)
        action_indices = jnp.arange(3, 3 + 3 * self.action_seq_horizon, 3)

        vr_t = self.rgb_out_proj(suffix_out[:, rgb_indices])
        print("vr_t.shape", vr_t.shape)

        vw_t = self.wrist_out_proj(suffix_out[:, wrist_indices])
        print("vw_t.shape", vw_t.shape)

        v_t = self.action_out_proj(suffix_out[:, action_indices])
        print("v_t.shape", v_t.shape)

        return (
            jnp.mean(jnp.abs(v_t - u_t), axis=-1),
            jnp.mean(jnp.abs(vr_t - ur_t), axis=-1),
            jnp.mean(jnp.abs(vw_t - uw_t), axis=-1),
        )

    # ==================================================================================
    # 构造 Latent 分支推理起点（支持 4 种策略 + τ∗）
    # ==================================================================================
    def _build_latent_init(
            self,
            rng: at.KeyArrayLike,
            batch_size: int,
            img_seq_horizon: int,
            scheme: str,
            history_latent: dict | None = None,
    ) -> tuple[jnp.ndarray, jnp.ndarray]:
        """根据 scheme 构造 rgb / wrist 两路 latent 起点。

        Args:
          rng: 用于 "gauss" 和 "virtual_cumsum" 策略的随机源。
          batch_size: batch 大小。
          img_seq_horizon: latent 序列长度（等于 action_seq_horizon * image_chunk）。
          scheme: 起点策略名。
          history_latent: 可选。形如
              {
                "base_0_rgb":            (B, D),   # 必填（extrap/ema 用）
                "left_wrist_0_rgb":      (B, D),   # 必填（extrap/ema 用）
                "base_0_rgb_ema":        (B, D),   # historical_ema 需要
                "left_wrist_0_rgb_ema":  (B, D),   # historical_ema 需要
              }

        Returns:
          xr_init, xw_init: 形状 (B, img_seq_horizon, compress_dim)，dtype=bfloat16。
        """
        D = self.compress_dim
        rgb_rng, wrist_rng = jax.random.split(rng, 2)

        if scheme == "gauss":
            xr_init = jax.random.normal(rgb_rng, (batch_size, img_seq_horizon, D)).astype(jnp.bfloat16)
            xw_init = jax.random.normal(wrist_rng, (batch_size, img_seq_horizon, D)).astype(jnp.bfloat16)

        elif scheme == "virtual_cumsum":
            rgb_virtual = jax.random.normal(rgb_rng, (batch_size, img_seq_horizon, D)).astype(jnp.bfloat16)
            wrist_virtual = jax.random.normal(wrist_rng, (batch_size, img_seq_horizon, D)).astype(jnp.bfloat16)
            xr_init = build_la_cumsum(rgb_virtual)
            xw_init = build_la_cumsum(wrist_virtual)

            pos_std = jnp.sqrt(jnp.arange(1, img_seq_horizon + 1, dtype=jnp.float32))
            pos_std_b = pos_std[None, :, None].astype(jnp.bfloat16)  # (1, T, 1)
            xr_init = xr_init / pos_std_b
            xw_init = xw_init / pos_std_b
        
        # ========== SALI 精确对齐版 ==========
        elif scheme == "virtual_cumsum_sali":
            rgb_virtual = jax.random.normal(rgb_rng, (batch_size, img_seq_horizon, D)).astype(jnp.bfloat16)
            wrist_virtual = jax.random.normal(wrist_rng, (batch_size, img_seq_horizon, D)).astype(jnp.bfloat16)
            xr_raw = build_la_cumsum(rgb_virtual)
            xw_raw = build_la_cumsum(wrist_virtual)

            # --- 改动：按 config 取对应数据集的实测 std ---
            dataset_name = self.config.sali_dataset
            assert dataset_name in SALI_MEASURED_STATS, (
                f"Unknown sali_dataset '{dataset_name}'. "
                f"Available: {list(SALI_MEASURED_STATS.keys())}"
            )
            stats = SALI_MEASURED_STATS[dataset_name]
            assert stats["rgb"] is not None and stats["wrist"] is not None, (
                f"SALI stats for '{dataset_name}' not measured yet."
            )
            
            # --- 关键：np -> jnp 转换在函数内部完成 ---
            rgb_measured_std   = jnp.asarray(stats["rgb"],   dtype=jnp.float32)
            wrist_measured_std = jnp.asarray(stats["wrist"], dtype=jnp.float32)

            # --- 安全检查：防止常数长度 < img_seq_horizon 时静默截断 ---
            assert rgb_measured_std.shape[0] >= img_seq_horizon, (
                f"SALI stats for '{dataset_name}' has length {rgb_measured_std.shape[0]}, "
                f"but img_seq_horizon={img_seq_horizon}. Please re-measure."
            )

            # virtual_cumsum 天然 std = sqrt(t+1)，目标 std = measured
            pos_std = jnp.sqrt(jnp.arange(1, img_seq_horizon + 1, dtype=jnp.float32))
            rgb_target   = rgb_measured_std[:img_seq_horizon]
            wrist_target = wrist_measured_std[:img_seq_horizon]

            rgb_scale   = (rgb_target   / pos_std)[None, :, None]
            wrist_scale = (wrist_target / pos_std)[None, :, None]

            xr_init = (xr_raw.astype(jnp.float32) * rgb_scale).astype(jnp.bfloat16)
            xw_init = (xw_raw.astype(jnp.float32) * wrist_scale).astype(jnp.bfloat16)
        # ======================================

        elif scheme == "linear_extrap":
            if history_latent is None:
                raise ValueError(
                    "latent_init_scheme='linear_extrap' requires history_latent dict "
                    "with 'base_0_rgb' and 'left_wrist_0_rgb' keys."
                )
            rgb_current = history_latent["base_0_rgb"].astype(jnp.bfloat16)
            wrist_current = history_latent["left_wrist_0_rgb"].astype(jnp.bfloat16)
            xr_init = build_la_linear_extrap(rgb_current, img_seq_horizon)
            xw_init = build_la_linear_extrap(wrist_current, img_seq_horizon)

        elif scheme == "historical_ema":
            if history_latent is None or "base_0_rgb_ema" not in history_latent:
                raise ValueError(
                    "latent_init_scheme='historical_ema' requires history_latent dict "
                    "with '*_ema' keys (EMA mean of past latents)."
                )
            rgb_current = history_latent["base_0_rgb"].astype(jnp.bfloat16)
            wrist_current = history_latent["left_wrist_0_rgb"].astype(jnp.bfloat16)
            rgb_ema = history_latent["base_0_rgb_ema"].astype(jnp.bfloat16)
            wrist_ema = history_latent["left_wrist_0_rgb_ema"].astype(jnp.bfloat16)
            xr_init = build_la_historical_ema(rgb_current, rgb_ema, img_seq_horizon)
            xw_init = build_la_historical_ema(wrist_current, wrist_ema, img_seq_horizon)

        else:
            raise ValueError(f"Unknown latent_init_scheme: {scheme}")

        return xr_init, xw_init

    # ==================================================================================
    # CII Inference：支持 4 种 latent 起点策略 + τ∗ 起始时间
    # ------------------------------------------------------------------------------------
    # Action 分支永远从 τ=tau_start 的标准高斯开始。
    # Latent 分支起点由 latent_init_scheme 决定。
    # scheme / tau 可通过调用参数 override，不需要重建 config。
    # ==================================================================================
    @override
    def sample_actions(
            self,
            rng: at.KeyArrayLike,
            observation: _model.Observation,
            *,
            num_steps: int = 10,
            history_latent: dict | None = None,
            latent_init_scheme: str | None = None,
            latent_init_tau: float | None = None,
    ) -> _model.Actions:
        """
        Args:
          rng: 随机种子。
          observation: 当前观测。
          num_steps: 保留参数（未实际使用，步数由 action_seq_horizon 决定）。
          history_latent: Historical EMA / Linear Extrap 策略需要。见 `_build_latent_init` 的 docstring。
          latent_init_scheme: 若提供，覆盖 config 中的 scheme。
          latent_init_tau: 若提供，覆盖 config 中的 τ∗。
        """
        # ── 确定本次推理使用的 scheme 和 tau（优先级：调用参数 > config 默认值）──
        scheme = latent_init_scheme if latent_init_scheme is not None else self._default_latent_init_scheme
        tau_start = float(
            latent_init_tau if latent_init_tau is not None else self._default_latent_init_tau
        )
        assert 0.0 < tau_start <= 1.0, f"latent_init_tau must be in (0, 1], got {tau_start}"

        preprocess_rng, action_noise_rng, latent_init_rng = jax.random.split(rng, 3)

        img_seq_horizon = self.action_seq_horizon * self.image_chunk
        action_seq_horizon = self.action_seq_horizon
        observation = _model.preprocess_observation(None, observation, train=False)

        # dt 为负，积分长度 = tau_start，保证积分终点为 τ=0
        dt = -tau_start / action_seq_horizon
        batch_size = observation.state.shape[0]

        # ── Action 分支起点：标准高斯 ──────────────────────────────────────
        # 注：即使 tau_start<1，action 分支也没有"真实先验"可用，仍从纯噪声开始
        noise = jax.random.normal(action_noise_rng, (batch_size, action_seq_horizon, self.action_dim))

        # ── Latent 分支起点：按 scheme 构造 ────────────────────────────────
        xr_init, xw_init = self._build_latent_init(
            latent_init_rng, batch_size, img_seq_horizon, scheme=scheme, history_latent=history_latent
        )

        # ── Prefix 前向，填充 KV cache（推理时 lam_out=None）─────────────────
        prefix_tokens, prefix_mask, prefix_ar_mask, _ = self.embed_prefix(observation, lam_out=None)
        prefix_attn_mask = make_attn_mask(prefix_mask, prefix_ar_mask)
        positions = jnp.cumsum(prefix_mask, axis=1) - 1
        _, kv_cache = self.PaliGemma.llm([prefix_tokens, None], mask=prefix_attn_mask, positions=positions)

        def step(carry):
            x_t, xr_t, xw_t, time = carry
            time_for_inference = jnp.full((x_t.shape[0],), time)

            suffix_tokens, suffix_mask, suffix_ar_mask = self.embed_suffix(
                observation, x_t, time_for_inference, xr_t, xw_t
            )

            suffix_attn_mask = make_attn_mask(suffix_mask, suffix_ar_mask)
            prefix_attn_mask_broadcast = einops.repeat(prefix_mask, "b p -> b s p", s=suffix_tokens.shape[1])
            full_attn_mask = jnp.concatenate([prefix_attn_mask_broadcast, suffix_attn_mask], axis=-1)
            assert full_attn_mask.shape == (
                batch_size,
                suffix_tokens.shape[1],
                prefix_tokens.shape[1] + suffix_tokens.shape[1],
            )
            positions = jnp.sum(prefix_mask, axis=-1)[:, None] + jnp.cumsum(suffix_mask, axis=-1) - 1

            (prefix_out, suffix_out), _ = self.PaliGemma.llm(
                [None, suffix_tokens],
                mask=full_attn_mask,
                positions=positions,
                kv_cache=kv_cache,
            )
            assert prefix_out is None

            rgb_indices = jnp.arange(1, 1 + 3 * self.action_seq_horizon, 3)
            wrist_indices = jnp.arange(2, 2 + 3 * self.action_seq_horizon, 3)
            action_indices = jnp.arange(3, 3 + 3 * self.action_seq_horizon, 3)

            vr_t = self.rgb_out_proj(suffix_out[:, rgb_indices])
            vw_t = self.wrist_out_proj(suffix_out[:, wrist_indices])
            v_t = self.action_out_proj(suffix_out[:, action_indices])

            return (
                x_t + dt * v_t,
                xr_t + dt * vr_t,
                xw_t + dt * vw_t,
                time + dt,
            )

        def cond(carry):
            _, _, _, time = carry
            return time >= -dt / 2

        x_0, _, _, _ = jax.lax.while_loop(cond, step, (noise, xr_init, xw_init, tau_start))
        return x_0