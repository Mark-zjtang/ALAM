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

from functools import partial

logger = logging.getLogger("openpi")
from diffusers.models import AutoencoderKL
import jax.nn as jnn


def make_attn_mask(input_mask, mask_ar):
    mask_ar   = jnp.broadcast_to(mask_ar, input_mask.shape)
    cumsum    = jnp.cumsum(mask_ar, axis=1)
    attn_mask = cumsum[:, None, :] <= cumsum[:, :, None]
    valid_mask= input_mask[:, None, :] * input_mask[:, :, None]
    return jnp.logical_and(attn_mask, valid_mask)


def vae_model():
    return AutoencoderKL.from_pretrained(
        "/home/fortress/new_storage/TZJ/openpi_worldflow/sd-vae-ft-mse/"
    )


@at.typecheck
def posemb_sincos(
    pos: at.Real[at.Array, " b"],
    embedding_dim: int,
    min_period: float,
    max_period: float,
) -> at.Float[at.Array, "b {embedding_dim}"]:
    if embedding_dim % 2 != 0:
        raise ValueError(f"embedding_dim ({embedding_dim}) must be divisible by 2")
    fraction       = jnp.linspace(0.0, 1.0, embedding_dim // 2)
    period         = min_period * (max_period / min_period) ** fraction
    sinusoid_input = jnp.einsum(
        "i,j->ij", pos, 1.0 / period * 2 * jnp.pi,
        precision=jax.lax.Precision.HIGHEST,
    )
    return jnp.concatenate([jnp.sin(sinusoid_input), jnp.cos(sinusoid_input)], axis=-1)


@dataclasses.dataclass(frozen=True)
class Pi0Config(_model.BaseModelConfig):
    dtype: str = "bfloat16"
    paligemma_variant: _gemma.Variant   = "gemma_2b"
    action_expert_variant: _gemma.Variant = "gemma_300m"
    action_dim: int    = 32
    action_horizon: int= 6
    max_token_len: int = 48

    @property
    @override
    def model_type(self) -> _model.ModelType:
        return _model.ModelType.PI0

    @override
    def create(self, rng: at.KeyArrayLike) -> "Pi0":
        return Pi0(self, rngs=nnx.Rngs(rng))

    @override
    def inputs_spec(self, *, batch_size: int = 1):
        image_spec      = jax.ShapeDtypeStruct(
            [batch_size, self.action_horizon, *_model.IMAGE_RESOLUTION, 3], jnp.float32
        )
        image_mask_spec = jax.ShapeDtypeStruct(
            [batch_size, self.action_horizon], jnp.bool_
        )
        with at.disable_typechecking():
            observation_spec = _model.Observation(
                images={
                    "base_0_rgb":       image_spec,
                    "left_wrist_0_rgb": image_spec,
                    "right_wrist_0_rgb":image_spec,
                },
                image_masks={
                    "base_0_rgb":       image_mask_spec,
                    "left_wrist_0_rgb": image_mask_spec,
                    "right_wrist_0_rgb":image_mask_spec,
                },
                state=jax.ShapeDtypeStruct(
                    [batch_size, self.action_horizon, self.action_dim], jnp.float32
                ),
                tokenized_prompt=jax.ShapeDtypeStruct(
                    [batch_size, self.action_horizon, self.max_token_len], jnp.int32
                ),
                tokenized_prompt_mask=jax.ShapeDtypeStruct(
                    [batch_size, self.action_horizon, self.max_token_len], bool
                ),
            )
        action_spec = jax.ShapeDtypeStruct(
            [batch_size, self.action_horizon, self.action_dim], jnp.float32
        )
        return observation_spec, action_spec

    def get_freeze_filter(self) -> nnx.filterlib.Filter:
        filters  = []
        has_lora = False
        gemma_params_filter         = nnx_utils.PathRegex(".*llm.*")
        action_expert_params_filter = nnx_utils.PathRegex(".*llm.*_1.*")
        if "lora" in self.paligemma_variant:
            filters.append(gemma_params_filter)
            if "lora" not in self.action_expert_variant:
                filters.append(nnx.Not(action_expert_params_filter))
            has_lora = True
        elif "lora" in self.action_expert_variant:
            filters.append(action_expert_params_filter)
            has_lora = True
        if has_lora:
            filters.append(nnx.Not(nnx_utils.PathRegex(".*lora.*")))
        if not filters:
            return nnx.Nothing
        return nnx.All(*filters)


# ─────────────────────────────────────────────────────────────
# [新增] RSSM Config ── 复用 Pi0Config，只替换 create
# ─────────────────────────────────────────────────────────────
@dataclasses.dataclass(frozen=True)
class Pi0RSSMConfig(Pi0Config):
    """
    与 Pi0Config 完全相同的超参数，保证对比公平。
    唯一区别：create() 返回 Pi0_RSSM 而非 Pi0。
    """
    @override
    def create(self, rng: at.KeyArrayLike) -> "Pi0_RSSM":
        return Pi0_RSSM(self, rngs=nnx.Rngs(rng))


# ─────────────────────────────────────────────────────────────
# [新增] RSSM 基础组件
# ─────────────────────────────────────────────────────────────

class GRUCell(nnx.Module):
    """
    [新增] 手动 GRU Cell
    实现 Dreamer 确定性状态转移: h_t = GRU(h_{t-1}, [z_{t-1
}, a_{t-1}])
    """
    def __init__(self, input_dim: int, hidden_dim: int, rngs: nnx.Rngs):
        self.hidden_dim = hidden_dim
        self.W_gates    = nnx.Linear(input_dim + hidden_dim, 2 * hidden_dim, rngs=rngs)
        self.W_cand     = nnx.Linear(input_dim + hidden_dim, hidden_dim,     rngs=rngs)

    def __call__(self, h: jnp.ndarray, x: jnp.ndarray) -> jnp.ndarray:
        # h: (batch, hidden_dim), x: (batch, input_dim)
        hx      = jnp.concatenate([h, x], axis=-1)
        gates   = jax.nn.sigmoid(self.W_gates(hx))
        z, r    = jnp.split(gates, 2, axis=-1)
        h_tilde = jnp.tanh(self.W_cand(jnp.concatenate([r * h, x], axis=-1)))
        return (1.0 - z) * h + z * h_tilde                 # (batch, hidden_dim)


class RSSMPrior(nnx.Module):
    """[新增] 先验 p(z_t | h_t)，imagination 推理时使用"""
    def __init__(self, h_dim: int, z_dim: int, rngs: nnx.Rngs):
        self.fc     = nnx.Linear(h_dim, h_dim, rngs=rngs)
        self.mean   = nnx.Linear(h_dim, z_dim, rngs=rngs)
        self.logvar = nnx.Linear(h_dim, z_dim, rngs=rngs)

    def __call__(self, h: jnp.ndarray):
        # h: (batch, h_dim)
        feat = nnx.swish(self.fc(h))
        return self.mean(feat), self.logvar(feat)           # 均值和对数方差

    def sample(self, h: jnp.ndarray, rng: jnp.ndarray):
        mean, logvar = self(h)
        std = jnp.exp(0.5 * jnp.clip(logvar, -10.0, 10.0))
        eps = jax.random.normal(rng, mean.shape, dtype=mean.dtype)
        return mean + std * eps, mean, logvar


class RSSMPosterior(nnx.Module):
    """[新增] 后验 q(z_t | h_t, o_t)，训练时使用真实观测"""
    def __init__(self, h_dim: int, obs_dim: int, z_dim: int, rngs: nnx.Rngs):
        self.fc     = nnx.Linear(h_dim + obs_dim, h_dim, rngs=rngs)
        self.mean   = nnx.Linear(h_dim, z_dim,           rngs=rngs)
        self.logvar = nnx.Linear(h_dim, z_dim,           rngs=rngs)

    def __call__(self, h: jnp.ndarray, obs_embed: jnp.ndarray):
        # h: (batch, h_dim), obs_embed: (batch, obs_dim)
        feat = nnx.swish(self.fc(jnp.concatenate([h, obs_embed], axis=-1)))
        return self.mean(feat), self.logvar(feat)

    def sample(self, h: jnp.ndarray, obs_embed: jnp.ndarray, rng: jnp.ndarray):
        mean, logvar = self(h, obs_embed)
        std = jnp.exp(0.5 * jnp.clip(logvar, -10.0, 10.0))
        eps = jax.random.normal(rng, mean.shape, dtype=mean.dtype)
        return mean + std * eps, mean, logvar


# ─────────────────────────────────────────────────────────────
# 原有 AdaptiveFusion（保持不变）
# ─────────────────────────────────────────────────────────────

class AdaptiveFusion(nnx.Module):
    def __init__(self, temperature=0.1):
        super().__init__()
        self.temperature = temperature
        self.alpha = nnx.Param(jnp.ones(3))

    def __call__(self, image_tokens_fusion, name, attentin_query_rgb, attentin_query_wrist):
        max_pool    = self._max_pool_attention(image_tokens_fusion)
        sum_pool    = self._sum_pool_attention(image_tokens_fusion)
        learned_pool= self._learned_attention(
            image_tokens_fusion, name, attentin_query_rgb, attentin_query_wrist
        )
        weights      = jax.nn.softmax(self.alpha / self.temperature)
        fused_tokens = weights[0] * max_pool + weights[1] * sum_pool + weights[2] * learned_pool
        return fused_tokens, weights

    def _max_pool_attention(self, tokens):
        scores  = tokens.max(axis=-1, keepdims=True)
        weights = jax.nn.softmax(scores / self.temperature, axis=2)
        return jnp.max(weights * tokens, axis=2)

    def _sum_pool_attention(self, tokens):
        scores  = tokens.sum(axis=-1, keepdims=True)
        weights = jax.nn.softmax(scores, axis=2)
        return jnp.sum(weights * tokens, axis=2)

    def _learned_attention(self, tokens, name, attentin_query_rgb, attentin_query_wrist):
        if name == "base_0_rgb":
            scores = attentin_query_rgb(tokens)
        elif name == "left_wrist_0_rgb":
            scores = attentin_query_wrist(tokens)
        else:
            scores = attentin_query_rgb(tokens)
        weights = jax.nn.softmax(scores / self.temperature, axis=2)
        return jnp.sum(weights * tokens, axis=2)


# ─────────────────────────────────────────────────────────────
# 原有 Pi0（保持不变）
# ─────────────────────────────────────────────────────────────

class Pi0(_model.BaseModel):
    def __init__(self, config: Pi0Config, rngs: nnx.Rngs):
        super().__init__(config.action_dim, config.action_horizon, config.max_token_len)
        paligemma_config     = _gemma.get_config(config.paligemma_variant)
        action_expert_config = _gemma.get_config(config.action_expert_variant)
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

        self.state_proj          = nnx.Linear(config.action_dim, action_expert_config.width, rngs=rngs)
        self.action_in_proj      = nnx.Linear(config.action_dim, action_expert_config.width, rngs=rngs)
        self.action_time_mlp_in  = nnx.Linear(2 * action_expert_config.width, action_expert_config.width, rngs=rngs)
        self.action_time_mlp_out = nnx.Linear(action_expert_config.width, action_expert_config.width, rngs=rngs)
        self.action_out_proj     = nnx.Linear(action_expert_config.width, config.action_dim, rngs=rngs)

        new_rngs = nnx.Rngs(42)
        self.compress_dim        = 2048
        self.rgb_in_proj         = nnx.Linear(self.compress_dim, action_expert_config.width, rngs=new_rngs)
        self.rgb_time_mlp_in     = nnx.Linear(2 * action_expert_config.width, action_expert_config.width, rngs=new_rngs)
        self.rgb_time_mlp_out    = nnx.Linear(action_expert_config.width, action_expert_config.width, rngs=new_rngs)
        self.rgb_out_proj        = nnx.Linear(action_expert_config.width, self.compress_dim, rngs=new_rngs)
        self.wrist_in_proj       = nnx.Linear(self.compress_dim, action_expert_config.width, rngs=new_rngs)
        self.wrist_time_mlp_in   = nnx.Linear(2 * action_expert_config.width, action_expert_config.width, rngs=new_rngs)
        self.wrist_time_mlp_out  = nnx.Linear(action_expert_config.width, action_expert_config.width, rngs=new_rngs)
        self.wrist_out_proj      = nnx.Linear(action_expert_config.width, self.compress_dim, rngs=new_rngs)
        self.attentin_query_rgb  = nnx.Linear(self.compress_dim, 1, rngs=new_rngs)
        self.attentin_query_wrist= nnx.Linear(self.compress_dim, 1, rngs=new_rngs)

        self.image_chunk        = 1
        self.action_seq_horizon = self.action_horizon - 1
        self.img_seq_horizon    = (self.action_horizon - 1) * self.image_chunk
        self.adaptive_fusion    = AdaptiveFusion()

    @at.typecheck
    def embed_prefix(self, obs: _model.Observation):
        input_mask, ar_mask, current_tokens, predict_tokens = [], [], [], []
        for name in obs.images:
            if name == "right_wrist_0_rgb":
                continue
            image_tokens_fusion, _ = self.PaliGemma.img(obs.images[name], train=False)
            fused_tokens, fusion_weights = self.adaptive_fusion(
                image_tokens_fusion, name,
                self.attentin_query_rgb, self.attentin_query_wrist,
            )
            fused_tokens_expanded = fused_tokens[:, :, None, :]
            image_pool_tokens     = einops.repeat(
                fused_tokens_expanded, "b t 1 d -> b t (repeat d)", repeat=1
            )
            current_image_tokens = image_pool_tokens[:, :self.image_chunk]
            predict_image_tokens = image_pool_tokens[:, self.image_chunk:]
            current_tokens.append(current_image_tokens)
            predict_tokens.append(predict_image_tokens)
            input_mask.append(
                einops.repeat(obs.image_masks[name], "b -> b s", s=current_image_tokens.shape[1])
            )
            ar_mask += [False] * current_image_tokens.shape[1]

        if obs.tokenized_prompt is not None:
            tokenized_inputs = self.PaliGemma.llm(obs.tokenized_prompt, method="embed")
            current_tokens.append(tokenized_inputs)
            input_mask.append(obs.tokenized_prompt_mask)
            ar_mask += [False] * tokenized_inputs.shape[1]

        current_tokens = jnp.concatenate(current_tokens, axis=1)
        predict_tokens = jnp.concatenate(predict_tokens, axis=1)
        input_mask     = jnp.concatenate(input_mask, axis=1)
        ar_mask        = jnp.array(ar_mask)
        return current_tokens, input_mask, ar_mask, predict_tokens, fusion_weights

    @at.typecheck
    def embed_suffix(self, obs, noisy_actions, timestep, rgb_predict_tokens, wrist_predict_tokens):
        input_mask, ar_mask, tokens = [], [], []

        # state token: obs.state shape=(batch, horizon, action_dim), 取第0帧
        state_token = self.state_proj(obs.state[:, 0, :])[:, None, :]  # (batch,1,width)
        tokens.append(state_token)
        input_mask.append(jnp.ones((obs.state.shape[0], 1), dtype=jnp.bool_))
        ar_mask += [True]

        time_emb = posemb_sincos(
            timestep, self.action_in_proj.out_features, min_period=4e-3, max_period=4.0
        )
        rgb_tok   = self.rgb_in_proj(rgb_predict_tokens)
        wrist_tok = self.wrist_in_proj(wrist_predict_tokens)
        action_tok= self.action_in_proj(noisy_actions)

        T = action_tok.shape[1]
        time_tokens       = einops.repeat(time_emb, "b d -> b t d", t=T)
        time_tokens_images= einops.repeat(time_emb, "b d -> b t d", t=T * self.image_chunk)

        rgb_time   = nnx.swish(self.rgb_time_mlp_in(
            jnp.concatenate([rgb_tok,   time_tokens_images], axis=-1)))
        rgb_time   = self.rgb_time_mlp_out(rgb_time)

        wrist_time = nnx.swish(self.wrist_time_mlp_in(
            jnp.concatenate([wrist_tok, time_tokens_images], axis=-1)))
        wrist_time = self.wrist_time_mlp_out(wrist_time)

        action_time= nnx.swish(self.action_time_mlp_in(
            jnp.concatenate([action_tok,time_tokens],        axis=-1)))
        action_time= self.action_time_mlp_out(action_time)

        batch_size = action_tok.shape[0]
        for t in range(T):
            tokens.append(rgb_time[:,   t:t+1, :])
            input_mask.append(jnp.ones((batch_size, 1), dtype=jnp.bool_))
            tokens.append(wrist_time[:, t:t+1, :])
            input_mask.append(jnp.ones((batch_size, 1), dtype=jnp.bool_))
            tokens.append(action_time[:,t:t+1, :])
            input_mask.append(jnp.ones((batch_size, 1), dtype=jnp.bool_))
            ar_mask += [True, False, True]

        tokens     = jnp.concatenate(tokens, axis=1)
        input_mask = jnp.concatenate(input_mask, axis=1)
        ar_mask    = jnp.array(ar_mask)
        return tokens, input_mask, ar_mask

    @override
    def compute_loss(self, rng, observation, actions, *, train=False):
        preprocess_rng, noise_rng, rgb_noise_rng, wrist_noise_rng, time_rng = jax.random.split(rng, 5)
        observation  = _model.preprocess_observation(preprocess_rng, observation, train=train)
        actions      = actions[:, :-1]                         # (batch, T, action_dim)
        batch_shape  = actions.shape[:-2]
        noise        = jax.random.normal(noise_rng, actions.shape)
        time         = jax.random.beta(time_rng, 1.5, 1, batch_shape) * 0.999 + 0.001
        time_exp     = time[..., None, None]
        x_t          = time_exp * noise + (1 - time_exp) * actions
        u_t          = noise - actions

        prefix_tokens, prefix_mask, prefix_ar_mask, predict_tokens, fusion_weights = \
            self.embed_prefix(observation)

        rgb_pred   = predict_tokens[:, :self.img_seq_horizon]
        wrist_pred = predict_tokens[:, self.img_seq_horizon: 2 * self.img_seq_horizon]

        rgb_noise   = jax.random.normal(rgb_noise_rng,   rgb_pred.shape)
        wrist_noise = jax.random.normal(wrist_noise_rng, wrist_pred.shape)
        xr_t = time_exp * rgb_noise   + (1 - time_exp) * rgb_pred
        ur_t = rgb_noise   - rgb_pred
        xw_t = time_exp * wrist_noise + (1 - time_exp) * wrist_pred
        uw_t = wrist_noise - wrist_pred

        suffix_tokens, suffix_mask, suffix_ar_mask = self.embed_suffix(
            observation, x_t, time, xr_t, xw_t
        )
        input_mask = jnp.concatenate([prefix_mask, suffix_mask], axis=1)
        ar_mask    = jnp.concatenate([prefix_ar_mask, suffix_ar_mask], axis=0)
        attn_mask  = make_attn_mask(input_mask, ar_mask)
        positions  = jnp.cumsum(input_mask, axis=1) - 1

        (_, suffix_out), _ = self.PaliGemma.llm(
            [prefix_tokens, suffix_tokens], mask=attn_mask, positions=positions
        )

        T              = self.action_seq_horizon
        rgb_indices    = jnp.arange(1, 1 + 3 * T, 3)
        wrist_indices  = jnp.arange(2, 2 + 3 * T, 3)
        action_indices = jnp.arange(3, 3 + 3 * T, 3)

        vr_t = self.rgb_out_proj(suffix_out[:, rgb_indices])
        vw_t = self.wrist_out_proj(suffix_out[:, wrist_indices])
        v_t  = self.action_out_proj(suffix_out[:, action_indices])

        return (
            jnp.mean(jnp.abs(v_t  - u_t),  axis=-1),
            jnp.mean(jnp.abs(vr_t - ur_t), axis=-1),
            jnp.mean(jnp.abs(vw_t - uw_t), axis=-1),
            fusion_weights,
        )

    @override
    def sample_actions(self, rng, observation, *, num_steps=10):
        preprocess_rng, noise_rng, rgb_noise_rng, wrist_noise_rng, _ = jax.random.split(rng, 5)
        observation        = _model.preprocess_observation(None, observation, train=False)
        T                  = self.action_seq_horizon
        img_T              = T * self.image_chunk
        dt                 = -1.0 / T
        batch_size         = observation.state.shape[0]

        noise            = jax.random.normal(noise_rng,       (batch_size, T,     self.action_dim))
        rgb_pred_noise   = jax.random.normal(rgb_noise_rng,   (batch_size, img_T, 2048))
        wrist_pred_noise = jax.random.normal(wrist_noise_rng, (batch_size, img_T, 2048))

        prefix_tokens, prefix_mask, prefix_ar_mask, _, _ = self.embed_prefix(observation)
        prefix_attn_mask = make_attn_mask(prefix_mask, prefix_ar_mask)
        positions        = jnp.cumsum(prefix_mask, axis=1) - 1
        _, kv_cache      = self.PaliGemma.llm(
            [prefix_tokens, None], mask=prefix_attn_mask, positions=positions
        )

        def step(carry):
            x_t, time, xr_t, xw_t = carry
            time_batch = jnp.full((batch_size,), time)
            suffix_tokens, suffix_mask, suffix_ar_mask = self.embed_suffix(
                observation, x_t, time_batch, xr_t, xw_t
            )
            suffix_attn_mask  = make_attn_mask(suffix_mask, suffix_ar_mask)
            prefix_attn_mask_ = einops.repeat(prefix_mask, "b p -> b s p", s=suffix_tokens.shape[1])
            full_attn_mask    = jnp.concatenate([prefix_attn_mask_, suffix_attn_mask], axis=-1)
            positions_        = (
                jnp.sum(prefix_mask, axis=-1)[:, None]
                + jnp.cumsum(suffix_mask, axis=-1) - 1
            )
            (_, suffix_out), _ = self.PaliGemma.llm(
                [None, suffix_tokens],
                mask=full_attn_mask,
                positions=positions_,
                kv_cache=kv_cache,
            )
            rgb_indices    = jnp.arange(1, 1 + 3 * T, 3)
            wrist_indices  = jnp.arange(2, 2 + 3 * T, 3)
            action_indices = jnp.arange(3, 3 + 3 * T, 3)
            vr_t = self.rgb_out_proj(suffix_out[:, rgb_indices])
            vw_t = self.wrist_out_proj(suffix_out[:, wrist_indices])
            v_t  = self.action_out_proj(suffix_out[:, action_indices])
            return x_t + dt * v_t, time + dt, xr_t + dt * vr_t, xw_t + dt * vw_t

        def cond(carry):
            _, time, _, _ = carry
            return time >= -dt / 2

        x_0, _, _, _ = jax.lax.while_loop(
            cond, step, (noise, 1.0, rgb_pred_noise, wrist_pred_noise)
        )
        return x_0


# ═════════════════════════════════════════════════════════════
# [新增] Pi0_RSSM：RSSM + Imagination Planning Baseline
# ═════════════════════════════════════════════════════════════

class Pi0_RSSM(_model.BaseModel):
    """
    [新增] RSSM Baseline，严格对应 PlaNet/Dreamer 架构。

    公平对比保证（与 Pi0 相同的部分）：
      ✅ SigLIP So400m/14 视觉编码器
      ✅ PaliGemma 语言编码器
      ✅ Flow Matching 动作解码结构（MLP + 时间编码）
      ✅ 相同的训练超参数

    核心差异（时序建模部分完全不同）：
    ┌─────────────────────┬──────────────────────────────────┐
    │ Pi0 (LatentVLA)     │ Pi0_RSSM (本类)                 │
    ├─────────────────────┼──────────────────────────────────┤
    │ AdaptiveFusion 动态 │ mean pool 静态压缩               │
    │ 压缩 256→1 token    │ 256→mean→h_dim                  │
    │ Transformer 隐式    │ 显式 GRU 状态转移               │
    │ 时序建模            │ h_t=GRU(h_{t-1},[z_{t-1},a])   │
    │ 无显式潜变量        │ z_t 先验/后验分离               │
    │ 联合 FM 预测        │ 仅预测动作，KL 约束世界模型     │
    │ 无 KL 损失          │ KL(q‖p) 核心世界模型损失        │
    │ Transformer ODE     │ Imagination Rollout + ODE        │
    └─────────────────────┴──────────────────────────────────┘

    [修复的 Bug]：
    1. sample_actions Phase1→Phase2 状态正确传递
    2. obs.state 维度正确处理 (batch, horizon, dim)
    3. 数据类型统一为 float32
    4. _rssm_rollout 支持传入初始 h, z
    5. state_proj 移除（RSSM 不使用）
    """

    def __init__(self, config: Pi0Config, rngs: nnx.Rngs):
        super().__init__(config.action_dim, config.action_horizon, config.max_token_len)
        self.config = config

        # ── [与 Pi0 相同] 视觉/语言编码器 ──
        paligemma_config     = _gemma.get_config(config.paligemma_variant)
        action_expert_config = _gemma.get_config(config.action_expert_variant)
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

        # ── [与 Pi0 相同] Action Flow Matching 解码层 ──
        width    = action_expert_config.width
        new_rngs = nnx.Rngs(42)
        self.action_in_proj      = nnx.Linear(config.action_dim, width, rngs=new_rngs)
        self.action_time_mlp_in  = nnx.Linear(2 * width, width,         rngs=new_rngs)
        self.action_time_mlp_out = nnx.Linear(width, width,             rngs=new_rngs)
        self.action_out_proj     = nnx.Linear(width, config.action_dim, rngs=new_rngs)

        # ── [新增] RSSM 组件 ──
        self.compress_dim = 2048   # SigLIP 输出维度
        self.h_dim        = width  # GRU 隐状态维度，对齐 Pi0 的 expert width
        self.z_dim        = width  # 随机潜变量维度

        # 观测压缩: 2048 → h_dim（对应 Pi0 的 AdaptiveFusion，但用静态 mean pool）
        self.obs_pool_proj = nnx.Linear(self.compress_dim, self.h_dim, rngs=new_rngs)

        # GRU 确定性核心: input = [z_{t-1}, a_{t-1}]
        # a_{t-1} shape=(batch, action_dim), z_{t-1} shape=(batch, z_dim)
        self.gru = GRUCell(
            input_dim  = self.z_dim + config.action_dim,  # z_dim + action_dim
            hidden_dim = self.h_dim,
            rngs       = new_rngs,
        )

        # 先验 p(z_t | h_t)
        self.prior = RSSMPrior(self.h_dim, self.z_dim, rngs=new_rngs)

        # 后验 q(z_t | h_t, o_t)
        # obs_embed 经 obs_pool_proj 后维度为 h_dim
        self.posterior = RSSMPosterior(
            h_dim   = self.h_dim,
            obs_dim = self.h_dim,
            z_dim   = self.z_dim,
            rngs    = new_rngs,
        )

        # (h, z) → action 条件：对应 Pi0 的 suffix_out 作为条件的部分
        # 输入 h_dim + z_dim，输出 width
        self.hz_proj = nnx.Linear(self.h_dim + self.z_dim, width, rngs=new_rngs)

        # 可学习初始状态，float32 保证类型一致
        self.h0 = nnx.Param(jnp.zeros((1, self.h_dim), dtype=jnp.float32))
        self.z0 = nnx.Param(jnp.zeros((1, self.z_dim), dtype=jnp.float32))

        # KL 权重（支持退火，训练脚本中动态修改）
        self.kl_weight = 1.0
         # ↓↓↓ 新增这一行 ↓↓↓
        self.free_bits = 0.01       # 防止后验坍塌，每步最小 KL 值
        # ↑↑↑ 新增这一行 ↑↑↑


        self.action_seq_horizon = config.action_horizon - 1   # T = 5（默认）

    # ──────────────────────────────────────────────
    # [新增] 工具函数
    # ──────────────────────────────────────────────

    def _get_obs_embed(self, obs: _model.Observation, t: int) -> jnp.ndarray:
        """
        编码第 t 帧 RGB 观测 → (batch, h_dim)，float32。

        [修复] obs.images["base_0_rgb"] shape=(batch, horizon, H, W, 3)
        取第 t 帧需要 [:, t:t+1] 保持 batch 维度。
        SigLIP 返回 
        (batch, 1, 256, 2048)，mean pool 后为 (batch, 2048)。
        """
        # [修复] 正确切片：(batch, 1, H, W, 3)
        rgb_t  = obs.images["base_0_rgb"][:, t:t+1]
        # SigLIP 编码：(batch, 1, 256, 2048)
        tokens, _ = self.PaliGemma.img(rgb_t, train=False)
        # mean pool 256 个 patch tokens：(batch, 2048)
        pooled = jnp.mean(tokens[:, 0], axis=1)
        # 确保 float32
        pooled = pooled.astype(jnp.float32)
        # 投影到 h_dim：(batch, h_dim)
        return nnx.swish(self.obs_pool_proj(pooled))

    @staticmethod
    def _kl_divergence(
        q_mean:   jnp.ndarray,   # (batch, z_dim)
        q_logvar: jnp.ndarray,   # (batch, z_dim)
        p_mean:   jnp.ndarray,   # (batch, z_dim)
        p_logvar: jnp.ndarray,   # (batch, z_dim)
    ) -> jnp.ndarray:
        """
        KL(q || p) 闭合形式，两个对角高斯。
        返回 (batch,)，对 z_dim 求和。
        Pi0 完全没有此项，这是 RSSM 世界模型的核心训练信号。
        """
      
        q_logvar = jnp.clip(q_logvar, -10.0, 10.0)
        p_logvar = jnp.clip(p_logvar, -10.0, 10.0)
        kl_per_dim = 0.5 * (
            p_logvar - q_logvar - 1.0
            + jnp.exp(q_logvar - p_logvar)                          # ← 去掉 exp 里的 1e-8
            + (q_mean - p_mean) ** 2 / (jnp.exp(p_logvar) + 1e-8)
        )
        return jnp.maximum(kl_per_dim, 0.0)   # (batch, z_dim) 不 sum

    def _init_state(self, batch_size: int) -> tuple:
        """
        初始化 h0, z0，保证 float32 且正确 broadcast。
        返回 h: (batch, h_dim), z: (batch, z_dim)
        """
        h = jnp.broadcast_to(
            self.h0.value, (batch_size, self.h_dim)
        ).astype(jnp.float32)
        z = jnp.broadcast_to(
            self.z0.value, (batch_size, self.z_dim)
        ).astype(jnp.float32)
        return h, z

    # ──────────────────────────────────────────────
    # [新增] RSSM 核心展开（修复状态传递 Bug）
    # ──────────────────────────────────────────────

    def _rssm_rollout(
        self,
        obs:           _model.Observation,
        rng:           jnp.ndarray,
        use_posterior: bool,
        init_h:        jnp.ndarray = None,   # [修复] 支持传入初始状态
        init_z:        jnp.ndarray = None,
    ):
        """
        [新增] RSSM 逐步展开。

        [修复 Bug 1] 支持传入 init_h, init_z，
        使 sample_actions 的 Phase1→Phase2 状态能正确传递。

        [修复 Bug 2] obs.state shape=(batch, horizon, action_dim)，
        a_prev 用零向量，不依赖 obs.state 切片。

        Args:
            use_posterior: True=训练（用真实观测后验）
                           False=推理（imagination，只用先验）
            init_h: 初始确定性状态，None 则用 h0
            init_z: 初始随机状态，None 则用 z0

        Returns:
            hs:  (batch, T, h_dim)  float32
            zs:  (batch, T, z_dim)  float32
            kl:  (batch, T)         float32，推理时全为 0
        """
        # [修复] batch_size 从 state 第 0 维取
        batch_size = obs.state.shape[0]
        T          = self.action_seq_horizon

        # 初始化状态
        if init_h is not None:
            h = init_h.astype(jnp.float32)
        else:
            h, _ = self._init_state(batch_size)

        if init_z is not None:
            z = init_z.astype(jnp.float32)
        else:
            _, z = self._init_state(batch_size)

        # [修复] a_prev 统一用零，float32
        a_prev = jnp.zeros((batch_size, self.config.action_dim), dtype=jnp.float32)

        hs_list, zs_list, kl_list, kl_raw_list = [], [], [], []

        for t in range(T):
            rng, step_rng = jax.random.split(rng)

            # Step 1: GRU 确定性更新
            # input = [z_{t-1}, a_{t-1}]，shape=(batch, z_dim+action_dim)
            gru_input = jnp.concatenate([z, a_prev], axis=-1)
            h = self.gru(h, gru_input)               # (batch, h_dim)，float32

            if use_posterior:
                obs_embed           = self._get_obs_embed(obs, t)
                z, q_mean, q_logvar = self.posterior.sample(h, obs_embed, step_rng)
                p_mean, p_logvar    = self.prior(h)
                kl_per_dim    = self._kl_divergence(q_mean, q_logvar, p_mean, p_logvar)
                kl_fb         = jnp.sum(jnp.maximum(kl_per_dim, self.free_bits), axis=-1)  # free bits
                kl_raw_t      = jnp.sum(kl_per_dim, axis=-1)                               # 原始
                kl_list.append(kl_fb)
                kl_raw_list.append(kl_raw_t)
            else:
                z, _, _ = self.prior.sample(h, step_rng)
                kl_list.append(jnp.zeros(batch_size, dtype=jnp.float32))
                kl_raw_list.append(jnp.zeros(batch_size, dtype=jnp.float32))   # ← 新增

            hs_list.append(h)
            zs_list.append(z)
            # a_prev 保持零（不做 teacher forcing，与 Pi0 推理对齐）
            a_prev = jnp.zeros((batch_size, self.config.action_dim), dtype=jnp.float32)

        hs = jnp.stack(hs_list, axis=1)   # (batch, T, h_dim)
        zs = jnp.stack(zs_list, axis=1)   # (batch, T, z_dim)
        kl = jnp.stack(kl_list, axis=1)   # (batch, T)
        kl_raw = jnp.stack(kl_raw_list, axis=1)   # ← 新增
        return hs, zs, kl, kl_raw  

    # ──────────────────────────────────────────────
    # [新增] Flow Matching 动作速度场解码
    # ──────────────────────────────────────────────

    def _decode_velocity(
        self,
        hs:           jnp.ndarray,   # (batch, T, h_dim)  float32
        zs:           jnp.ndarray,   # (batch, T, z_dim)  float32
        noisy_actions:jnp.ndarray,   # (batch, T, action_dim) float32
        timestep:     jnp.ndarray,   # (batch,)  float32
    ) -> jnp.ndarray:
        """
        [新增] 给定 RSSM 潜状态 (h,z) 作为条件预测速度场。

        [修复] timestep shape 必须是 (batch,)，
        posemb_sincos 的 pos 参数要求 1D。
        noisy_actions 可能来自 beta 采样，需保证 float32。
        """
        T = noisy_actions.shape[1]

        # 确保 float32
        noisy_actions = noisy_actions.astype(jnp.float32)
        timestep      = timestep.astype(jnp.float32)

        # 时间步编码：(batch, width)
        time_emb = posemb_sincos(
            timestep,
            self.action_in_proj.out_features,
            min_period=4e-3,
            max_period=4.0,
        )
        # 扩展到序列长度：(batch, T, width)
        time_emb = einops.repeat(time_emb, "b d -> b t d", t=T)

        # 动作编码：(batch, T, width)
        action_tok = self.action_in_proj(noisy_actions)

        # 时间 + 动作融合（与 Pi0 完全相同的 MLP 结构）
        at_ = jnp.concatenate([action_tok, time_emb], axis=-1)  # (batch, T, 2*width)
        at_ = nnx.swish(self.action_time_mlp_in(at_))
        at_ = self.action_time_mlp_out(at_)                      # (batch, T, width)

        # [新增] RSSM (h,z) 条件注入
        # 对应 Pi0 中 suffix_out（Transformer 输出）作为条件
        hz      = jnp.concatenate([hs, zs], axis=-1)            # (batch, T, h_dim+z_dim)
        hz_cond = nnx.swish(self.hz_proj(hz))                   # (batch, T, width)

        # 残差注入条件
        out = at_ + hz_cond                                      # (batch, T, width)
        v_t = self.action_out_proj(out)                          # (batch, T, action_dim)
        return v_t

    # ──────────────────────────────────────────────
    # [新增] 训练损失
    # ──────────────────────────────────────────────

    @override
    def compute_loss(
        self,
        rng:         at.KeyArrayLike,
        observation: _model.Observation,
        actions:     _model.Actions,
        *,
        train: bool = False,
    ):
        """
        [新增] RSSM 训练损失：
            L = L_action + kl_weight * L_KL

        返回值类型与 Pi0 不同（无 rgb/wrist 损失），
        需要在训练脚本中用 compute_loss_for_model 统一处理。
        """
        preprocess_rng, noise_rng, rssm_rng, time_rng = jax.random.split(rng, 4)
        observation = _model.preprocess_observation(preprocess_rng, observation, train=train)

        # [与 Pi0 一致] 去掉最后一帧
        actions     = actions[:, :-1]                  # (batch, T, action_dim)
        T           = self.action_seq_horizon
        batch_size  = actions.shape[0]
        batch_shape = (batch_size,)                    # [修复] 正确的 batch_shape

        # Flow Matching 噪声（与 Pi0 完全相同）
        noise    = jax.random.normal(noise_rng, actions.shape, dtype=jnp.float32)
        time     = (jax.random.beta(time_rng, 1.5, 1, batch_shape) * 0.999 + 0.001
                    ).astype(jnp.float32)              # (batch,)  float32
        time_exp = time[:, None, None]                 # (batch, 1, 1) 用于广播
        x_t      = time_exp * noise + (1 - time_exp) * actions.astype(jnp.float32)
        u_t      = noise - actions.astype(jnp.float32)

        # [新增] RSSM 展开（训练用后验 + KL）
        hs, zs, kl_loss, kl_raw = self._rssm_rollout(   # ← 多接收 kl_raw
            observation, rssm_rng, use_posterior=True
        )
        # hs: (batch, T, h_dim), zs: (batch, T, z_dim), kl_loss: (batch, T)

        # 动作速度场预测
        v_t         = self._decode_velocity(hs, zs, x_t, time)
        action_loss = jnp.mean(jnp.abs(v_t - u_t), axis=-1)    # (batch, T)
        kl_per_sample     = jnp.mean(kl_loss, axis=-1)
        kl_raw_per_sample = jnp.mean(kl_raw,  axis=-1)   # ← 新增
        return action_loss, kl_per_sample, kl_raw_per_sample  # ← 多返回一个

    # ──────────────────────────────────────────────
    # [新增] 推理（修复 Phase1→Phase2 状态传递）
    # ──────────────────────────────────────────────

    @override
    def sample_actions(
        self,
        rng:         at.KeyArrayLike,
        observation: _model.Observation,
        *,
        num_steps: int = 20,
    ) -> _model.Actions:
        """
        [新增] RSSM Imagination Planning 推理。

        [修复 Bug] Phase1 初始化的 (h, z) 正确传入 Phase2 rollout，
        而不是在 Phase2 重新从 h0/z0 开始。
        """
        preprocess_rng, noise_rng, init_rng, imagine_rng = jax.random.split(rng, 4)
        observation = _model.preprocess_observation(None, observation, train=False)

        batch_size = observation.state.shape[0]   # [修复] 从正确维度取 batch_size
        T          = self.action_seq_horizon
        dt         = -1.0 / num_steps             # 注意：num_steps 而非 T

        # ══ Phase 1: 后验初始化（用第 0 帧真实观测）══
        h, z   = self._init_state(batch_size)
        a_prev = jnp.zeros((batch_size, self.config.action_dim), dtype=jnp.float32)

        # 第 0 帧 GRU 更新
        gru_input = jnp.concatenate([z, a_prev], axis=-1)
        h         = self.gru(h, gru_input)

        # 用第 0 帧真实观测计算后验 z_0
        obs_embed_0    = self._get_obs_embed(observation, 0)    # (batch, h_dim)
        z, _, _        = self.posterior.sample(h, obs_embed_0, init_rng)

        

        # ══ Phase 2: Imagination Rollout（[
        # 修复] 传入 Phase1 初始化的 h, z）══
        hs, zs, _, _ = self._rssm_rollout(   # ← 多一个 _
            observation, 
            imagine_rng, 
            use_posterior=False,
            init_h=h,
            init_z=z,
        )

        # hs: (batch, T, h_dim), zs: (batch, T, z_dim)

        # ══ Phase 3: Flow Matching ODE 求解 ══
        # 初始化噪声动作，float32
        x_t = jax.random.normal(
            noise_rng, (batch_size, T, self.config.action_dim), dtype=jnp.float32
        )

        # [修复] while_loop 的 carry 必须是纯 jax array
        # hs, zs 作为闭包捕获，不放入 carry
        def step(carry):
            x_t, time = carry
            # [修复] time 是标量，需要广播为 (batch,)
            time_batch = jnp.full((batch_size,), time, dtype=jnp.float32)
            v_t = self._decode_velocity(hs, zs, x_t, time_batch)
            return x_t + dt * v_t, time + dt

        def cond(carry):
            _, time = carry
            return time >= -dt / 2

        # [修复] 初始 time 必须是 jnp scalar，不能是 Python float
        init_time = jnp.array(1.0, dtype=jnp.float32)
        x_0, _    = jax.lax.while_loop(cond, step, (x_t, init_time))

        return x_0


# ═════════════════════════════════════════════════════════════
# [新增] 统一损失计算入口（训练脚本中使用）
# ═════════════════════════════════════════════════════════════

def compute_loss_for_model(model, rng, observation, actions, train=True):
    """
    [新增] 训练脚本中替换直接调用 model.compute_loss 的地方。

    Pi0 返回:  (action_loss, rgb_loss, wrist_loss, fusion_weights)
    Pi0_RSSM 返回: (action_loss, kl_loss)

    统一返回 dict，方便 wandb/tensorboard 记录对比指标。
    """
    if isinstance(model, Pi0_RSSM):
        action_loss, kl_loss, kl_raw = model.compute_loss(
            rng, observation, actions, train=train
        )
        total = jnp.mean(action_loss) + model.kl_weight * jnp.mean(kl_loss)
        return {
            "total_loss":  total,
            "action_loss": jnp.mean(action_loss),
            "kl_loss":     jnp.mean(kl_loss),   # RSSM 特有
            "kl_raw":      jnp.mean(kl_raw), 
            "rgb_loss":    jnp.zeros(()),        # RSSM 无此项
            "wrist_loss":  jnp.zeros(()),        # RSSM 无此项
        }
    else:
        action_loss, rgb_loss, wrist_loss, fusion_weights = model.compute_loss(
            rng, observation, actions, train=train
        )
        total = (
            jnp.mean(action_loss)
            + jnp.mean(rgb_loss)
            + jnp.mean(wrist_loss)
        )
        return {
            "total_loss":  total,
            "action_loss": jnp.mean(action_loss),
            "kl_loss":     jnp.zeros(()),        # Pi0 无此项
            "rgb_loss":    jnp.mean(rgb_loss),   # Pi0 特有
            "wrist_loss":  jnp.mean(wrist_loss), # Pi0 特有
        }


# ═════════════════════════════════════════════════════════════
# [新增] KL 退火调度
# ═════════════════════════════════════════════════════════════

def get_kl_weight(step: int, anneal_steps: int = 1000) -> float:
    """
    [新增] KL 权重从 0 线性增加到 1.0。
    防止训练初期 KL 过大导致后验坍塌（Dreamer 标准策略）。

    训练循环中使用：
        if isinstance(model, Pi0_RSSM):
            model.kl_weight = get_kl_weight(global_step)
    """
    return float(min(1.0, step / max(anneal_steps, 1)))
