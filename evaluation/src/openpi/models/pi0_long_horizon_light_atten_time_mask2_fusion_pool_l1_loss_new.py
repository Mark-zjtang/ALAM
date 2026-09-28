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
from diffusers.models import AutoencoderKL
import jax.nn as jnn

import numpy as np
import os
import jax

_token_buffer = {"before": [], "after": []}
_token_save_count = 0

def _save_tokens_numpy(before_np, after_np):
    """jit外执行，可以用numpy"""
    global _token_save_count
    
    _token_buffer["before"].append(np.array(before_np, dtype=np.float32))
    _token_buffer["after"].append(np.array(after_np, dtype=np.float32))
    _token_save_count += 1

    if _token_save_count % 20 == 0:
        save_dir = "/mnt/workspace/tangzuojin.tzj/physics_latent_world/token_analysis"
        os.makedirs(save_dir, exist_ok=True)
        np.savez(
            os.path.join(save_dir, "tokens.npz"),
            before_pool=np.array(_token_buffer["before"]),  # [N, 256, 2048]
            after_pool=np.array(_token_buffer["after"]),    # [N, 2048]
        )
        print(f"[TokenAnalysis] 已保存 {_token_save_count} 个样本")


def _save_tokens_for_analysis(before_pool, after_pool):
    """jit内调用，切片在这里做"""
    before_slice = before_pool[0, 0]  # (256, 2048)
    after_slice  = after_pool[0, 0]   # (2048,)
    jax.debug.callback(_save_tokens_numpy, before_slice, after_slice)





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


def vae_model():
    return AutoencoderKL.from_pretrained("/home/fortress/new_storage/TZJ/openpi_worldflow/sd-vae-ft-mse/")


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
        # new_params_filter = nnx_utils.PathRegex(".*(img_pool_query|rgb_in_proj|rgb_out_proj|wrist_in_proj|wrist_out_proj).*")
        if "lora" in self.paligemma_variant:
            filters.append(
                gemma_params_filter,
            )
            if "lora" not in self.action_expert_variant:
                # If only freeze gemma params, exclude action expert params.
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
            # If any lora is used, exclude all lora params.
            filters.append(
                nnx.Not(nnx_utils.PathRegex(".*lora.*")),
            )

        # Not filter new parmas
        # filters.append(nnx.Not(new_params_filter))

        if not filters:
            return nnx.Nothing
        return nnx.All(*filters)

# 在文件顶部确保有正确的导入
from flax import nnx
import jax.numpy as jnp
import jax
import einops

class AdaptiveFusion(nnx.Module):
    def __init__(self, temperature=0.1):
        super().__init__()
        self.temperature = temperature
        # 使用 nnx 的方式定义参数
        self.alpha = nnx.Param(jnp.ones(3))
    
    def __call__(self, image_tokens_fusion, name, attentin_query_rgb, attentin_query_wrist):
        batch_size, action_horizon, num_tokens, hidden_dim = image_tokens_fusion.shape
        
        # 1. 计算三种pooling结果
        max_pool = self._max_pool_attention(image_tokens_fusion)
        sum_pool = self._sum_pool_attention(image_tokens_fusion) 
        learned_pool = self._learned_attention(image_tokens_fusion, name, attentin_query_rgb, attentin_query_wrist)
        
        # 2. 学习权重并softmax归一化
        weights = jax.nn.softmax(self.alpha / self.temperature)
        
        # 3. 加权融合
        fused_tokens = (weights[0] * max_pool + 
                       weights[1] * sum_pool + 
                       weights[2] * learned_pool)
        
        return fused_tokens, weights
    
    def _max_pool_attention(self, tokens):
        attention_scores = tokens.max(axis=-1, keepdims=True)
        attention_weights = jax.nn.softmax(attention_scores / self.temperature, axis=2)
        weighted_tokens = attention_weights * tokens
        return jnp.max(weighted_tokens, axis=2)
    
    def _sum_pool_attention(self, tokens):
        attention_scores = tokens.sum(axis=-1, keepdims=True)
        attention_weights = jax.nn.softmax(attention_scores, axis=2)
        weighted_tokens = attention_weights * tokens
        return jnp.sum(weighted_tokens, axis=2)
    
    def _learned_attention(self, tokens, name, attentin_query_rgb, attentin_query_wrist):
        if name == "base_0_rgb":
            attention_scores = attentin_query_rgb(tokens)
        elif name == "left_wrist_0_rgb":
            attention_scores = attentin_query_wrist(tokens)
        else:
            attention_scores = attentin_query_rgb(tokens)
            
        attention_weights = jax.nn.softmax(attention_scores / self.temperature, axis=2)
        weighted_tokens = attention_weights * tokens
        return jnp.sum(weighted_tokens, axis=2)

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

        self.attentin_query_rgb = nnx.Linear(self.compress_dim, 1, rngs=new_rngs)
        self.attentin_query_wrist = nnx.Linear(self.compress_dim, 1, rngs=new_rngs)


        self.image_chunk = 1
        self.action_seq_horizon = self.action_horizon - 1
        self.img_seq_horizon = (self.action_horizon - 1) * self.image_chunk  # default = self.action_horizon-1

        self.adaptive_fusion = AdaptiveFusion()

  


    def vectorized_unpatchify(self, x, patch_size=2, image_size=(224, 224), out_channels=3):
        """

        Args:
            x: (batch_size, seq_len, patch_h, patch_w, in_channels)
            patch_size: patch大小
            image_size: 目标图像尺寸
            out_channels: 输出通道数

        Returns:
            imgs: (batch_size, seq_len, image_h, image_w, out_channels)
        """
        batch_size, seq_len, patch_h, patch_w, in_channels = x.shape
        image_h, image_w = image_size
        p = patch_size

        # 计算网格
        grid_h = image_h // p
        grid_w = image_w // p

        # 重塑为序列格式: (batch_size * seq_len, grid_h, grid_w, p, p, in_channels)
        x_reshaped = x.reshape(batch_size * seq_len, grid_h, grid_w, p, p, in_channels)

        # 转置维度: (batch_size * seq_len, grid_h, p, grid_w, p, in_channels)
        x_transposed = jnp.transpose(x_reshaped, (0, 1, 3, 2, 4, 5))

        # : (batch_size * seq_len, image_h, image_w, in_channels)
        imgs_flat = x_transposed.reshape(batch_size, seq_len, image_h, image_w, in_channels)

        return imgs

    
    @at.typecheck
    def embed_prefix(
            self, obs: _model.Observation
    ) -> tuple[at.Float[at.Array, "b s emb"], at.Bool[at.Array, "b s"], at.Bool[at.Array, " s"], at.Float[
        at.Array, "..."], at.Float[at.Array, "..."]]:
        input_mask = []
        ar_mask = []
        current_tokens = []
        predict_tokens = []
        pool_weights = []
        

        # embed images
        for name in obs.images:
            if name == "right_wrist_0_rgb":
                continue

            print(f"{name}.shape", obs.images[name].shape)
            # input_image.shape: (batch_size, action_horizon, 224, 224, 3)
            print(f"=======================input {name}=====================================")
            image_tokens_fusion, _ = self.PaliGemma.img(obs.images[name],train=False)  # (batch_size, action_horizon, 256, 2048)
            print("image_tokens_fusion.shape", image_tokens_fusion.shape)
            
            fused_tokens, fusion_weights = self.adaptive_fusion(
            image_tokens_fusion, 
            name,
            self.attentin_query_rgb,  
            self.attentin_query_wrist  
        )
            print("fused_tokens.shape", fused_tokens.shape)  # (batch_size, action_horizon, 1, 2048)

            fused_tokens_expanded = fused_tokens[:, :, None, :] 
            print("fused_tokens_expanded.shape", fused_tokens_expanded.shape)  # (32, 11, 1, 2048)
            
            image_pool_tokens = einops.repeat(
                fused_tokens_expanded, 
                "b t 1 d -> b t (repeat d)", 
                repeat=1
            )  # (batch_size, action_horizon, 1, 2048)
            
            print("image_pool_tokens.shape", image_pool_tokens.shape)
            current_image_tokens = image_pool_tokens[:, :self.image_chunk]
            print("current_image_tokens.shape", current_image_tokens.shape)
            predict_image_tokens = image_pool_tokens[:, self.image_chunk:]
            print("predict_image_tokens.shape", predict_image_tokens.shape)

            current_tokens.append(current_image_tokens)
            predict_tokens.append(predict_image_tokens)
            input_mask.append(
                einops.repeat(
                    obs.image_masks[name],
                    "b -> b s",
                    s=current_image_tokens.shape[1],
                )
            )
            # image tokens attend to each other
            ar_mask += [False] * current_image_tokens.shape[1]

            if name == "base_0_rgb":
                _save_tokens_for_analysis(
                before_pool=image_tokens_fusion,  # (batch, horizon, 256, 2048)
                after_pool=fused_tokens,          # (batch, horizon, 2048)
                )

        # add language (aka tokenized inputs)
        if obs.tokenized_prompt is not None:
            tokenized_inputs = self.PaliGemma.llm(obs.tokenized_prompt, method="embed")
            print("tokenized_inputs.shape", tokenized_inputs.shape)
            current_tokens.append(tokenized_inputs)
            input_mask.append(obs.tokenized_prompt_mask)
            # full attention between image and language inputs
            ar_mask += [False] * tokenized_inputs.shape[1]
        current_tokens = jnp.concatenate(current_tokens, axis=1)
        predict_tokens = jnp.concatenate(predict_tokens, axis=1)
        print("final_current_tokens.shape", current_tokens.shape)
        print("final_predict_tokens.shape", predict_tokens.shape)
        input_mask = jnp.concatenate(input_mask, axis=1)
        print("image_input_mask", input_mask.shape)
        ar_mask = jnp.array(ar_mask)

    
        return current_tokens, input_mask, ar_mask, predict_tokens, fusion_weights


    @at.typecheck
    def embed_suffix(
            self, obs: _model.Observation, noisy_actions: _model.Actions, timestep: at.Float[at.Array, " b"],
            rgb_predict_tokens: at.Float[at.Array, "..."], wrist_predict_tokens: at.Float[at.Array, "..."]
    ) -> tuple[at.Float[at.Array, "..."], at.Bool[at.Array, "..."], at.Bool[at.Array, "..."]]:
        input_mask = []
        ar_mask = []
        tokens = []

        rgb_input_mask = []
        rgb_ar_mask = []
        rgb_tokens = []

        wrist_input_mask = []
        wrist_ar_mask = []
        wrist_tokens = []

        # add a single state token
        state_token = self.state_proj(obs.state)[:, None, :]
        tokens.append(state_token)
        input_mask.append(jnp.ones((obs.state.shape[0], 1), dtype=jnp.bool_))
        # image/language inputs do not attend to state or actions
        ar_mask += [True]

        # embed timestep using sine-cosine positional encoding with sensitivity in the range [0, 1]
        time_emb = posemb_sincos(timestep, self.action_in_proj.out_features, min_period=4e-3, max_period=4.0)
        # mix timestep + action information using an MLP

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
            
            # rgb token at timestep t
            tokens.append(rgb_time_tokens[:, t:t+1, :])
            input_mask.append(jnp.ones((batch_size, 1), dtype=jnp.bool_))
            
            # wrist token at timestep t
            tokens.append(wrist_time_tokens[:, t:t+1, :])
            input_mask.append(jnp.ones((batch_size, 1), dtype=jnp.bool_))

            # action token at timestep t
            tokens.append(action_time_tokens[:, t:t+1, :])
            input_mask.append(jnp.ones((batch_size, 1), dtype=jnp.bool_))
            
        
        tokens = jnp.concatenate(tokens, axis=1)
        input_mask = jnp.concatenate(input_mask, axis=1)
        

        for t in range(time_steps):
            
            # RGB token for timestep t
            ar_mask.append(True)  
            
            # Wrist token for timestep t
            ar_mask.append(False) 

            # Action token for timestep t
            ar_mask.append(True)  
        
    
        print("final.tokens.shape", tokens.shape)
        print("final.input_mask.shape", input_mask.shape)
        ar_mask = jnp.array(ar_mask)
        print("final.ar_mask.shape", ar_mask.shape)
     
    
        return tokens, input_mask, ar_mask


    @override
    def compute_loss(
            self, rng: at.KeyArrayLike, observation: _model.Observation, actions: _model.Actions, *, train: bool = False
    ) -> at.Float[at.Array, "*b ah"]:
        preprocess_rng, noise_rng, rgb_noise_rng, wrist_noise_rng, time_rng = jax.random.split(rng, 5)
        observation = _model.preprocess_observation(preprocess_rng, observation, train=train)
        print("===================================================long horzion==========================================")
        actions = actions[:, :-1]
        batch_shape = actions.shape[:-2]
        noise = jax.random.normal(noise_rng, actions.shape)

        time = jax.random.beta(time_rng, 1.5, 1, batch_shape) * 0.999 + 0.001
        time_expanded_action = time[..., None, None]

        x_t = time_expanded_action * noise + (1 - time_expanded_action) * actions
        u_t = noise - actions

        prefix_tokens, prefix_mask, prefix_ar_mask, predict_tokens, fusion_weights = self.embed_prefix(observation)
        print("prefix_tokens.shape", prefix_tokens.shape)

        rgb_pred = predict_tokens[:, : self.img_seq_horizon]
        wrist_pred = predict_tokens[:, self.img_seq_horizon: 2 * self.img_seq_horizon]

        rgb_pred_noise = jax.random.normal(rgb_noise_rng, rgb_pred.shape)
        wrist_pred_noise = jax.random.normal(rgb_noise_rng, wrist_pred.shape)

        xr_t = time_expanded_action * rgb_pred_noise + (1 - time_expanded_action) * rgb_pred
        ur_t = rgb_pred_noise - rgb_pred

        xw_t = time_expanded_action * wrist_pred_noise + (1 - time_expanded_action) * wrist_pred
        uw_t = wrist_pred_noise - wrist_pred

        suffix_tokens, suffix_mask, suffix_ar_mask = self.embed_suffix(observation, x_t, time, xr_t, xw_t)
        print("suffix_tokens.shape", suffix_tokens.shape)
        input_mask = jnp.concatenate([prefix_mask, suffix_mask], axis=1)
        ar_mask = jnp.concatenate([prefix_ar_mask, suffix_ar_mask], axis=0)
        attn_mask = make_attn_mask(input_mask, ar_mask)
        positions = jnp.cumsum(input_mask, axis=1) - 1

        (prefix_out, suffix_out), _ = self.PaliGemma.llm([prefix_tokens, suffix_tokens], mask=attn_mask, positions=positions)

        print("suffix_out_0.shape:", suffix_out.shape)

    
        # new token order: [state, rgb_t1, wrist_t1, action_t1, rgb_t2, wrist_t2, action_t2 ...]

        state_idx = 0  
        
        
        # RGB tokens: Position 1, 4, 7, ... 
        rgb_start_idx = 1
        rgb_indices = jnp.arange(rgb_start_idx, rgb_start_idx + 3 * self.action_seq_horizon, 3)
        
        # Wrist tokens: Position 2, 5, 8, ... 
        wrist_start_idx = 2
        wrist_indices = jnp.arange(wrist_start_idx, wrist_start_idx + 3 * self.action_seq_horizon, 3)

        # Action tokens: Position 3, 6, 9, ... 
        action_start_idx = 3
        action_indices = jnp.arange(action_start_idx, action_start_idx + 3 * self.action_seq_horizon, 3)

        vr_t = self.rgb_out_proj(suffix_out[:, rgb_indices])
        print("vr_t.shape", vr_t.shape)

        vw_t = self.wrist_out_proj(suffix_out[:, wrist_indices])
        print("vw_t.shape", vw_t.shape)

        v_t = self.action_out_proj(suffix_out[:, action_indices])
        print("v_t.shape", v_t.shape)
     
        return jnp.mean(jnp.abs(v_t - u_t), axis=-1), jnp.mean(jnp.abs(vr_t - ur_t), axis=-1), jnp.mean(jnp.abs(vw_t - uw_t), axis=-1), fusion_weights

    @override
    def sample_actions(
            self,
            rng: at.KeyArrayLike,
            observation: _model.Observation,
            *,
            num_steps: int=10, #num_steps: int | at.Int[at.Array, ""] = 10
    ) -> _model.Actions:
        preprocess_rng, noise_rng, rgb_noise_rng, wrist_noise_rng, time_rng = jax.random.split(rng, 5)
    
        img_seq_horizon = self.action_seq_horizon * self.image_chunk  # default = self.action_horizon-1
        action_seq_horizon = self.action_seq_horizon
        observation = _model.preprocess_observation(None, observation, train=False)
        # note that we use the convention more common in diffusion literature, where t=1 is noise and t=0 is the target
        # distribution. yes, this is the opposite of the pi0 paper, and I'm sorry.
        dt = -1.0 / action_seq_horizon
        batch_size = observation.state.shape[0]
        print("=============================action_seq_horizon================================", self.action_seq_horizon)
        noise = jax.random.normal(noise_rng, (batch_size, action_seq_horizon, self.action_dim))
        print("noise.shape", noise.shape)
        rgb_pred_noise = jax.random.normal(rgb_noise_rng, (batch_size, img_seq_horizon, 2048))
        print("rgb_pred_noise.shape", rgb_pred_noise.shape)
        wrist_pred_noise = jax.random.normal(rgb_noise_rng, (batch_size, img_seq_horizon, 2048))
        print("wrist_pred_noise.shape", wrist_pred_noise.shape)

        prefix_tokens, prefix_mask, prefix_ar_mask, predict_tokens, fusion_weights = self.embed_prefix(observation)
        print("weight_fusion", fusion_weights)
        prefix_attn_mask = make_attn_mask(prefix_mask, prefix_ar_mask)
        positions = jnp.cumsum(prefix_mask, axis=1) - 1
        _, kv_cache = self.PaliGemma.llm([prefix_tokens, None], mask=prefix_attn_mask, positions=positions)
        # first fill KV cache with a forward pass of the prefix
        def step(carry):
            x_t, time, xr_t, xw_t = carry
            time_for_inference = jnp.full((x_t.shape[0],), time)  # (batch_size,)

            suffix_tokens, suffix_mask, suffix_ar_mask = self.embed_suffix(
                observation, x_t, time_for_inference, xr_t, xw_t
            )

            # `suffix_attn_mask` is shape (b, suffix_len, suffix_len) indicating how the suffix tokens can attend to each
            # other
            suffix_attn_mask = make_attn_mask(suffix_mask, suffix_ar_mask)
            # `prefix_attn_mask` is shape (b, suffix_len, prefix_len) indicating how the suffix tokens can attend to the
            # prefix tokens
            prefix_attn_mask = einops.repeat(prefix_mask, "b p -> b s p", s=suffix_tokens.shape[1])
            # `combined_mask` is shape (b, suffix_len, prefix_len + suffix_len) indicating how the suffix tokens (which
            # generate the queries) can attend to the full prefix + suffix sequence (which generates the keys and values)
            full_attn_mask = jnp.concatenate([prefix_attn_mask, suffix_attn_mask], axis=-1)
            assert full_attn_mask.shape == (
                batch_size,
                suffix_tokens.shape[1],
                prefix_tokens.shape[1] + suffix_tokens.shape[1],
            )
            # `positions` is shape (b, suffix_len) indicating the positions of the suffix tokens
            positions = jnp.sum(prefix_mask, axis=-1)[:, None] + jnp.cumsum(suffix_mask, axis=-1) - 1

            (prefix_out, suffix_out), _ = self.PaliGemma.llm(
                [None, suffix_tokens],
                mask=full_attn_mask,
                positions=positions,
                kv_cache=kv_cache,
            )
            assert prefix_out is None

            # new token order: [state, rgb_t1, wrist_t1, action_t1, rgb_t2, wrist_t2, action_t2 ...]

            state_idx = 0  
            
            # RGB tokens: Position 1, 4, 7, ... 
            rgb_start_idx = 1
            rgb_indices = jnp.arange(rgb_start_idx, rgb_start_idx + 3 * self.action_seq_horizon, 3)
            
            # Wrist tokens: Position 2, 5, 8, ... 
            wrist_start_idx = 2
            wrist_indices = jnp.arange(wrist_start_idx, wrist_start_idx + 3 * self.action_seq_horizon, 3)

            # Action tokens: Position 3, 6, 9, ... 
            action_start_idx = 3
            action_indices = jnp.arange(action_start_idx, action_start_idx + 3 * self.action_seq_horizon, 3)


            vr_t = self.rgb_out_proj(suffix_out[:, rgb_indices])
            print("vr_t.shape", vr_t.shape)

            vw_t = self.wrist_out_proj(suffix_out[:, wrist_indices])
            print("vw_t.shape", vw_t.shape)

            v_t = self.action_out_proj(suffix_out[:, action_indices])
            print("v_t.shape", v_t.shape)

            return x_t + dt * v_t, time + dt, xr_t + dt * vr_t, xw_t + dt * vw_t

        def cond(carry):
            x_t, time, xr_t, xw_t = carry
            return time >= -dt / 2

        x_0, _, _, _ = jax.lax.while_loop(cond, step, (noise, 1.0, rgb_pred_noise, wrist_pred_noise))
        return x_0
