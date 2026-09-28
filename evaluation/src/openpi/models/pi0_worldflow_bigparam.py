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
logger = logging.getLogger("openpi")
from diffusers.models import AutoencoderKL
import jax.nn as jnn


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
    action_dim: int = 32 # default=32
    action_horizon: int = 11 # default=50
    max_token_len: int = 48 # default=48

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
        self.rgb_in_proj = nnx.Linear(2048, action_expert_config.width, rngs=new_rngs)
        self.rgb_out_proj = nnx.Linear(action_expert_config.width, 2048, rngs=new_rngs)
        self.rgb_time_mlp_in = nnx.Linear(2 * action_expert_config.width, action_expert_config.width, rngs=new_rngs)
        self.rgb_time_mlp_out = nnx.Linear(action_expert_config.width, action_expert_config.width, rngs=new_rngs)
        
        self.wrist_in_proj = nnx.Linear(2048, action_expert_config.width, rngs=new_rngs)
        self.wrist_out_proj = nnx.Linear(action_expert_config.width, 2048, rngs=new_rngs)
        self.wrist_time_mlp_in = nnx.Linear(2 * action_expert_config.width, action_expert_config.width, rngs=new_rngs)
        self.wrist_time_mlp_out = nnx.Linear(action_expert_config.width, action_expert_config.width, rngs=new_rngs)
        
        self.attentin_query = nnx.Linear(2048, 1, rngs=new_rngs)
        self.pool_fusion = nnx.Linear(2048*4, 2048, rngs=new_rngs)

        self.image_chunk = 4
        self.action_seq_horizon = self.action_horizon-1
        self.img_seq_horizon = (self.action_horizon-1)*self.image_chunk # default = self.action_horizon-1

    def simple_image_upsampler(self, x, target_size=(224, 224, 3)):
        """
        最简单的图像上采样器
        """
        batch_size, seq_len, h, w, c = x.shape
        target_h, target_w, target_c = target_size
        
        # 重塑: (batch_size * seq_len, h, w, c)
        x_flat = x.reshape(-1, h, w, c)
        
        # 使用插值
        from jax.image import resize
        upsampled = resize(x_flat, (x_flat.shape[0], target_h, target_w, c), 
                        method='bilinear')
        
        # 简单的通道调整：从2048降到3
        if c == 2048 and target_c == 3:
            # 方法1: 取前3个通道
            upsampled = upsampled[:, :, :, :3]
            # 方法2: 或者使用平均
            # upsampled = jnp.mean(upsampled, axis=-1, keepdims=True)
            # upsampled = jnp.tile(upsampled, (1, 1, 1, 3))
        
        # 标准化
        upsampled = jnp.tanh(upsampled)
        
        # 重塑: (batch_size, seq_len, target_h, target_w, target_c)
        result = upsampled.reshape(batch_size, seq_len, target_h, target_w, target_c)
        
        return result
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
        
        # 合并块: (batch_size * seq_len, image_h, image_w, in_channels)
        imgs_flat = x_transposed.reshape(batch_size, seq_len, image_h, image_w, in_channels)
     
        return imgs
      


    @at.typecheck
    def embed_prefix(
        self, obs: _model.Observation
    ) -> tuple[at.Float[at.Array, "b s emb"], at.Bool[at.Array, "b s"], at.Bool[at.Array, " s"], at.Float[at.Array, "..."]]:
        input_mask = []
        ar_mask = []
        current_tokens = []
        predict_tokens = []

        # embed images
        for name in obs.images:
            if name == "right_wrist_0_rgb":
                continue

            print(f"{name}.shape", obs.images[name].shape)
            # input_image.shape: (batch_size, action_horizon, 224, 224, 3)
            print(f"=======================input {name}=====================================")
            image_tokens, _ = self.PaliGemma.img(obs.images[name], train=False) # (batch_size, action_horizon, 256, 2048)

            print("image_tokens.shape", image_tokens.shape) 
           
            # attention_weights by max pooling
            attention_scores_max = image_tokens.max(axis=-1, keepdims=True)  # (batch_size, action_horizon, 256, 1)
            print("attention_scores_max.shape", attention_scores_max.shape)
            temperature = 0.1  
            attention_weights_max = jax.nn.softmax(attention_scores_max / temperature, axis=2)  # (batch_size, action_horizon, 256, 1)
            weighted_tokens_max = attention_weights_max * image_tokens  # (batch_size, action_horizon, 256, 2048)
            max_pool_tokens = jnp.max(weighted_tokens_max, axis=2)  # (batch_size, action_horizon, 1, 2048)

            # attention_weights by sum pooling
            attention_scores_sum = image_tokens.sum(axis=-1, keepdims=True) # (batch_size, action_horzion, 256, 1)
            print("attention_scores_sum.shape", attention_scores_sum.shape) 
            attention_weights_sum = jax.nn.softmax(attention_scores_sum, axis=2) #  (batch_size, action_horzion, 256, 1)
            weighted_tokens_sum = attention_weights_sum * image_tokens # (batch_size, action_horzion, 256, 2048)
            sum_pool_tokens = jnp.sum(weighted_tokens_sum, axis=2) # (batch_size, action_horzion, 1, 2048)
            
            # attention_weights by attn pooling
            attention_scores_learned = self.attentin_query(image_tokens) # (batch_size, action_horizon, 256, 1)
            attention_weights_learned = jax.nn.softmax(attention_scores_learned / temperature, axis=2)
            weighted_tokens_learned = attention_weights_learned * image_tokens
            learned_pool_tokens = jnp.sum(weighted_tokens_learned, axis=2)  # (batch_size, action_horizon, 1, 2048)

            # attention_weigths by mean pooling
            mean_pool_tokens = jnp.mean(image_tokens, axis=2) # (batch_size, action_horzion, 2048)
  
            all_pool_tokens = jnp.concatenate([max_pool_tokens, sum_pool_tokens, learned_pool_tokens, mean_pool_tokens], axis=-2) # (batch_size, action_horizon, 4, 2048)
            image_pool_tokens = all_pool_tokens.reshape(all_pool_tokens.shape[0], -1, all_pool_tokens.shape[-1]) # (batch_size, action_horizon*4, 2048)
            print("image_pool_tokens.shape", image_pool_tokens.shape)
            current_image_tokens = image_pool_tokens[:, :self.image_chunk]
            print("current_image_tokens.shape",  current_image_tokens.shape)
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

        # add language (aka tokenized inputs)
        if obs.tokenized_prompt is not None:
            tokenized_inputs = self.PaliGemma.llm(obs.tokenized_prompt, method="embed")
            print("tokenized_inputs.shape",  tokenized_inputs.shape)
            current_tokens.append(tokenized_inputs)
            input_mask.append(obs.tokenized_prompt_mask)
            # full attention between image and language inputs
            ar_mask += [False] * tokenized_inputs.shape[1]
        current_tokens = jnp.concatenate(current_tokens, axis=1)
        predict_tokens = jnp.concatenate(predict_tokens, axis=1)
        print("final_current_tokens.shape", current_tokens.shape)
        print("final_predict_tokens.shape", predict_tokens.shape)
        input_mask = jnp.concatenate(input_mask, axis=1)
        ar_mask = jnp.array(ar_mask)
        return current_tokens, input_mask, ar_mask, predict_tokens

    @at.typecheck
    def embed_suffix(
        self, obs: _model.Observation, noisy_actions: _model.Actions, timestep: at.Float[at.Array, " b"], rgb_predict_tokens: at.Float[at.Array, "..."], wrist_predict_tokens: at.Float[at.Array, "..."]
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
        print("nosie_actions.shape", noisy_actions.shape)
        print("rgb_predict_tokens.shape", rgb_predict_tokens.shape)
        print("wrist_predict_tokens", wrist_predict_tokens.shape)
        action_tokens = self.action_in_proj(noisy_actions)
        rgb_tokens = self.rgb_in_proj(rgb_predict_tokens)
        wrist_tokens = self.wrist_in_proj(wrist_predict_tokens)

        time_tokens = einops.repeat(time_emb, "b emb -> b s emb", s=self.action_horizon-1)
        time_tokens_images = einops.repeat(time_emb, "b emb -> b s emb", s=(self.action_horizon-1) * self.image_chunk)
        
        action_time_tokens = jnp.concatenate([action_tokens, time_tokens], axis=-1)
        action_time_tokens = self.action_time_mlp_in(action_time_tokens)
        action_time_tokens = nnx.swish(action_time_tokens)
        action_time_tokens = self.action_time_mlp_out(action_time_tokens)

        rgb_time_tokens = jnp.concatenate([rgb_tokens, time_tokens_images], axis=-1)
        rgb_time_tokens = self.rgb_time_mlp_in(rgb_time_tokens)
        rgb_time_tokens = nnx.swish(rgb_time_tokens)
        rgb_time_tokens = self.rgb_time_mlp_out(rgb_time_tokens)

        wrist_time_tokens = jnp.concatenate([wrist_tokens, time_tokens_images], axis=-1)
        wrist_time_tokens = self.wrist_time_mlp_in(wrist_time_tokens)
        wrist_time_tokens = nnx.swish(wrist_time_tokens)
        wrist_time_tokens = self.wrist_time_mlp_out(wrist_time_tokens)

        print("action_time_tokens", action_time_tokens.shape)
        print("rgb_time_tokens.shape", rgb_time_tokens.shape)
        print("wrist_time_tokens.shape", wrist_time_tokens.shape)

        tokens.append(action_time_tokens)
        tokens.append(rgb_time_tokens)
        tokens.append(wrist_time_tokens)

        input_mask.append(jnp.ones(action_time_tokens.shape[:2], dtype=jnp.bool_))
        input_mask.append(jnp.ones(rgb_time_tokens.shape[:2], dtype=jnp.bool_))
        input_mask.append(jnp.ones(wrist_time_tokens.shape[:2], dtype=jnp.bool_))
        
        # image/language/state inputs do not attend to action tokens
        ar_mask += [True] + ([False] * (self.action_horizon - 1 -1)) 
        ar_mask += [True]*self.image_chunk + ([False] * (self.action_horizon - 1 -1)*self.image_chunk)
        ar_mask += [True]*self.image_chunk + ([False] * (self.action_horizon - 1 -1)*self.image_chunk)
       
        tokens = jnp.concatenate(tokens, axis=1)
        print("final.tokens.shape", tokens.shape)
  
        input_mask = jnp.concatenate(input_mask, axis=1)
        print("final.input_mask.shape", input_mask.shape)
        ar_mask = jnp.array(ar_mask)
        
        return tokens, input_mask, ar_mask

    @override
    def compute_loss(
        self, rng: at.KeyArrayLike, observation: _model.Observation, actions: _model.Actions,  *, train: bool = False
    ) -> at.Float[at.Array, "*b ah"]:
        preprocess_rng, noise_rng, rgb_noise_rng, wrist_noise_rng, time_rng = jax.random.split(rng, 5)
        observation = _model.preprocess_observation(preprocess_rng, observation, train=train)

        actions = actions[:, :-1]
        print("actions1.shape", actions.shape)
        batch_shape = actions.shape[:-2]
        noise = jax.random.normal(noise_rng, actions.shape)
    
        time = jax.random.beta(time_rng, 1.5, 1, batch_shape) * 0.999 + 0.001
        time_expanded_action = time[..., None, None]
        time_expanded_image = time[..., None, None, None, None]
        print("time_expand_action.shape", time_expanded_action.shape)
        print("actions2.shape", actions.shape)

        x_t = time_expanded_action * noise + (1 - time_expanded_action) * actions
        print("x_t.shape", x_t.shape)
        u_t = noise - actions
        print("u_t.shape", u_t.shape)

        # one big forward pass of prefix + suffix at once
        for name in observation.images:
            if name == "base_0_rgb":
                rgb_noise = jax.random.normal(rgb_noise_rng, observation.images[name][:,1:].shape)
                rgb_original = observation.images[name]
                rgb_current = rgb_original[:,:1]
                print("rgb_current.shape", rgb_current.shape)
                rgb_pred_noise = time_expanded_image * rgb_noise + (1 - time_expanded_image) *  rgb_original[:, 1:]
                print("rgb_pred_noise.shape", rgb_pred_noise.shape)
                ur_t = rgb_pred_noise - rgb_original[:, 1:]
                print("ur_t.shape", ur_t.shape)
                rgb_seq_image = jax.numpy.concatenate([rgb_current, rgb_pred_noise], axis=1)
                print("rgb_seq_image.shape", rgb_seq_image)
                observation.images[name] = rgb_seq_image

            elif name == "left_wrist_0_rgb":
                wrist_noise = jax.random.normal(wrist_noise_rng, observation.images[name][:,1:].shape)
                wrist_original = observation.images[name]
                wrist_current = wrist_original[:, :1]
                print("wrist_current.shape", wrist_current.shape)
                wrist_pred_noise = time_expanded_image * wrist_noise + (1 - time_expanded_image) *  wrist_original[:, 1:]
                print("wrist_pred_noise.shape", wrist_pred_noise.shape)
                uw_t = wrist_pred_noise - wrist_original[:, 1:]
                print("uw_t.shape", uw_t.shape)
                wrist_seq_image = jax.numpy.concatenate([wrist_current, wrist_pred_noise], axis=1)
                print("wrist_seq_image.shape", wrist_seq_image.shape)
                observation.images[name] = wrist_seq_image

        prefix_tokens, prefix_mask, prefix_ar_mask, predict_tokens = self.embed_prefix(observation)
        
        rgb_pred = predict_tokens[:, : self.img_seq_horizon]
        wrist_pred = predict_tokens[:, self.img_seq_horizon: 2*self.img_seq_horizon]

        suffix_tokens, suffix_mask, suffix_ar_mask = self.embed_suffix(observation, x_t, time, rgb_pred, wrist_pred)
        input_mask = jnp.concatenate([prefix_mask, suffix_mask], axis=1)
        ar_mask = jnp.concatenate([prefix_ar_mask, suffix_ar_mask], axis=0)
        attn_mask = make_attn_mask(input_mask, ar_mask)
        positions = jnp.cumsum(input_mask, axis=1) - 1
        
        (prefix_out, suffix_out), _ = self.PaliGemma.llm([prefix_tokens, suffix_tokens], mask=attn_mask, positions=positions)

        print("suffix_out.shape", suffix_out.shape)
        v_t = self.action_out_proj(suffix_out[:, -2*self.img_seq_horizon-self.action_seq_horizon :-2*self.img_seq_horizon])
        
        vr_t = self.rgb_out_proj(suffix_out[:, -2*self.img_seq_horizon : -self.img_seq_horizon])
        vr_t = vr_t.reshape(vr_t.shape[0], self.action_seq_horizon, 2, 2, 2048)
        vr_t = self.simple_image_upsampler(vr_t)
        print("vr_t.shape", vr_t.shape)
        
        vw_t = self.wrist_out_proj(suffix_out[:, -self.img_seq_horizon :])
        vw_t = vw_t.reshape(vw_t.shape[0], self.action_seq_horizon, 2, 2, 2048)
        vw_t = self.simple_image_upsampler(vw_t)
        print("vw_t.shape", vw_t.shape)
        
        return jnp.mean(jnp.square(v_t - u_t), axis=-1), jnp.mean(jnp.square(ur_t - vr_t), axis=-1), jnp.mean(jnp.square(uw_t - vw_t), axis=-1)

    @override 
    def sample_actions(
        self,
        rng: at.KeyArrayLike,
        observation: _model.Observation,
        *,
        num_steps: int | at.Int[at.Array, ""] = 10,
    ) -> _model.Actions:
        preprocess_rng, noise_rng, rgb_noise_rng, wrist_noise_rng, time_rng = jax.random.split(rng, 5)
        observation = _model.preprocess_observation(None, observation, train=False)
        # note that we use the convention more common in diffusion literature, where t=1 is noise and t=0 is the target
        # distribution. yes, this is the opposite of the pi0 paper, and I'm sorry.
        dt = -1.0 / num_steps
        batch_size = observation.state.shape[0]
        noise = jax.random.normal(noise_rng, (batch_size, self.action_horizon, self.action_dim))

        # first fill KV cache with a forward pass of the prefix
        prefix_tokens, prefix_mask, prefix_ar_mask, predict_tokens = self.embed_prefix(observation)

        rgb_predict_tokens = predict_tokens[:, : self.img_seq_horizon]
        wrist_predict_tokens = predict_tokens[:, self.img_seq_horizon: 2*self.img_seq_horizon]

        rgb_noise = jax.random.normal(rgb_noise_rng, rgb_predict_tokens.shape)
        wrist_noise = jax.random.normal(wrist_noise_rng, wrist_predict_tokens.shape)
        
        prefix_tokens, prefix_mask, prefix_ar_mask, predict_tokens = self.embed_prefix(observation)
        prefix_attn_mask = make_attn_mask(prefix_mask, prefix_ar_mask)
        positions = jnp.cumsum(prefix_mask, axis=1) - 1
        _, kv_cache = self.PaliGemma.llm([prefix_tokens, None], mask=prefix_attn_mask, positions=positions)

        def step(carry):
            x_t, time, r_t, w_t = carry
            print("xt.shape", x_t.shape)
            print("rt.shape", r_t.shape)
            print("wt.shape", w_t.shape)
          
            suffix_tokens, suffix_mask, suffix_ar_mask = self.embed_suffix(
                observation, x_t, time, r_t, w_t
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
                [None, suffix_tokens], mask=full_attn_mask, positions=positions, kv_cache=kv_cache
            )
            assert prefix_out is None
            v_t = self.action_out_proj(suffix_out[:, -2*self.img_seq_horizon-self.action_seq_horizon :-2*self.img_seq_horizon])

            return x_t + dt * v_t, time + dt

        def cond(carry):
            x_t, time, r_t, w_t = carry
            return time >= -dt / 2

        x_0, _ = jax.lax.while_loop(cond, step, (noise, 1.0, rgb_noise, wrist_noise))
        return x_0
