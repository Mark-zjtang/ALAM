import dataclasses
import logging
from typing import Literal, TypeAlias, Self

import einops
import flax.nnx as nnx
import flax.nnx.bridge as nnx_bridge
import jax
import jax.numpy as jnp
from typing_extensions import override

from openpi.models import model_pi0 as _model
import openpi.models.gemma as _gemma
import openpi.models.siglip_pi0 as _siglip
from openpi.shared import array_typing as at
import openpi.shared.nnx_utils as nnx_utils
import functools
import numpy as np

logger = logging.getLogger("openpi")



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



from openpi import transforms
PrefixAttentionSchedule: TypeAlias = Literal["linear", "exp", "ones", "zeros"]

def realtime_action(
        policy,
        obs: dict,
        num_steps: int,
        prev_action_chunk: jax.Array,  # [batch, horizon, action_dim]
        inference_delay: int,
        prefix_attention_horizon: int,
        execution_horizon: int,
        prefix_attention_schedule: PrefixAttentionSchedule,
        max_guidance_weight: float,
    ) -> jax.Array:
        dt = -1.0 / num_steps
        _, rng = jax.random.split(policy._rng)
        model = policy.model

        prev_action_chunk = jnp.asarray(prev_action_chunk)[np.newaxis, ...]
        prev_action_chunk = transforms.pad_to_dim(prev_action_chunk, model.action_dim)

        # Transform
        inputs = jax.tree.map(lambda x: x, obs)
        inputs = policy._input_transform(inputs)
        # Make a batch and convert to jax.Array.
        inputs = jax.tree.map(lambda x: jnp.asarray(x)[np.newaxis, ...], inputs)
        observation = _model.Observation.from_dict(inputs)
        obs = _model.preprocess_observation(None, observation, train=False)

        # first fill KV cache with a forward pass of the prefix
        prefix_tokens, prefix_mask, prefix_ar_mask = model.embed_prefix(obs)
        prefix_attn_mask = make_attn_mask(prefix_mask, prefix_ar_mask)
        positions = jnp.cumsum(prefix_mask, axis=1) - 1
        _, kv_cache = model.PaliGemma.llm([prefix_tokens, None], mask=prefix_attn_mask, positions=positions)

        def step(carry, _):
            x_t, time = carry

            @functools.partial(jax.vmap, in_axes=(0, 0, 0, None))  # over batch
            def pinv_corrected_velocity(obs, x_t, y, t):
                def denoiser(x_t):
                    # Add batch dimension back to obs since it was removed by vmap
                    obs_batched = jax.tree_map(lambda x: x[None], obs)
                    v_t = model.step(obs_batched, x_t[None], t, prefix_mask, prefix_tokens, kv_cache)
                    # Remove batch dimension since we're inside vmap
                    v_t = v_t[0]
                    return x_t - v_t * t, v_t # TODO: Note this is different in original repo

                x_0, vjp_fun, v_t = jax.vjp(denoiser, x_t, has_aux=True)
                weights = get_prefix_weights(
                    inference_delay, prefix_attention_horizon, prefix_attention_horizon+execution_horizon, prefix_attention_schedule
                )
                error = (y - x_0) * weights[:, None]
                pinv_correction = vjp_fun(error)[0]
                inv_r2 = ((1-t)**2 + t**2) / (t**2)
                c = jnp.nan_to_num(t / (1 - t), posinf=max_guidance_weight)
                guidance_weight = jnp.minimum(c*inv_r2, max_guidance_weight)
                return v_t + guidance_weight * pinv_correction

            v_t = pinv_corrected_velocity(obs, x_t, prev_action_chunk, time)
            return (x_t + dt * v_t, time + dt), None

        noise = jax.random.normal(rng, shape=(1, 20, 32))
        (x_0, _), _ = jax.lax.scan(step, (noise, 1.0), length=num_steps)

        assert x_0.shape == (1, 20, 32), x_0.shape
        x_0 = x_0[0, :, :14]
        return x_0

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

PrefixAttentionSchedule: TypeAlias = Literal["linear", "exp", "ones", "zeros"]
def get_prefix_weights(start: int, end: int, total: int, schedule: "PrefixAttentionSchedule") -> jax.Array:
    """With start=2, end=6, total=10, the output will be:
    1  1  4/5 3/5 2/5 1/5 0  0  0  0
           ^              ^
         start           end
    `start` (inclusive) is where the chunk starts being allowed to change. `end` (exclusive) is where the chunk stops
    paying attention to the prefix. if start == 0, then the entire chunk is allowed to change. if end == total, then the
    entire prefix is attended to.

    `end` takes precedence over `start` in the sense that, if `end < start`, then `start` is pushed down to `end`. Thus,
    if `end` is 0, then the entire prefix will always be ignored.
    """
    start = jnp.minimum(start, end)
    if schedule == "ones":
        w = jnp.ones(total)
    elif schedule == "zeros":
        w = (jnp.arange(total) < start).astype(jnp.float32)
    elif schedule == "linear" or schedule == "exp":
        w = jnp.clip((start - 1 - jnp.arange(total)) / (end - start + 1) + 1, 0, 1)
        if schedule == "exp":
            w = w * jnp.expm1(w) / (jnp.e - 1)
    else:
        raise ValueError(f"Invalid schedule: {schedule}")
    return jnp.where(jnp.arange(total) >= end, 0, w)

@dataclasses.dataclass(frozen=True)
class Pi0Config(_model.BaseModelConfig):
    dtype: str = "bfloat16"
    paligemma_variant: _gemma.Variant = "gemma_2b"
    action_expert_variant: _gemma.Variant = "gemma_300m"

    # Set the model specific defaults.
    action_dim: int = 32
    action_horizon: int = 30  # default 40
    max_token_len: int = 48

    @property
    @override
    def model_type(self) -> _model.ModelType:
        return _model.ModelType.PI0

    @override
    def create(self, rng: at.KeyArrayLike) -> "Pi0":
        return Pi0(self, rngs=nnx.Rngs(rng))

    @override
    def inputs_spec(self, *, batch_size: int = 1) -> tuple[_model.Observation, _model.Actions]:
        image_spec = jax.ShapeDtypeStruct([batch_size, *_model.IMAGE_RESOLUTION, 3], jnp.float32)
        image_mask_spec = jax.ShapeDtypeStruct([batch_size], jnp.bool_)

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
                state=jax.ShapeDtypeStruct([batch_size, self.action_dim], jnp.float32),
                tokenized_prompt=jax.ShapeDtypeStruct([batch_size, self.max_token_len], jnp.int32),
                tokenized_prompt_mask=jax.ShapeDtypeStruct([batch_size, self.max_token_len], bool),
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

    @at.typecheck
    def embed_prefix(
        self, obs: _model.Observation
    ) -> tuple[at.Float[at.Array, "b s emb"], at.Bool[at.Array, "b s"], at.Bool[at.Array, " s"]]:
        input_mask = []
        ar_mask = []
        tokens = []
        # embed images
        for name in obs.images:
            image_tokens, _ = self.PaliGemma.img(obs.images[name], train=False)

            tokens.append(image_tokens)
            input_mask.append(
                einops.repeat(
                    obs.image_masks[name],
                    "b -> b s",
                    s=image_tokens.shape[1],
                )
            )
            # image tokens attend to each other
            ar_mask += [False] * image_tokens.shape[1]

        # add language (aka tokenized inputs)
        if obs.tokenized_prompt is not None:
            tokenized_inputs = self.PaliGemma.llm(obs.tokenized_prompt, method="embed")
            tokens.append(tokenized_inputs)
            input_mask.append(obs.tokenized_prompt_mask)
            # full attention between image and language inputs
            ar_mask += [False] * tokenized_inputs.shape[1]
        tokens = jnp.concatenate(tokens, axis=1)
        input_mask = jnp.concatenate(input_mask, axis=1)
        ar_mask = jnp.array(ar_mask)
        return tokens, input_mask, ar_mask

    @at.typecheck
    def embed_suffix(
        self, obs: _model.Observation, noisy_actions: _model.Actions, timestep: at.Float[at.Array, " b"]
    ) -> tuple[at.Float[at.Array, "b s emb"], at.Bool[at.Array, "b s"], at.Bool[at.Array, " s"]]:
        input_mask = []
        ar_mask = []
        tokens = []
        # add a single state token
        state_token = self.state_proj(obs.state)[:, None, :]
        tokens.append(state_token)
        input_mask.append(jnp.ones((obs.state.shape[0], 1), dtype=jnp.bool_))
        # image/language inputs do not attend to state or actions
        ar_mask += [True]

        # embed timestep using sine-cosine positional encoding with sensitivity in the range [0, 1]
        time_emb = posemb_sincos(timestep, self.action_in_proj.out_features, min_period=4e-3, max_period=4.0)
        # mix timestep + action information using an MLP
        action_tokens = self.action_in_proj(noisy_actions)
        time_tokens = einops.repeat(time_emb, "b emb -> b s emb", s=self.action_horizon)
        action_time_tokens = jnp.concatenate([action_tokens, time_tokens], axis=-1)
        action_time_tokens = self.action_time_mlp_in(action_time_tokens)
        action_time_tokens = nnx.swish(action_time_tokens)
        action_time_tokens = self.action_time_mlp_out(action_time_tokens)
        tokens.append(action_time_tokens)
        input_mask.append(jnp.ones(action_time_tokens.shape[:2], dtype=jnp.bool_))
        # image/language/state inputs do not attend to action tokens
        ar_mask += [True] + ([False] * (self.action_horizon - 1))
        tokens = jnp.concatenate(tokens, axis=1)
        input_mask = jnp.concatenate(input_mask, axis=1)
        ar_mask = jnp.array(ar_mask)
        return tokens, input_mask, ar_mask

    @override
    def compute_loss(
        self, rng: at.KeyArrayLike, observation: _model.Observation, actions: _model.Actions, *, train: bool = False
    ) -> at.Float[at.Array, "*b ah"]:
        preprocess_rng, noise_rng, time_rng = jax.random.split(rng, 3)
        observation = _model.preprocess_observation(preprocess_rng, observation, train=train)

        batch_shape = actions.shape[:-2]
        noise = jax.random.normal(noise_rng, actions.shape)
        time = jax.random.beta(time_rng, 1.5, 1, batch_shape) * 0.999 + 0.001
        time_expanded = time[..., None, None]
        x_t = time_expanded * noise + (1 - time_expanded) * actions
        u_t = noise - actions

        # one big forward pass of prefix + suffix at once
        prefix_tokens, prefix_mask, prefix_ar_mask = self.embed_prefix(observation)
        suffix_tokens, suffix_mask, suffix_ar_mask = self.embed_suffix(observation, x_t, time)
        input_mask = jnp.concatenate([prefix_mask, suffix_mask], axis=1)
        ar_mask = jnp.concatenate([prefix_ar_mask, suffix_ar_mask], axis=0)
        attn_mask = make_attn_mask(input_mask, ar_mask)
        positions = jnp.cumsum(input_mask, axis=1) - 1
        (prefix_out, suffix_out), _ = self.PaliGemma.llm(
            [prefix_tokens, suffix_tokens], mask=attn_mask, positions=positions
        )
        v_t = self.action_out_proj(suffix_out[:, -self.action_horizon :])

        return jnp.mean(jnp.square(v_t - u_t), axis=-1)

    @override
    def sample_actions(
        self,
        rng: at.KeyArrayLike,
        observation: _model.Observation,
        *,
        num_steps: int | at.Int[at.Array, ""] = 10,
    ) -> _model.Actions:
        observation = _model.preprocess_observation(None, observation, train=False)
        # note that we use the convention more common in diffusion literature, where t=1 is noise and t=0 is the target
        # distribution. yes, this is the opposite of the pi0 paper, and I'm sorry.
        dt = -1.0 / num_steps
        batch_size = observation.state.shape[0]
        noise = jax.random.normal(rng, (batch_size, self.action_horizon, self.action_dim))

        # first fill KV cache with a forward pass of the prefix
        prefix_tokens, prefix_mask, prefix_ar_mask = self.embed_prefix(observation)
        prefix_attn_mask = make_attn_mask(prefix_mask, prefix_ar_mask)
        positions = jnp.cumsum(prefix_mask, axis=1) - 1
        _, kv_cache = self.PaliGemma.llm([prefix_tokens, None], mask=prefix_attn_mask, positions=positions)

        def step(carry):
            x_t, time = carry
            suffix_tokens, suffix_mask, suffix_ar_mask = self.embed_suffix(
                observation, x_t, jnp.broadcast_to(time, batch_size)
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
            v_t = self.action_out_proj(suffix_out[:, -self.action_horizon :])

            return x_t + dt * v_t, time + dt

        def cond(carry):
            x_t, time = carry
            # robust to floating-point error
            return time >= -dt / 2

        x_0, _ = jax.lax.while_loop(cond, step, (noise, 1.0))
        return x_0
    from typing import Literal, TypeAlias, Self
    
    # def step(
    #     self,
    #     obs_batched,
    #     x_t,
    #     t,
    #     prefix_mask, 
    #     prefix_tokens, 
    #     kv_cache
    # ):
    #     suffix_tokens, suffix_mask, suffix_ar_mask = self.embed_suffix(
    #             obs_batched, x_t, jnp.broadcast_to(t, x_t.shape[0])
    #     )
    #     suffix_attn_mask = make_attn_mask(suffix_mask, suffix_ar_mask)
    #     prefix_attn_mask = einops.repeat(prefix_mask, "b p -> b s p", s=suffix_tokens.shape[1])
    #     full_attn_mask = jnp.concatenate([prefix_attn_mask, suffix_attn_mask], axis=-1)
    #     assert full_attn_mask.shape == (
    #                     full_attn_mask.shape[0],
    #                     suffix_tokens.shape[1],
    #                     prefix_tokens.shape[1] + suffix_tokens.shape[1],
    #                 )
    #     positions = jnp.sum(prefix_mask, axis=-1)[:, None] + jnp.cumsum(suffix_mask, axis=-1) - 1
    #     (prefix_out, suffix_out), _ = self.PaliGemma.llm(
    #                     [None, suffix_tokens], mask=full_attn_mask, positions=positions, kv_cache=kv_cache
    #                 )
    #     assert prefix_out is None
    #     v_t = self.action_out_proj(suffix_out[:, -self.action_horizon :])
    #     return v_t


    def sample_realtime_actions(
        self,
        rng: at.KeyArrayLike,
        observation: _model.Observation,
        prev_action_chunk: jax.Array,
        inference_delay: int,
        *,
        num_steps: int | at.Int[at.Array, ""] = 10,
        # inference_delay: int,
        # prefix_attention_horizon: int,
        # prefix_attention_schedule: PrefixAttentionSchedule,
        # max_guidance_weight: float,
    ) -> _model.Actions:
        #inference_delay = 5
        prefix_attention_horizon = 30
        prefix_attention_schedule = "exp"
        max_guidance_weight = 5
        observation = _model.preprocess_observation(None, observation, train=False)
        # note that we use the convention more common in diffusion literature, where t=1 is noise and t=0 is the target
        # distribution. yes, this is the opposite of the pi0 paper, and I'm sorry.
        dt = -1.0 / num_steps
        batch_size = observation.state.shape[0]
        noise = jax.random.normal(rng, (batch_size, self.action_horizon, self.action_dim))

        # first fill KV cache with a forward pass of the prefix
        prefix_tokens, prefix_mask, prefix_ar_mask = self.embed_prefix(observation)
        prefix_attn_mask = make_attn_mask(prefix_mask, prefix_ar_mask)
        positions = jnp.cumsum(prefix_mask, axis=1) - 1
        _, kv_cache = self.PaliGemma.llm([prefix_tokens, None], mask=prefix_attn_mask, positions=positions)

        def step_real(carry, _):
            x_t, time = carry
            #print(time)
            #@functools.partial(jax.vmap, in_axes=(0, 0, 0, None))  # over batch
            def pinv_corrected_velocity(observation, x_t, y, t):
                def denoiser(x_t):
                    #obs_batched = jax.tree_map(lambda x: x[None], observation)

                    suffix_tokens, suffix_mask, suffix_ar_mask = self.embed_suffix(
                        observation, x_t, jnp.broadcast_to(time, batch_size)
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
                    v_t = self.action_out_proj(suffix_out[:, -self.action_horizon :])
                    #v_t = self(obs[None], x_t[None], t)[0]
                    return x_t - v_t * t, v_t######change (1-t) to t, + to - ##############
                x_0, vjp_fun, v_t = jax.vjp(denoiser, x_t, has_aux=True)
                weights = get_prefix_weights(
                    inference_delay, prefix_attention_horizon, 50, prefix_attention_schedule
                )
                error = (y - x_0) * weights[:, None]
                #print((y - x_0).shape)
                #print(weights[:, None])
                pinv_correction = vjp_fun(error)[0]
                # print(error.shape)
                # print(pinv_correction.shape)
                # constants from paper
                inv_r2 = ((1-t)**2 + t ** 2) / (t ** 2)######change (1-t) to t ##############
                c = jnp.nan_to_num(t / (1-t), posinf=max_guidance_weight)######change (1-t) to t ##############
                #inv_r2 = jnp.nan_to_num((t**2 + (1 - t) ** 2) / ((1 - t) ** 2), posinf=max_guidance_weight)
                #c = jnp.nan_to_num((1 - t) / t, posinf=max_guidance_weight)
                guidance_weight = jnp.minimum(c * inv_r2, max_guidance_weight)
                return v_t - guidance_weight * pinv_correction##### change + to - ###############

            v_t = pinv_corrected_velocity(observation, x_t, prev_action_chunk, time)
            return (x_t + dt * v_t, time + dt), None

        def cond(carry):
            x_t, time = carry
            # robust to floating-point error
            return time >= -dt / 2

        #x_0, _ = jax.lax.while_loop(cond, step_real, (noise, 1.0))
        (x_0, _), _ = jax.lax.scan(step_real, (noise, 1.0), length=num_steps)
        return x_0
