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
from openpi.models.pytorch_lam_wrapper import get_or_create_lam

logger = logging.getLogger("openpi")


# ═══════════════════════════════════════════════════════════════════════════════
#  make_attn_mask
#
#  ar_mask[i] = False → token i 与前一个 token 同组，组内双向互看
#  ar_mask[i] = True  → token i 开启新的因果步，只能看 ≤ 自身步的 token
# ═══════════════════════════════════════════════════════════════════════════════
def make_attn_mask(
    input_mask: at.Bool[at.Array, "b s"],
    mask_ar:    at.Bool[at.Array, " s"],
) -> at.Bool[at.Array, "b s s"]:
    mask_ar = jnp.broadcast_to(mask_ar, input_mask.shape)
    cumsum  = jnp.cumsum(mask_ar, axis=1)
    causal  = cumsum[:, None, :] <= cumsum[:, :, None]     # (B, S, S)
    valid   = input_mask[:, None, :] * input_mask[:, :, None]
    return jnp.logical_and(causal, valid)


def posemb_sincos(
    pos:           at.Real[at.Array, " b"],
    embedding_dim: int,
    min_period:    float,
    max_period:    float,
) -> at.Float[at.Array, "b {embedding_dim}"]:
    if embedding_dim % 2 != 0:
        raise ValueError(f"embedding_dim ({embedding_dim}) must be divisible by 2")
    fraction = jnp.linspace(0.0, 1.0, embedding_dim // 2)
    period   = min_period * (max_period / min_period) ** fraction
    sinusoid = jnp.einsum(
        "i,j->ij", pos, 1.0 / period * 2 * jnp.pi,
        precision=jax.lax.Precision.HIGHEST,
    )
    return jnp.concatenate([jnp.sin(sinusoid), jnp.cos(sinusoid)], axis=-1)


# ═══════════════════════════════════════════════════════════════════════════════
#  Config
# ═══════════════════════════════════════════════════════════════════════════════
@dataclasses.dataclass(frozen=True)
class Pi0Config(_model.BaseModelConfig):
    dtype:                 str            = "bfloat16"
    paligemma_variant:     _gemma.Variant = "gemma_2b"
    action_expert_variant: _gemma.Variant = "gemma_300m"

    action_dim:     int = 32
    action_horizon: int = 6
    max_token_len:  int = 48

    latent_action_pred: bool      = False
    vq_remap:           bool|None = None
    unknown_index:      int|None  = None
    phy_lam_ckpt:       str|None  = None

    @property
    @override
    def model_type(self) -> _model.ModelType:
        return _model.ModelType.PI0

    @override
    def create(self, rng: at.KeyArrayLike) -> "Pi0":
        return Pi0(self, rngs=nnx.Rngs(rng))

    @override
    def inputs_spec(
        self, *, batch_size: int = 1
    ) -> tuple[_model.Observation, _model.Actions]:
        image_spec = jax.ShapeDtypeStruct(
            [batch_size, self.action_horizon, *_model.IMAGE_RESOLUTION, 3],
            jnp.float32,
        )
        image_mask_spec = jax.ShapeDtypeStruct(
            [batch_size, self.action_horizon], jnp.bool_
        )
        with at.disable_typechecking():
            observation_spec = _model.Observation(
                images={
                    "base_0_rgb":        image_spec,
                    "left_wrist_0_rgb":  image_spec,
                    "right_wrist_0_rgb": image_spec,
                },
                image_masks={
                    "base_0_rgb":        image_mask_spec,
                    "left_wrist_0_rgb":  image_mask_spec,
                    "right_wrist_0_rgb": image_mask_spec,
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
        gemma_filter         = nnx_utils.PathRegex(".*llm.*")
        action_expert_filter = nnx_utils.PathRegex(".*llm.*_1.*")

        if "lora" in self.paligemma_variant:
            filters.append(gemma_filter)
            if "lora" not in self.action_expert_variant:
                filters.append(nnx.Not(action_expert_filter))
            has_lora = True
        elif "lora" in self.action_expert_variant:
            filters.append(action_expert_filter)
            has_lora = True

        if has_lora:
            filters.append(nnx.Not(nnx_utils.PathRegex(".*lora.*")))

        if not filters:
            return nnx.Nothing
        return nnx.All(*filters)


# ═══════════════════════════════════════════════════════════════════════════════
#  Model
# ═══════════════════════════════════════════════════════════════════════════════
class Pi0(_model.BaseModel):
    """
    自回归动作生成模型

    Token 序列
    ─────────────────────────────────────────────────────────────────────
    prefix (双向 ar=False):
        [img_tokens(256×N_cam) | lang_tokens(L)]

    suffix (严格因果 ar=True, 每 token 独立步):
        [state(1) | rgb_0..rgb_{R-1}(R) | wrist_0..wrist_{W-1}(W)
         | a_0 | a_1 | ... | a_{A-1}]

    训练 teacher forcing:
        输入:  [state | rgb(R) | wrist(W) | a_0 .. a_{A-2}]   长度 = 1+R+W+A-1
        目标:  a_0 .. a_{A-1}                                  长度 = A
        对齐:
          suffix_out[:, R+W+0]   → action_out_proj → pred_a_0   ← target a_0
          suffix_out[:, R+W+k]   → action_out_proj → pred_a_k   ← target a_k
          suffix_out[:, R+W+A-1] → action_out_proj → pred_a_{A-1}

        注: pred_start = R+W 而非 1+R+W，因为 suffix_out 下标从 0 开始
            suffix[0]=state → suffix_out[0] 预测 suffix[1]=rgb_0
            suffix[R+W]=wrist[-1] → suffix_out[R+W] 预测 a_0  ✓

    推理 autoregressive:
        Step 0: encode prefix → kv_cache_0
        Step 1: encode [state|rgb|wrist] → kv_cache_1
                ctx_out[:, -1] → a_0
        Step k: encode [a_{k-1}] → kv_cache_k
                out → a_k
    ─────────────────────────────────────────────────────────────────────
    """

    def __init__(self, config: Pi0Config, rngs: nnx.Rngs):
        super().__init__(
            config.action_dim, config.action_horizon, config.max_token_len
        )
        paligemma_config     = _gemma.get_config(config.paligemma_variant)
        action_expert_config = _gemma.get_config(config.action_expert_variant)
        expert_width         = action_expert_config.width   # 1024

        # ── PaliGemma ────────────────────────────────────────────────────────
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
        img.lazy_init(
            next(iter(config.fake_obs().images.values())), train=False, rngs=rngs
        )
        self.PaliGemma = nnx.Dict(llm=llm, img=img)

        # ── LAM ──────────────────────────────────────────────────────────────
        if config.latent_action_pred:
            phy_lam = get_or_create_lam(
                pretrained_path=config.phy_lam_ckpt, device="cuda"
            )
            if getattr(config, "vq_remap", None) is not None:
                phy_lam.setup_remap(config.vq_remap, config.unknown_index)
            lam_input_dim  = (
                phy_lam.config.action_num_codes * phy_lam.config.latent_dim
            )                                        # e.g. 4×128=512
            lam_output_dim = expert_width * 2        # 2048
            self.lam_proj  = nnx.Linear(
                lam_input_dim, lam_output_dim, rngs=rngs
            )
        else:
            phy_lam = None
        self.latent_action_tokenizer = phy_lam

        # ── 维度常量 ─────────────────────────────────────────────────────────
        self.compress_dim       = 2048
        self.image_chunk        = 1
        self.action_seq_horizon = self.action_horizon - 1
        self.img_seq_horizon    = self.action_seq_horizon * self.image_chunk

        # ── 投影层 ───────────────────────────────────────────────────────────
        self.state_proj      = nnx.Linear(config.action_dim,  expert_width, rngs=rngs)
        self.rgb_in_proj     = nnx.Linear(self.compress_dim,  expert_width, rngs=rngs)
        self.wrist_in_proj   = nnx.Linear(self.compress_dim,  expert_width, rngs=rngs)
        self.action_in_proj  = nnx.Linear(config.action_dim,  expert_width, rngs=rngs)
        self.action_out_proj = nnx.Linear(expert_width, config.action_dim, rngs=rngs)
        # 注: 自回归版本不需要 time_emb / flow-matching MLP，全部移除

    # ─────────────────────────────────────────────────────────────────────────
    # embed_prefix
    # ─────────────────────────────────────────────────────────────────────────
    @at.typecheck
    def embed_prefix(
        self,
        obs:     _model.Observation,
        lam_out: dict | None = None,
    ) -> tuple[
        at.Float[at.Array, "b p emb"],
        at.Bool[at.Array,  "b p"],
        at.Bool[at.Array,  " p"],
        at.Float[at.Array, "..."],   # predict_tokens: (B, R+W, 2048)
    ]:
        tokens         = []
        inp_mask       = []
        ar_mask        = []
        predict_tokens = []

        for name in obs.images:
            if name == "right_wrist_0_rgb":
                continue

            images         = obs.images[name]       # (B, T, H, W, C)
            B, T, H, W, C = images.shape

            # ── LAM latent ──────────────────────────────────────────────────
            if self.latent_action_tokenizer is not None and lam_out is not None:
                phys    = lam_out[name]["physical_latent_action"].squeeze(-1)
                w_seq   = phys.reshape(B, T - 1, -1).astype(jnp.bfloat16)
                lam_tok = self.lam_proj(w_seq)     # (B, T-1, 2048)
            else:
                lam_tok = jnp.zeros((B, 0, self.compress_dim), dtype=jnp.bfloat16)

            # ── 当前帧图像 token ─────────────────────────────────────────────
            cur_img_tok, _ = self.PaliGemma.img(
                obs.images[name][:, :1], train=False
            )
            cur_img_tok = cur_img_tok.squeeze(1)   # (B, 256, emb)

            tokens.append(cur_img_tok)
            predict_tokens.append(lam_tok)

            # image_masks[name]: (B, T) or (B,) — 取第一帧或直接使用
            img_mask = obs.image_masks[name]
            if img_mask.ndim == 2:
                img_mask = img_mask[:, 0]          # (B,) 取当前帧 mask
            inp_mask.append(
                einops.repeat(img_mask, "b -> b s", s=cur_img_tok.shape[1])
            )
            ar_mask += [False] * cur_img_tok.shape[1]   # prefix 内双向

        # ── 语言 token ───────────────────────────────────────────────────────
        if obs.tokenized_prompt is not None:
            lang_tok = self.PaliGemma.llm(obs.tokenized_prompt, method="embed")
            tokens.append(lang_tok)
            inp_mask.append(obs.tokenized_prompt_mask)
            ar_mask += [False] * lang_tok.shape[1]

        prefix_tokens  = jnp.concatenate(tokens,         axis=1)  # (B, P, emb)
        predict_tokens = jnp.concatenate(predict_tokens, axis=1)  # (B, R+W, 2048)
        prefix_mask    = jnp.concatenate(inp_mask,       axis=1)  # (B, P)
        prefix_ar_mask = jnp.array(ar_mask)                       # (P,)

        return prefix_tokens, prefix_mask, prefix_ar_mask, predict_tokens

    # ─────────────────────────────────────────────────────────────────────────
    # embed_suffix_ar
    #
    #  suffix 布局:
    #    [state(1) | rgb_0..rgb_{R-1}(R) | wrist_0..wrist_{W-1}(W) | actions(A')]
    #
    #  ar_mask: 全部 True（每个 token 独立因果步）
    #
    #  actions:
    #    训练时 = a_{0..A-2}  (B, A-1, action_dim)
    #    推理时 = 已生成动作   (B, k,   action_dim), k ∈ [0, A-1)
    #    None   = 只编码 context（推理阶段 0）
    # ─────────────────────────────────────────────────────────────────────────
    @at.typecheck
    def embed_suffix_ar(
        self,
        obs:                  _model.Observation,
        rgb_predict_tokens:   at.Float[at.Array, "b r d"],
        wrist_predict_tokens: at.Float[at.Array, "b w d"],
        actions:              at.Float[at.Array, "b a action_dim"] | None = None,
    ) -> tuple[
        at.Float[at.Array, "b s emb"],
        at.Bool[at.Array,  "b s"],
        at.Bool[at.Array,  " s"],
    ]:
        tokens   = []
        inp_mask = []
        ar_mask  = []

        # obs.state: (B, action_dim) 或 (B, T, action_dim)
        state = obs.state
        if state.ndim == 3:
            state = state[:, 0]                    # 取当前帧 (B, action_dim)
        B = state.shape[0]

        # ── 1. state token ───────────────────────────────────────────────────
        state_tok = self.state_proj(state)[:, None, :]   # (B, 1, emb)
        tokens.append(state_tok)
        inp_mask.append(jnp.ones((B, 1), dtype=jnp.bool_))
        ar_mask.append(True)

        # ── 2. rgb tokens ────────────────────────────────────────────────────
        rgb_emb = self.rgb_in_proj(rgb_predict_tokens)   # (B, R, emb)
        R = rgb_emb.shape[1]
        tokens.append(rgb_emb)
        inp_mask.append(jnp.ones((B, R), dtype=jnp.bool_))
        ar_mask += [True] * R

        # ── 3. wrist tokens ──────────────────────────────────────────────────
        wrist_emb = self.wrist_in_proj(wrist_predict_tokens)  # (B, W, emb)
        W = wrist_emb.shape[1]
        tokens.append(wrist_emb)
        inp_mask.append(jnp.ones((B, W), dtype=jnp.bool_))
        ar_mask += [True] * W

        # ── 4. action tokens（可选）─────────────────────────────────────────
        if actions is not None:
            action_emb = self.action_in_proj(actions)    # (B, A', emb)
            A_in = action_emb.shape[1]
            tokens.append(action_emb)
            inp_mask.append(jnp.ones((B, A_in), dtype=jnp.bool_))
            ar_mask += [True] * A_in

        tokens   = jnp.concatenate(tokens,   axis=1)    # (B, S, emb)
        inp_mask = jnp.concatenate(inp_mask, axis=1)    # (B, S)
        ar_mask  = jnp.array(ar_mask)                   # (S,)

        return tokens, inp_mask, ar_mask

    # ─────────────────────────────────────────────────────────────────────────
    # compute_loss —— 自回归 teacher forcing + L1
    # ─────────────────────────────────────────────────────────────────────────
    @override
    def compute_loss(
        self,
        rng:         at.KeyArrayLike,
        observation: _model.Observation,
        actions:     _model.Actions,
        *,
        train:   bool      = False,
        lam_out: dict|None = None,
    ) -> at.Float[at.Array, "*b a"]:

        # ── fix 1: split 出两个 key，避免只用 1 个 ──────────────────────────
        preprocess_rng, _ = jax.random.split(rng, 2)
        observation = _model.preprocess_observation(
            preprocess_rng, observation, train=train
        )

        # actions: (B, T, action_dim)，取前 A 步为目标
        A           = self.action_seq_horizon
        actions_tgt = actions[:, :A]               # (B, A, action_dim)
        B           = actions_tgt.shape[0]

        # ── Prefix ───────────────────────────────────────────────────────────
        prefix_tokens, prefix_mask, prefix_ar_mask, predict_tokens = \
            self.embed_prefix(observation, lam_out=lam_out)

        R          = self.img_seq_horizon
        W          = self.img_seq_horizon
        rgb_pred   = predict_tokens[:, :R]           # (B, R, 2048)
        wrist_pred = predict_tokens[:, R: R + W]     # (B, W, 2048)

        # ── Suffix (teacher forcing) ──────────────────────────────────────────
        # 输入 a_0..a_{A-2}，模型在每个位置预测下一个 action
        actions_in = actions_tgt[:, :-1]             # (B, A-1, action_dim)

        suffix_tokens, suffix_mask, suffix_ar_mask = self.embed_suffix_ar(
            observation, rgb_pred, wrist_pred, actions=actions_in
        )
        # suffix 布局:
        #   idx:  0      1..R    1+R..1+R+W    1+R+W..1+R+W+A-2
        #   tok:  state  rgb(R)  wrist(W)       actions_in(A-1)
        # 总长度: 1 + R + W + (A-1)

        # ── 拼接并构造 attention mask ────────────────────────────────────────
        all_mask = jnp.concatenate([prefix_mask,    suffix_mask],    axis=1)
        all_ar   = jnp.concatenate([prefix_ar_mask, suffix_ar_mask])
        attn_mask = make_attn_mask(all_mask, all_ar)   # (B, P+S, P+S)
        positions  = jnp.cumsum(all_mask, axis=1) - 1  # (B, P+S)

        # ── 前向 ─────────────────────────────────────────────────────────────
        (_, suffix_out), _ = self.PaliGemma.llm(
            [prefix_tokens, suffix_tokens],
            mask=attn_mask,
            positions=positions,
        )
        # suffix_out: (B, 1+R+W+A-1, emb)

        # ── 提取 action 预测输出 ──────────────────────────────────────────────
        #
        # suffix_out 下标对应关系（因果：位置 k 的输出预测位置 k+1 的 token）:
        #
        #   suffix_out[:, 0]         state 输出      → 预测 rgb_0     (不用)
        #   suffix_out[:, 1]         rgb_0 输出      → 预测 rgb_1     (不用)
        #   ...
        #   suffix_out[:, R]         rgb_{R-1} 输出  → 预测 wrist_0   (不用)
        #   suffix_out[:, R+1]       wrist_0 输出    → 预测 wrist_1   (不用)
        #   ...
        #   suffix_out[:, R+W]       wrist_{W-1} 输出 → 预测 a_0     ← pred_start
        #   suffix_out[:, R+W+1]     a_0 输出        → 预测 a_1
        #   ...
        #   suffix_out[:, R+W+A-1]   a_{A-2} 输出    → 预测 a_{A-1}
        #
        # slice: suffix_out[:, R+W : R+W+A]  共 A 个
        pred_start   = R + W
        pred_end     = R + W + A
        # 断言切片不越界
        assert suffix_out.shape[1] >= pred_end, (
            f"suffix_out 长度 {suffix_out.shape[1]} < pred_end {pred_end}，"
            f"请检查 R={R}, W={W}, A={A}"
        )
        action_out   = suffix_out[:, pred_start:pred_end]  # (B, A, emb)
        pred_actions = self.action_out_proj(action_out)    # (B, A, action_dim)

        # ── L1 loss ───────────────────────────────────────────────────────────
        loss  = jnp.mean(jnp.abs(pred_actions - actions_tgt), axis=-1)  # (B, A)
        zeros = jnp.zeros_like(loss)
        return loss, zeros, zeros

    # ─────────────────────────────────────────────────────────────────────────
    # sample_actions —— 自回归推理
    # ─────────────────────────────────────────────────────────────────────────
    @override
    def sample_actions(
        self,
        rng:         at.KeyArrayLike,
        observation: _model.Observation,
        *,
        num_steps: int = 10,
        ) -> _model.Actions:

        observation = _model.preprocess_observation(None, observation, train=False)
        B = observation.state.shape[0]
        A = self.action_seq_horizon   # 5
        R = self.img_seq_horizon      # 5
        W = self.img_seq_horizon      # 5

        # ── Prefix ────────────────────────────────────────────────────────
        prefix_tokens, prefix_mask, prefix_ar_mask, _ = \
            self.embed_prefix(observation)
        #                                          ↑ predict_tokens 不用

        # ── Suffix：与训练完全一致的结构，rgb/wrist 用 zeros 占位 ──────────
        suffix_tokens, suffix_mask, suffix_ar_mask = self.embed_suffix_ar(
            observation,
            rgb_predict_tokens   = jnp.zeros((B, R, self.compress_dim), dtype=jnp.bfloat16),
            wrist_predict_tokens     = jnp.zeros((B, W, self.compress_dim), dtype=jnp.bfloat16),
            actions    = jnp.zeros((B, A-1, self.action_dim), dtype=jnp.bfloat16),
        )

        # ── Attention mask ─────────────────────────────────────────────────
        all_mask  = jnp.concatenate([prefix_mask,    suffix_mask],    axis=1)
        all_ar    = jnp.concatenate([prefix_ar_mask, suffix_ar_mask])
        attn_mask = make_attn_mask(all_mask, all_ar)
        positions = jnp.cumsum(all_mask, axis=1) - 1

        # ── 单次前向 ────────────────────────────────────────────────────────
        (_, suffix_out), _ = self.PaliGemma.llm(
            [prefix_tokens, suffix_tokens],
            mask=attn_mask,
            positions=positions,
        )
        # suffix_out: (B, 1+R+W+(A-1), emb) = (B, 15, emb)

        # ── 只取 action 对应位置 ───────────────────────────────────────────
        # suffix 结构: [state(1) | rgb(R) | wrist(W) | action_in(A-1)]
        # 输出对应:
        #   pos R+W   (wrist最后) → pred a0   ← pred_start
        #   pos R+W+1 (a0位置)   → pred a1
        #   ...
        #   pos R+W+A-1          → pred a_{A-1}
        pred_start   = R + W              # = 10
        pred_end     = R + W + A          # = 15
        action_out   = suffix_out[:, pred_start:pred_end]   # (B, A, emb)
        pred_actions = self.action_out_proj(action_out)     # (B, A, action_dim)

        return pred_actions
