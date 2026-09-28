import torch
import torch.nn as nn
import numpy as np
import jax.numpy as jnp
import omegaconf
import hydra
import os


class _LAMForwardWrapper(nn.Module):
    """
    DataParallel 只能并行 forward()
    只返回需要的非标量张量，避免 scalar gather warning
    """
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, videos):
        lam_out = self.model.vq_encode(videos)

        # ★ 只返回张量，丢掉标量 commit_loss
        return (
            lam_out['physical_latent_action'],  # (B, 1, num_codes, latent_dim)
            lam_out['indices'],                  # (B, 1, num_codes, 1)
            lam_out['z_q'],                      # (B, 1, num_codes, latent_dim)
        )


class PyTorchLAMWrapper:
    def __init__(self, pretrained_path: str, device: str = "cuda"):
        config_path     = os.path.join(pretrained_path, "config.yaml")
        checkpoint_path = os.path.join(pretrained_path, "pytorch_model.bin")

        config = omegaconf.OmegaConf.load(config_path)
        model  = hydra.utils.instantiate(config)
        model.config = config

        missing_keys, unexpected_keys = model.load_state_dict(
            torch.load(checkpoint_path, map_location="cpu"),
            strict=False
        )
        missing_root_keys = set([k.split(".")[0] for k in missing_keys])
        print(f"load {checkpoint_path}")
        print(f"missing:    {missing_root_keys}")
        print(f"unexpected: {unexpected_keys}")

        model.eval()
        for p in model.parameters():
            p.requires_grad_(False)

        self.vector_quantizer = model.vector_quantizer
        self.config           = config

        wrapped = _LAMForwardWrapper(model)

        n_gpus = torch.cuda.device_count()
        if n_gpus > 1:
            print(f"[LAM] DataParallel 启动：{n_gpus} 张卡并行编码")
            self.model = nn.DataParallel(
                wrapped,
                device_ids=list(range(n_gpus))
            )
        else:
            print(f"[LAM] 单卡模式")
            self.model = wrapped

        self.model = self.model.to("cuda:0")
        self.device = "cuda:0"
        self.n_gpus = n_gpus

    def setup_remap(self, vq_remap, unknown_index: str = "closest"):
        base = self.model.module.model if self.n_gpus > 1 else self.model.model
        base.vector_quantizer.setup_remap(
            remap=vq_remap,
            unknown_index=unknown_index,
        )

    @torch.no_grad()
    def encode_latent_action(
        self,
        cond_pixel_values:   jnp.ndarray,
        target_pixel_values: jnp.ndarray,
    ) -> dict:

        cond_pt = _jax_to_torch(cond_pixel_values,   self.device)
        tgt_pt  = _jax_to_torch(target_pixel_values, self.device)

        videos = torch.stack([cond_pt, tgt_pt], dim=1)

        # ★ 接收元组（已去掉标量 commit_loss）
        physical_latent_action, indices, z_q = self.model(videos)

        return {
            'physical_latent_action': _torch_to_jax(physical_latent_action),
            'token_ids':              _torch_to_jax(indices),
            'z_q':                    _torch_to_jax(z_q),
        }


# ── 转换工具函数 ─────────────────────────────────────────

def _jax_to_torch(x: jnp.ndarray, device: str) -> torch.Tensor:
    arr = np.array(x)
    if arr.ndim == 4:
        arr = arr.transpose(0, 3, 1, 2)
    return torch.from_numpy(arr).to(device)


def _torch_to_jax(x: torch.Tensor) -> jnp.ndarray:
    arr = x.detach().cpu().numpy()
    if arr.ndim == 4:
        arr = arr.transpose(0, 2, 3, 1)
    return jnp.array(arr)


# ── 全局缓存 ─────────────────────────────────────────────

_LAM_REGISTRY: dict = {}

def get_or_create_lam(pretrained_path: str, device: str = "cuda"):
    print("pretrained_path", pretrained_path)
    key = (pretrained_path, device)
    if key not in _LAM_REGISTRY:
        _LAM_REGISTRY[key] = PyTorchLAMWrapper(pretrained_path, device)
    return _LAM_REGISTRY[key]


# ── encode 函数 ──────────────────────────────────────────

def _encode_lam_outside_jit(lam, observation) -> dict:
    results = {}

    for name, images in observation.images.items():
        if name == "right_wrist_0_rgb":
            continue

        images_np = np.array(images)
        B, T, H, W, C = images_np.shape

        cond_flat   = images_np[:, :-1].reshape(B*(T-1), H, W, C)
        target_flat = images_np[:, 1: ].reshape(B*(T-1), H, W, C)

        results[name] = lam.encode_latent_action(
            cond_pixel_values   = jnp.array(cond_flat),
            target_pixel_values = jnp.array(target_flat),
        )

    return results
