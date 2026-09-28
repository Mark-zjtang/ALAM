"""Path-only adapter for the preserved ALAM v3 implementation.

The inherited model and forward computation are unchanged.  The adapter prevents
the preserved constructor from retaining internal cache and mirror settings
while LPIPS loads its VGG weights.
"""

import os

from uni_world_model.latent_action_model.models import ctl_latent_action_tokenizer_v3 as _preserved


class LatentActionTokenizer(_preserved.LatentActionTokenizer):
    """ALAM v3 with a caller-controlled model cache path."""

    def __init__(self, *args, **kwargs):
        portable_torch_home = os.environ.get("ALAM_TORCH_HOME") or os.environ.get("TORCH_HOME") or ".cache/torch"
        caller_hf_endpoint = os.environ.get("HF_ENDPOINT")
        original_lpips = _preserved.lpips.LPIPS

        def load_lpips_from_portable_cache(*lpips_args, **lpips_kwargs):
            os.environ["TORCH_HOME"] = portable_torch_home
            return original_lpips(*lpips_args, **lpips_kwargs)

        _preserved.lpips.LPIPS = load_lpips_from_portable_cache
        try:
            super().__init__(*args, **kwargs)
        finally:
            _preserved.lpips.LPIPS = original_lpips
            os.environ["TORCH_HOME"] = portable_torch_home
            if caller_hf_endpoint is None:
                os.environ.pop("HF_ENDPOINT", None)
            else:
                os.environ["HF_ENDPOINT"] = caller_hf_endpoint
