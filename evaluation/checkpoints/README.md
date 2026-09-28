# Checkpoint download layout

Model binaries are released on Hugging Face and intentionally excluded from the
GitHub repository.  After downloading, the following relative paths must exist:

```text
evaluation/checkpoints/
├── alam/
│   ├── metaworld_epoch19_step58216/{config.yaml,pytorch_model.bin}
│   └── libero_epoch16_step49024/{config.yaml,pytorch_model.bin}
├── metaworld/alam_plus_pi_metaworld_mt50_step30000/
│   ├── _CHECKPOINT_METADATA
│   ├── assets/metaworld_mt50/norm_stats.json
│   └── params/...
└── libero/alam_plus_pi_libero_step30000/
    ├── _CHECKPOINT_METADATA
    ├── assets/libero_real/norm_stats.json
    └── params/...
```

From the repository root, run `bash workflows/publishing/download_release.sh --models-only`
to download the three model repositories in `workflows/publishing/huggingface_manifest.json`.
