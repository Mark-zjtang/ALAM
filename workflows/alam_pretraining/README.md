# ALAM pretraining

Prepare CALVIN and converted Open X-Embodiment data, then set `ALAM_CALVIN_ROOT` and `ALAM_OXE_VIDEO_ROOT` in the root `.env`.

From the repository root:

```bash
bash workflows/alam_pretraining/install.sh
bash workflows/alam_pretraining/pretrain.sh
```

`configs/lam/alam_pretrain.yaml` is the public training config. Use `pretrain.sh --dry-run` to inspect resolved paths. The full-scale training setup used 128 H20 GPUs; multi-node launches set `ALAM_NUM_MACHINES`, total `ALAM_NUM_PROCESSES`, `MACHINE_RANK`, `MAIN_PROCESS_IP`, and `MAIN_PROCESS_PORT`.

To check a downloaded tokenizer or run CALVIN validation:

```bash
bash workflows/alam_pretraining/evaluate_checkpoint.sh cpu
bash workflows/alam_pretraining/evaluate_validation.sh --gpu 0 --samples 10
```

Use these workflows rather than invoking the preserved `train_lam.py` directly.
