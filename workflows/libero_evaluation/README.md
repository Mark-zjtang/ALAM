# LIBERO evaluation

Both MetaWorld and LIBERO default to the shared ALAM tokenizer `metaworld_epoch19_step58216` (epoch 19, step 58,216). The directory name is retained for compatibility. Existing `.env` files should set both `ALAM_METAWORLD_TOKENIZER` and `ALAM_LIBERO_TOKENIZER` to `evaluation/checkpoints/alam/metaworld_epoch19_step58216`.

The project owner confirms that the shared epoch-19 configuration has passed LIBERO validation. Published benchmark scores remain the recorded results of their original evaluation runs.

From the repository root, after [downloading the models](../publishing/README.md):

```bash
bash workflows/libero_evaluation/install.sh
bash workflows/libero_evaluation/evaluate_libero.sh spatial --gpu 0
bash workflows/libero_evaluation/evaluate_libero.sh object --gpu 0
bash workflows/libero_evaluation/evaluate_libero.sh goal --gpu 0
bash workflows/libero_evaluation/evaluate_libero.sh long --gpu 0
```

Each suite runs 10 tasks × 50 trials with the shared LIBERO policy. Use `--trials 1` for a short check, or `--dry-run` to inspect settings. Run the four commands separately; a sequential wrapper is also available as `bash workflows/libero_evaluation/evaluate_all_suites.sh --gpu 0`.

To evaluate your own ALAM + π0 checkpoint on one suite:

```bash
bash workflows/libero_evaluation/evaluate_libero.sh spatial --gpu 0 \
  --policy-checkpoint /path/to/libero_run/30000 \
  --alam-checkpoint /path/to/alam_checkpoint \
  --output-dir outputs/evaluation/my_libero_run/spatial
```

Replace `spatial` with `object`, `goal`, or `long`. To run all four suites serially, use `bash workflows/libero_evaluation/evaluate_all_suites.sh` with the same options and no suite argument; it creates one subdirectory per suite under `--output-dir`.

The policy directory must contain `params/` and `assets/libero_real/norm_stats.json`. The ALAM directory must contain `config.yaml` and `pytorch_model.bin`, matching the tokenizer used for policy training. Omit `--alam-checkpoint` to use the configured default tokenizer. CLI paths override `.env`; relative paths start at the repository root. Checkpoints must be compatible with this repository's LIBERO ALAM + π0 configuration. Checkpoint options cannot be combined with `--client-only`.

**Paper result (Table 2, success rate %):**

| Spatial | Object | Goal | Long | Average |
| ---: | ---: | ---: | ---: | ---: |
| 99.2 | 99.6 | 99.0 | 94.4 | 98.1 |

The paper reports 500 episodes per suite and one shared trained checkpoint. These are the [paper](https://arxiv.org/pdf/2605.10819) values, not guarantees for a fresh evaluation; individual runs can vary.
