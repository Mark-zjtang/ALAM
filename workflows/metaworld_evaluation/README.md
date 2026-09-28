# MetaWorld MT50 evaluation

From the repository root, after [downloading the models](../publishing/README.md):

```bash
bash workflows/metaworld_evaluation/install.sh
bash workflows/metaworld_evaluation/evaluate_mt50.sh --gpu 0
```

The default runs 50 tasks × 10 episodes with one policy server and one serial client. Add `--dry-run` to inspect the command.

To evaluate your own ALAM + π0 checkpoint:

```bash
bash workflows/metaworld_evaluation/evaluate_mt50.sh --gpu 0 \
  --policy-checkpoint /path/to/metaworld_run/30000 \
  --alam-checkpoint /path/to/alam_checkpoint \
  --output-dir outputs/evaluation/my_metaworld_run
```

The policy directory must contain `params/` and `assets/metaworld_mt50/norm_stats.json`. The ALAM directory must contain `config.yaml` and `pytorch_model.bin`, matching the tokenizer used for policy training. Omit `--alam-checkpoint` to use the configured default tokenizer. CLI paths override `.env`; relative paths start at the repository root. Checkpoints must be compatible with this repository's MetaWorld ALAM + π0 configuration. Checkpoint options cannot be combined with `--client-only`.

**Paper result (Table 1, success rate %):**

| Easy | Medium | Hard | Very Hard | Average |
| ---: | ---: | ---: | ---: | ---: |
| 89.3 | 83.6 | 85.0 | 82.0 | 85.0 |

The paper's average is the macro-average of the four difficulty tiers, not the episode-weighted rate. A new 500-episode run may differ. See the [paper](https://arxiv.org/pdf/2605.10819) and the [release profile](RELEASE_PROFILE.md) for local-run settings and evidence.
