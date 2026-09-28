# MetaWorld MT50 evaluation

From the repository root, after [downloading the models](../publishing/README.md):

```bash
bash workflows/metaworld_evaluation/install.sh
bash workflows/metaworld_evaluation/evaluate_mt50.sh --gpu 0
```

The default runs 50 tasks × 10 episodes with one policy server and one serial client. Add `--dry-run` to inspect the command.

**Paper result (Table 1, success rate %):**

| Easy | Medium | Hard | Very Hard | Average |
| ---: | ---: | ---: | ---: | ---: |
| 89.3 | 83.6 | 85.0 | 82.0 | 85.0 |

The paper's average is the macro-average of the four difficulty tiers, not the episode-weighted rate. A new 500-episode run may differ. See the [paper](https://arxiv.org/pdf/2605.10819) and the [release profile](RELEASE_PROFILE.md) for local-run settings and evidence.
