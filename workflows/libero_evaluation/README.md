# LIBERO evaluation

From the repository root, after [downloading the models](../publishing/README.md):

```bash
bash workflows/libero_evaluation/install.sh
bash workflows/libero_evaluation/evaluate_libero.sh spatial --gpu 0
bash workflows/libero_evaluation/evaluate_libero.sh object --gpu 0
bash workflows/libero_evaluation/evaluate_libero.sh goal --gpu 0
bash workflows/libero_evaluation/evaluate_libero.sh long --gpu 0
```

Each suite runs 10 tasks × 50 trials with the shared LIBERO policy. Use `--trials 1` for a short check, or `--dry-run` to inspect settings. Run the four commands separately; a sequential wrapper is also available as `bash workflows/libero_evaluation/evaluate_all_suites.sh --gpu 0`.

**Paper result (Table 2, success rate %):**

| Spatial | Object | Goal | Long | Average |
| ---: | ---: | ---: | ---: | ---: |
| 99.2 | 99.6 | 99.0 | 94.4 | 98.1 |

The paper reports 500 episodes per suite and one shared trained checkpoint. These are the [paper](https://arxiv.org/pdf/2605.10819) values, not guarantees for a fresh evaluation; individual runs can vary.
