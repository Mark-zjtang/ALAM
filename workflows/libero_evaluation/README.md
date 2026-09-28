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

The [selected local results](../best_observed_evaluation/README.md) are historical observations, not guaranteed scores for fresh runs. The retained Spatial client log alone does not establish its server checkpoint identity.
