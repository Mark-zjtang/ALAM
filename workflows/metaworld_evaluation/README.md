# MetaWorld MT50 evaluation

From the repository root, after [downloading the models](../publishing/README.md):

```bash
bash workflows/metaworld_evaluation/install.sh
bash workflows/metaworld_evaluation/evaluate_mt50.sh --gpu 0
```

The default runs 50 tasks × 10 episodes with one policy server and one serial client. Add `--dry-run` to inspect the command. The completed release-profile observation was **432/500** (86.4% episode-weighted; 85.37% difficulty-macro) with MuJoCo 3.6.0. A new run may differ.

Exact settings and evidence: [release profile](RELEASE_PROFILE.md). The separate [best-observed launcher](../best_observed_evaluation/README.md) requires frozen historical local assets.
