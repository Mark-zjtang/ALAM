# Best-observed evaluation

Run each benchmark separately, in the listed order, from the repository root:

| Benchmark | Command | Selected completed rerun |
| --- | --- | ---: |
| MetaWorld MT50 | `bash workflows/best_observed_evaluation/evaluate_metaworld.sh` | 432/500 (86.4% episode-weighted; 85.37% difficulty-macro) |
| LIBERO Spatial | `bash workflows/best_observed_evaluation/evaluate_libero_spatial.sh` | 495/500 |
| LIBERO Object | `bash workflows/best_observed_evaluation/evaluate_libero_object.sh` | 496/500 |
| LIBERO Goal | `bash workflows/best_observed_evaluation/evaluate_libero_goal.sh` | 493/500 |
| LIBERO Long | `bash workflows/best_observed_evaluation/evaluate_libero_long.sh` | 475/500 |

Append `--dry-run` to print the selected commands without starting an
evaluation. Each real run creates a new output directory under
`outputs/best_observed_evaluation/`; historical logs are never overwritten.
The four LIBERO scripts use the same verified policy/tokenizer weights. Goal
uses the independently evaluated 36-step inference method.
MetaWorld is pinned to the completed camera-x=0.71 run with a MuJoCo 3.6.0
client; its original result and protocol files are hash-checked.

These are five **separate completed local evaluations**, not a newly
remeasured joint result. Repeated runs may score differently. The scripts
preserve the recorded evaluation settings and check their source and
checkpoint inputs before running.

This is a local, auditable staging bundle, not yet a portable public release:
it needs the frozen sibling runtime, checkpoint directories and Python
environments. Redistribution rights and fresh-machine execution remain
separate release gates.
