# MetaWorld MT50 open-source release profile

The public default is `oss_release_x071`: camera `corner2` at absolute MuJoCo
world position `(0.71, 0.075, 0.70)`. The machine-readable contract is
[`release_profile_x071.json`](release_profile_x071.json).

## Run

```bash
bash workflows/metaworld_evaluation/install.sh
XLA_PYTHON_CLIENT_MEM_FRACTION=0.4 \
  bash workflows/metaworld_evaluation/evaluate_mt50.sh --gpu 0 --port 8000
```

The launcher defaults to 50 tasks, 10 episodes per task, 200 steps, raw/effective
action horizon 6/5, and chunk/replan 5. It starts one dedicated policy server
and one serial client. `--camera-x`, `--camera-y`, and `--camera-z` provide
explicit overrides. Each non-dry run writes `client_runtime.json` and
`server_runtime.json` beside the evaluator output, including installed package
versions, the complete protocol, and SHA-256 hashes of the executed release
sources.

The validated client used CPython 3.11.16, MetaWorld 2.0.0, MuJoCo 3.6.0,
Gymnasium 1.2.2, NumPy 1.26.4, Torch 2.2.0+cu121, and WebSockets 15.0.1. The
server used JAX/JAXLIB 0.5.3, Flax 0.10.2, Orbax Checkpoint 0.11.13, NumPy
1.26.4, Torch 2.7.0, and WebSockets 15.0.1. The CUDA suffix of the client Torch
wheel is platform-specific; simulator actions are serialized from CPU arrays.

## Code boundary

| Role | Public file | Integrity |
| --- | --- | --- |
| MetaWorld evaluator | `evaluation/examples/metaworld/eval_metaworld_policy_client_0409.py` | SHA-256 `fea35805...a1d1` |
| Preserved policy RNG | `evaluation/src/openpi/policies/policy.py` | SHA-256 `88513163...67a7` |
| Preserved model | `evaluation/src/openpi/models/physics_va_flow.py` | SHA-256 `aecb982a...0d92` |
| Portable server adapter | `workflows/pi0_post_training/serve_policy.py` | records checkpoint and effective/raw horizon at runtime |
| Portable client adapter | `workflows/pi0_post_training/run_evaluation_client.py` | disables the WebSockets 15 client keepalive ping during blocking first JAX compilation |
| Public launcher | `workflows/metaworld_evaluation/evaluate_mt50.sh` | delegates to the portable launcher and records both runtimes |

The public evaluation profile uses camera x=`0.71`. The packaged evaluator is
covered by `evaluation/SOURCE_MANIFEST.sha256`; paths, transport handling, and
runtime recording live in the workflow layer.

The 2026-09-02 observation used an isolated reconstructed server entry and
configuration overlay rather than invoking the public adapter filename itself.
Their SHA-256 values (`43a8a0e3...d10b`, `6371e4eb...acd5`, and overlay-tree
`2994c6b2...e9c7`) are frozen in the JSON contract. The public adapter declares
the same checkpoint, ALAM epoch, raw/effective horizon, and RNG initialization;
it is a portable equivalent, not a claim that the wrapper files are
byte-identical. The proven client transport-only overlay SHA-256 is
`a43a4745...9e9b`, and its `ping_interval=None` behavior is implemented by the
public process-local adapter.

## Recorded result

The completed 2026-09-02 observation at x=0.71 was 432/500 (86.4%
episode-weighted), with 85.366883% macro success across Easy, Medium, Hard, and
Very Hard. Its raw evaluator log SHA-256 is recorded in the JSON contract.

The evaluator constructs `metaworld.MT50()` before applying its seed; the
task variants are not fixed by the later `env.reset(seed=...)` call. New runs
are fresh evaluations and their percentages may differ from this observation.
