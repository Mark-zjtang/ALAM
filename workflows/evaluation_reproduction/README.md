# Evaluation environment alignment

This guide describes the tested environment for evaluating the released
step-30,000 MetaWorld and LIBERO policies with the shared
`alam_pretrain_latent_action_tokenizer`. All four LIBERO suites use the same
LIBERO policy.

## Tested versions

| Component | Policy server | MetaWorld client | LIBERO client |
| --- | --- | --- | --- |
| Python | 3.11.16 | 3.11.16 | 3.8.20 |
| Torch | 2.7.0+cu128 | 2.2.0+cu121 | 1.11.0+cu113 |
| torchvision | 0.22.0+cu128 | — | 0.12.0+cu113 |
| NumPy | 1.26.4 | 1.26.4 | 1.22.4 |
| JAX / JAXLIB | 0.5.3 / 0.5.3 | — | — |
| Flax / Orbax Checkpoint | 0.10.2 / 0.11.13 | — | — |
| Simulator | — | MetaWorld 2.0.0 | robosuite 1.4.1 |
| Simulator MuJoCo | — | 3.6.0 | 3.2.3 |
| Gym API | — | Gymnasium 1.2.2 | gym 0.25.2 |
| WebSockets | 15.0.1 | 15.0.1 | 13.1 |

Host: one NVIDIA B200 with 183359 MiB memory, driver 595.71.05, Ubuntu 22.04.5
LTS, kernel 6.8.0-85-generic. The server's Torch CUDA 12.8 build supports this
GPU; it is an explicit adjustment to the lock-based server installation.
The MetaWorld client's Torch wheel retains its recorded CUDA 12.1 build,
but the evaluation client uses CPU actions. These wheel suffixes are not
instructions to use one global CUDA environment for all three roles.

The server also has MuJoCo 2.3.7 as an installed dependency; simulation runs
in the separate client environments, using the versions in the table.

Full installed package-version inventories are provided for
[pi0 (218 packages)](environment/pi0.json),
[MetaWorld (58 packages)](environment/metaworld.json), and
[LIBERO (127 packages)](environment/libero.json).
These inventories include the Python patch version and record the tested
installation; they are not portable lockfiles or a claim that the historical
machine's full system image was identical.

## Align a fresh installation

Run from the repository root with `uv` available on `PATH`. Use a fresh
environment directory: the installer reuses existing environments and does
not change an existing interpreter to a requested patch version.

```bash
set -e
uv python install 3.11.16 3.8.20
uv venv --python 3.11.16 .venvs-reproduction/pi0
uv venv --python 3.11.16 .venvs-reproduction/metaworld
uv venv --python 3.8.20 .venvs-reproduction/libero

bash workflows/install/install_environments.sh \
  --components pi0,metaworld,libero --env-root .venvs-reproduction

# B200 server adjustment, applied after the official lock-based installation.
uv pip install --python .venvs-reproduction/pi0/bin/python \
  --extra-index-url https://download.pytorch.org/whl/cu128 \
  --index-strategy unsafe-best-match \
  'torch==2.7.0+cu128' 'torchvision==0.22.0+cu128'

# Match the recorded MetaWorld client wheel; evaluation actions remain on CPU.
uv pip install --python .venvs-reproduction/metaworld/bin/python \
  --extra-index-url https://download.pytorch.org/whl/cu121 \
  --index-strategy unsafe-best-match 'torch==2.2.0+cu121'

for role in pi0 metaworld libero; do
  uv pip check --python ".venvs-reproduction/$role/bin/python"
  python3 workflows/install/check_installed_files.py \
    --python ".venvs-reproduction/$role/bin/python"
done
```

Use the inventories to check the installed versions before starting a run.
The following check reports differences instead of silently changing packages;
a fresh dependency resolution may differ from the recorded installation.

```bash
set -e
for role in pi0 metaworld libero; do
  ".venvs-reproduction/$role/bin/python" - "$role" <<'PY'
import importlib.metadata as metadata
import json
from pathlib import Path
import platform
import sys

role = sys.argv[1]
reference = json.loads(Path(
    f"workflows/evaluation_reproduction/environment/{role}.json"
).read_text())
actual = {d.metadata["Name"]: d.version for d in metadata.distributions()
          if d.metadata["Name"]}
differences = {name: {"expected": reference["packages"].get(name),
                      "actual": actual.get(name)}
               for name in sorted(set(reference["packages"]) | set(actual))
               if reference["packages"].get(name) != actual.get(name)}
assert platform.python_version() == reference["python"], platform.python_version()
assert not differences, json.dumps(differences, indent=2)
print(role, platform.python_version(), len(actual), "packages: match")
PY
done

# Confirm that both server frameworks can execute on the GPU.
.venvs-reproduction/pi0/bin/python - <<'PY'
import torch
assert torch.cuda.is_available()
print(torch.ones(1, device="cuda").cpu())
PY
.venvs-reproduction/pi0/bin/python - <<'PY'
import jax
import jax.numpy as jnp
assert jax.default_backend() == "gpu", jax.devices()
print(jnp.ones(1).block_until_ready())
PY
```

The generic installers request Python 3.11/3.8, not these exact patch versions.
Re-running the policy installer can restore its locked Torch build, so apply
the B200 adjustment and repeat the version/GPU checks after reinstalling.

## Run the evaluation

First [download the released models](../publishing/README.md), then select the
aligned interpreters. Use a distinct output directory for each repeat so that
existing logs are retained.

```bash
set -e
export PI0_PYTHON="$PWD/.venvs-reproduction/pi0/bin/python"
export METAWORLD_PYTHON="$PWD/.venvs-reproduction/metaworld/bin/python"
export LIBERO_PYTHON="$PWD/.venvs-reproduction/libero/bin/python"
export ALAM_OUTPUT_ROOT="$PWD/outputs/aligned_run_1"

unset XLA_PYTHON_CLIENT_PREALLOCATE XLA_PYTHON_CLIENT_MEM_FRACTION
unset XLA_CLIENT_MEM_FRACTION XLA_PYTHON_CLIENT_ALLOCATOR
unset OMP_NUM_THREADS OPENBLAS_NUM_THREADS

XLA_PYTHON_CLIENT_MEM_FRACTION=0.4 \
  bash workflows/metaworld_evaluation/evaluate_mt50.sh --gpu 0 --port 8000
bash workflows/libero_evaluation/evaluate_all_suites.sh --gpu 0 --port 8000
```

Run serially: one policy server and one simulator client at a time. The four
LIBERO suites run sequentially and restart the same policy on port 8000.
MetaWorld uses seed 10, 50 tasks × 10 episodes, 200 steps, camera `corner2` at
`(0.71, 0.075, 0.70)`, raw/effective horizon 6/5, and replan 5. LIBERO uses seed
7 and 10 tasks × 50 episodes per suite; Spatial/Object effective horizon 14,
Goal/Long 18, and replan 5/10/7/12 respectively.

## Reproducibility notes

Matching Python/package versions and configured seeds does not guarantee
identical episode outcomes. MetaWorld constructs task variants before
evaluator seeding, so matching task names and episode positions does not
ensure the same physical scene. Same-seed repeats are not an independent
multi-seed study.
