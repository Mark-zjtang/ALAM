#!/usr/bin/env python3
"""CPU-only structural and provenance validation for the public release."""

from __future__ import annotations

import argparse
import filecmp
import hashlib
import json
import os
import subprocess
from pathlib import Path
import sys

# This audit is commonly invoked directly from a clean source checkout. Avoid
# creating the very cache files that its hygiene gate is designed to reject.
sys.dont_write_bytecode = True

WORKFLOWS_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(WORKFLOWS_ROOT))

from common.paths import REPO_ROOT, repo_path, require_file


ORIGINAL_ALAM_ROOT = repo_path(os.environ.get("ALAM_INTERNAL_ALAM_SOURCE", "../ALAM"))
ORIGINAL_DOWNSTREAM_ROOT = repo_path(
    os.environ.get("ALAM_INTERNAL_DOWNSTREAM_SOURCE", "../physics_latent_world")
)
ALAM_CONFIG_SHA256 = "2f4b98ebafec559b9cb0d0b51eed5bcc88d1a23f3c51af821b58b8860016c653"

DOWNSTREAM_LEGACY_SOURCES = {
    "legacy_launchers/libero_all_replan_sweep_reference.sh": "run_multi_eval_reuslt.sh",
    "legacy_launchers/libero_goal_replan_sweep_0504.sh": "run_libero_goal_multi_tasks.sh",
    "legacy_launchers/libero_goal_replan_sweep_0506.sh": "run_libero_single_multi_tasks.sh",
    "legacy_launchers/libero_long_replan_sweep_0506.sh": "run_libero_10_multi_tasks.sh",
    "legacy_launchers/libero_object_replan_sweep_0504.sh": "run_libero_object_multi_tasks.sh",
    "legacy_launchers/metaworld_client_reference.sh": "run_local_client.sh",
    "legacy_launchers/misnamed_spatial_launcher_reference.sh": "run_libero_spatial_multi_tasks.sh",
    "legacy_launchers/policy_server_reference.sh": "run_local_server.sh",
}

DOWNSTREAM_EVIDENCE_SOURCES = {
    "evaluation_evidence/libero/record_experiments_results.txt": "record_experiments_results.txt",
    (
        "evaluation_evidence/libero/"
        "replan10_libero_object_phylam_train_20_infer_14_eval_30k_0504.log"
    ): (
        "checkpoints/pi0_libero_low_mem_finetune/"
        "replan10_libero_object_phylam_train_20_infer_14_eval_30k_0504.log"
    ),
    "evaluation_evidence/libero/replan12_libero_10.log": (
        "checkpoints/pi0_libero_low_mem_finetune/replan12_libero_10.log"
    ),
    (
        "evaluation_evidence/libero/"
        "replan5_libero_spatial_phylam_train_20_infer__eval_30k_0506_last.log"
    ): (
        "checkpoints/pi0_libero_low_mem_finetune/"
        "replan5_libero_spatial_phylam_train_20_infer__eval_30k_0506_last.log"
    ),
    "evaluation_evidence/libero/replan7_libero_goal.log": (
        "checkpoints/pi0_libero_low_mem_finetune/replan7_libero_goal.log"
    ),
    "evaluation_evidence/metaworld/mt50_cam_x_0.70.log": (
        "examples/metaworld/eval_results/"
        "phylam_gpu1_ah6_l1_loss_steps30001_0412_phy2_mix_11_larger_dataset_epoch19_step_58216_"
        "infer_ah5_rstep_5_0409_shell_new_eval_30k_cpos_x_0.7/"
        "pi0_wflow_policy_49_task_evaluate_logging_2026-04-13 11:29:50/"
        "phylam_gpu1_ah6_l1_loss_steps30001_0412_phy2_mix_11_larger_dataset_epoch19_step_58216_"
        "infer_ah5_rstep_5_0409_shell_new_eval_30k_cpos_x_0.7_episodes_10.log"
    ),
}

HF_REPOSITORIES = {
    "alam_pretrain": "Mark-ZJTang/alam_pretrain",
    "alam_plus_pi_libero": "Mark-ZJTang/alam_plus_pi_libero",
    "alam_plus_pi_metaworld_mt50": "Mark-ZJTang/alam_plus_pi_metaworld_mt50",
    "calvin_task_abc_d": "Mark-ZJTang/CALVIN_task_ABC_D",
    "oxe_mix10_datasets": "Mark-ZJTang/Oxe_mix10_datasets",
    "metaworld_mt50_lerobot": "Mark-ZJTang/metaworld_mt50",
    "libero_real_lerobot": "Mark-ZJTang/libero_real",
}

MODEL_LAYOUT = {
    "alam_metaworld": "evaluation/checkpoints/alam/metaworld_epoch19_step58216/pytorch_model.bin",
    "alam_libero": "evaluation/checkpoints/alam/libero_epoch16_step49024/pytorch_model.bin",
    "pi0_metaworld": "evaluation/checkpoints/metaworld/alam_plus_pi_metaworld_mt50_step30000/params/_METADATA",
    "pi0_libero": "evaluation/checkpoints/libero/alam_plus_pi_libero_step30000/params/_METADATA",
}


def load_hf_artifacts() -> dict[str, dict]:
    manifest_path = REPO_ROOT / "workflows" / "publishing" / "huggingface_manifest.json"
    artifacts = json.loads(manifest_path.read_text(encoding="utf-8"))["artifacts"]
    return {artifact["id"]: artifact for artifact in artifacts}


def check_public_paths() -> None:
    files = [REPO_ROOT / ".env.example", REPO_ROOT / "configs" / "libero" / "config.yaml"]
    for relative in (
        "common",
        "install",
        "alam_pretraining",
        "pi0_post_training",
        "metaworld_evaluation",
        "libero_evaluation",
        "tests",
    ):
        directory = REPO_ROOT / "workflows" / relative
        files.extend(path for path in directory.rglob("*") if path.suffix in {".py", ".sh"})
    files.extend((REPO_ROOT / "workflows" / "publishing").glob("*.py"))
    forbidden = ("/mnt/workspace/", "/home/tangzuojin", "/home/fortress")
    failures = []
    for path in files:
        if path.suffix in {".pyc", ".bin"}:
            continue
        text = path.read_text(encoding="utf-8")
        for marker in forbidden:
            if marker in text:
                failures.append(f"{path.relative_to(REPO_ROOT)} contains {marker}")
    if failures:
        raise AssertionError("Public runtime path audit failed:\n" + "\n".join(failures))


def check_public_structure() -> None:
    for removed in ("release", "release_tools", "uni_world_model"):
        if (REPO_ROOT / removed).exists():
            raise AssertionError(f"Removed public path unexpectedly exists: {removed}")
    required = (
        REPO_ROOT / "Algebraic_latent_action_model",
        REPO_ROOT
        / "evaluation"
        / "src"
        / "openpi"
        / "models"
        / "physics_lam_model"
        / "Algebraic_latent_action_model",
    )
    for path in required:
        require_directory(path)
    for path in (
        REPO_ROOT / "workflows" / "alam_pretraining" / "pretrain.sh",
        REPO_ROOT / "workflows" / "alam_pretraining" / "evaluate_checkpoint.sh",
        REPO_ROOT / "workflows" / "pi0_post_training" / "finetune_metaworld.sh",
        REPO_ROOT / "workflows" / "pi0_post_training" / "finetune_libero.sh",
        REPO_ROOT / "workflows" / "metaworld_evaluation" / "evaluate_mt50.sh",
        REPO_ROOT / "workflows" / "libero_evaluation" / "evaluate_libero.sh",
    ):
        require_file(path, "one-command public workflow")
    old_downstream = (
        REPO_ROOT / "evaluation" / "src" / "openpi" / "models" / "physics_lam_model" / "uni_world_model"
    )
    if old_downstream.exists():
        raise AssertionError(f"Old downstream package path unexpectedly exists: {old_downstream}")

    config_dir = REPO_ROOT / "configs" / "lam"
    configs = sorted(path.name for path in config_dir.iterdir() if path.is_file())
    if configs != ["alam_pretrain.yaml"]:
        raise AssertionError(f"Expected only configs/lam/alam_pretrain.yaml, found: {configs}")
    config_digest = hashlib.sha256((config_dir / "alam_pretrain.yaml").read_bytes()).hexdigest()
    if config_digest != ALAM_CONFIG_SHA256:
        raise AssertionError(f"alam_pretrain.yaml hash mismatch: {config_digest}")
    root_documents = sorted(path.name for path in REPO_ROOT.glob("*.md"))
    if root_documents != ["README.md"]:
        raise AssertionError(f"Expected one root document README.md, found: {root_documents}")
    if (REPO_ROOT / "docs").exists():
        raise AssertionError("Consolidated top-level docs/ directory unexpectedly exists")
    print("public stage/package/config structure: PASS")


def check_release_hygiene() -> None:
    generated: list[str] = []
    ignored_top_level = {
        ".cache",
        ".git",
        ".runtime",
        ".tools",
        ".venvs",
        "data",
        "hf_staging",
        "outputs",
    }
    for root, dirs, names in os.walk(REPO_ROOT, topdown=True, followlinks=False):
        root_path = Path(root)
        relative_root = root_path.relative_to(REPO_ROOT)
        if relative_root == Path("."):
            dirs[:] = [name for name in dirs if name not in ignored_top_level]
        if relative_root.parts[:2] == ("evaluation", "checkpoints"):
            dirs[:] = []
            continue
        file_names = set(names)
        for name in (*dirs, *names):
            path = root_path / name
            is_core_dump = name in file_names and (
                name == "core" or name.startswith("core-") or name.startswith("core.")
            )
            if (
                name == "__pycache__"
                or path.suffix in {".pyc", ".pyo"}
                or name == ".DS_Store"
                or is_core_dump
            ):
                generated.append(path.relative_to(REPO_ROOT).as_posix())
    if generated:
        raise AssertionError("Generated source-tree artifacts found:\n" + "\n".join(generated))
    print("release hygiene: PASS")


def check_huggingface_mapping() -> None:
    by_id = load_hf_artifacts()
    if len(by_id) != len(HF_REPOSITORIES):
        raise AssertionError("Duplicate Hugging Face artifact IDs")
    for artifact_id, expected_repo in HF_REPOSITORIES.items():
        artifact = by_id.get(artifact_id)
        if artifact is None or artifact.get("repo_id") != expected_repo:
            raise AssertionError(f"Hugging Face mapping mismatch for {artifact_id}")
        if artifact.get("repo_type") not in {"model", "dataset"}:
            raise AssertionError(f"Invalid Hugging Face repository type for {artifact_id}")
        if not artifact.get("download_path"):
            raise AssertionError(f"Missing download path for {artifact_id}")
    print(f"Hugging Face download mapping: PASS ({len(HF_REPOSITORIES)} repositories)")


def check_root_source_identity() -> None:
    if not ORIGINAL_ALAM_ROOT.is_dir():
        raise FileNotFoundError(f"Original ALAM source is not mounted: {ORIGINAL_ALAM_ROOT}")
    checked = 0
    for line in (REPO_ROOT / "SOURCE_MANIFEST.sha256").read_text(encoding="utf-8").splitlines():
        _, relative_value = line.split("  ", 1)
        relative = Path(relative_value.removeprefix("./"))
        if relative == Path("configs/lam/alam_pretrain.yaml"):
            source = ORIGINAL_ALAM_ROOT / "configs" / "lam" / "ctl_v3_lam_calvin_mix_data.yaml"
        elif relative.parts[0] == "Algebraic_latent_action_model":
            source = ORIGINAL_ALAM_ROOT / "uni_world_model" / Path(*relative.parts[1:])
        else:
            source = ORIGINAL_ALAM_ROOT / relative
        public = REPO_ROOT / relative
        if not source.is_file():
            raise FileNotFoundError(f"Missing original ALAM source: {source}")
        if not filecmp.cmp(public, source, shallow=False):
            raise AssertionError(f"ALAM source byte mismatch: {public} != {source}")
        checked += 1
    if checked != 25:
        raise AssertionError(f"Expected 25 byte-preserved ALAM files, compared {checked}")
    print(f"renamed ALAM source byte comparison: PASS ({checked}/25 files)")


def downstream_source_path(relative: Path) -> Path:
    value = relative.as_posix()
    if value in DOWNSTREAM_LEGACY_SOURCES:
        return ORIGINAL_DOWNSTREAM_ROOT / DOWNSTREAM_LEGACY_SOURCES[value]
    if value in DOWNSTREAM_EVIDENCE_SOURCES:
        return ORIGINAL_DOWNSTREAM_ROOT / DOWNSTREAM_EVIDENCE_SOURCES[value]
    prefix = "src/openpi/models/physics_lam_model/Algebraic_latent_action_model/"
    if value.startswith(prefix):
        suffix = value.removeprefix(prefix)
        return ORIGINAL_DOWNSTREAM_ROOT / "src/openpi/models/physics_lam_model/uni_world_model" / suffix
    return ORIGINAL_DOWNSTREAM_ROOT / relative


def check_downstream_source_identity() -> None:
    if not ORIGINAL_DOWNSTREAM_ROOT.is_dir():
        raise FileNotFoundError(f"Original downstream source is not mounted: {ORIGINAL_DOWNSTREAM_ROOT}")
    manifest = REPO_ROOT / "evaluation" / "SOURCE_MANIFEST.sha256"
    checked = 0
    camera_override_checked = False
    for line in manifest.read_text(encoding="utf-8").splitlines():
        _, relative_value = line.split("  ", 1)
        relative = Path(relative_value.removeprefix("./"))
        public = REPO_ROOT / "evaluation" / relative
        source = downstream_source_path(relative)
        if not source.is_file():
            raise FileNotFoundError(f"Missing original downstream source: {source}")
        if relative == Path("examples/metaworld/eval_metaworld_policy_client_0409.py"):
            original = source.read_bytes()
            old_default = b'parser.add_argument("--cam_pos_x", type=float, default=0.75)'
            new_default = b'parser.add_argument("--cam_pos_x", type=float, default=0.71)'
            if original.count(old_default) != 1 or original.replace(old_default, new_default) != public.read_bytes():
                raise AssertionError(f"MetaWorld camera default override differs from its audited one-line change: {public}")
            camera_override_checked = True
        elif not filecmp.cmp(public, source, shallow=False):
            raise AssertionError(f"Downstream source byte mismatch: {public} != {source}")
        checked += 1
    if checked != 1345 or not camera_override_checked:
        raise AssertionError(f"Expected 1,345 downstream files including one camera override, checked {checked}")
    print("downstream release source audit: PASS (1,345 files checked)")


def check_source_manifests() -> None:
    subprocess.run(["sha256sum", "--quiet", "-c", "SOURCE_MANIFEST.sha256"], cwd=REPO_ROOT, check=True)
    subprocess.run(
        ["sha256sum", "--quiet", "-c", "SOURCE_MANIFEST.sha256"],
        cwd=REPO_ROOT / "evaluation",
        check=True,
    )
    print("source manifests: PASS")


def check_evaluation_evidence() -> None:
    subprocess.run(
        [sys.executable, str(REPO_ROOT / "workflows" / "audit" / "verify_evaluation_evidence.py")],
        cwd=REPO_ROOT,
        check=True,
    )
    print("evaluation evidence recalculation: PASS")


def check_static_files() -> None:
    subprocess.run(
        [sys.executable, str(REPO_ROOT / "workflows" / "audit" / "static_audit.py")],
        cwd=REPO_ROOT,
        check=True,
    )


def check_model_layout() -> None:
    for name, relative in MODEL_LAYOUT.items():
        path = repo_path(relative)
        if not path.is_file():
            raise FileNotFoundError(f"{name}: {path}")
    for policy_dir in (
        repo_path("evaluation/checkpoints/metaworld/alam_plus_pi_metaworld_mt50_step30000"),
        repo_path("evaluation/checkpoints/libero/alam_plus_pi_libero_step30000"),
    ):
        if (policy_dir / "train_state").exists():
            raise AssertionError(f"Inference release unexpectedly contains train_state: {policy_dir}")


def check_model_layout_if_available(*, required: bool) -> None:
    present = [repo_path(relative).is_file() for relative in MODEL_LAYOUT.values()]
    if all(present):
        check_model_layout()
        print("packaged model layout: PASS")
        return
    if any(present):
        missing = [name for (name, _), exists in zip(MODEL_LAYOUT.items(), present, strict=True) if not exists]
        raise FileNotFoundError(f"Partial packaged model layout; missing: {', '.join(missing)}")
    if required:
        raise FileNotFoundError(
            "Packaged models are absent; download all three model repositories before checkpoint layout validation"
        )
    print("packaged model layout: SKIP (clean source clone; no model files downloaded)")


def require_directory(path: Path) -> Path:
    if not path.is_dir():
        raise FileNotFoundError(path)
    return path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--require-weights", action="store_true", help="Require downloaded checkpoint files and validate their layout (not file hashes)")
    parser.add_argument("--source-code", action="store_true", help="Byte-compare copied code/evidence to internal originals")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    print("[audit] public structure and repository mapping", flush=True)
    check_public_structure()
    check_huggingface_mapping()
    print("[audit] relative runtime paths and generated-file hygiene", flush=True)
    check_public_paths()
    check_release_hygiene()
    print("[audit] preserved source manifests and static syntax", flush=True)
    check_source_manifests()
    check_static_files()
    print("[audit] packaged model layout and evaluation evidence", flush=True)
    check_model_layout_if_available(required=args.require_weights)
    check_evaluation_evidence()
    if args.source_code:
        print("[audit] byte-compare public source with both read-only originals", flush=True)
        check_root_source_identity()
        check_downstream_source_identity()
    print("workflow validation: PASS")


if __name__ == "__main__":
    main()
