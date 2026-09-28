#!/usr/bin/env python3
"""Parse release source/config files and validate repository-local symlinks."""

from __future__ import annotations

import ast
import json
import os
from pathlib import Path
import re
import subprocess


REPO_ROOT = Path(__file__).resolve().parents[2]
SKIP_TOP_LEVEL = {
    ".cache",
    ".git",
    ".mypy_cache",
    ".pytest_cache",
    ".runtime",
    ".tools",
    ".venvs",
    "data",
    "hf_staging",
    "outputs",
}


def retained_files(suffixes: set[str]) -> list[Path]:
    files: list[Path] = []
    for root, dirs, names in os.walk(REPO_ROOT, topdown=True, followlinks=False):
        root_path = Path(root)
        relative_root = root_path.relative_to(REPO_ROOT)
        if relative_root == Path("."):
            dirs[:] = [name for name in dirs if name not in SKIP_TOP_LEVEL]
        else:
            dirs[:] = [name for name in dirs if name != "__pycache__"]
        if relative_root.parts[:2] == ("evaluation", "checkpoints"):
            dirs[:] = []
            continue
        for name in names:
            path = root_path / name
            if path.suffix in suffixes and path.is_file():
                files.append(path)
    return sorted(files)


def check_python() -> int:
    files = retained_files({".py"})
    for path in files:
        ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    return len(files)


def check_json() -> int:
    files = retained_files({".json"})
    for path in files:
        json.loads(path.read_text(encoding="utf-8"))
    return len(files)


def check_yaml() -> tuple[int, bool]:
    files = retained_files({".yaml", ".yml"})
    try:
        import yaml
    except ModuleNotFoundError:
        return len(files), False
    for path in files:
        yaml.safe_load(path.read_text(encoding="utf-8"))
    return len(files), True


def check_shell() -> int:
    files = retained_files({".sh"})
    for path in files:
        subprocess.run(["bash", "-n", str(path)], check=True)
    return len(files)


def check_symlinks() -> int:
    links: list[Path] = []
    for root, dirs, names in os.walk(REPO_ROOT, topdown=True, followlinks=False):
        root_path = Path(root)
        relative_root = root_path.relative_to(REPO_ROOT)
        if relative_root == Path("."):
            dirs[:] = [name for name in dirs if name not in SKIP_TOP_LEVEL]
        else:
            dirs[:] = [name for name in dirs if name != "__pycache__"]
        links.extend(path for name in (*dirs, *names) if (path := root_path / name).is_symlink())
    links.sort()
    for path in links:
        target = path.resolve(strict=True)
        if not target.is_relative_to(REPO_ROOT):
            raise AssertionError(f"Symlink escapes repository: {path} -> {target}")
    return len(links)


def check_root_readme_links() -> int:
    readme = REPO_ROOT / "README.md"
    targets = re.findall(r"\[[^\]]*\]\(([^)]+)\)", readme.read_text(encoding="utf-8"))
    local_targets = []
    for value in targets:
        if value.startswith(("http://", "https://", "mailto:", "#")):
            continue
        target = value.split("#", 1)[0]
        if not (REPO_ROOT / target).exists():
            raise FileNotFoundError(f"Broken root README link: {value}")
        local_targets.append(value)
    return len(local_targets)


def main() -> None:
    python_count = check_python()
    json_count = check_json()
    yaml_count, yaml_checked = check_yaml()
    shell_count = check_shell()
    symlink_count = check_symlinks()
    readme_link_count = check_root_readme_links()
    yaml_status = str(yaml_count) if yaml_checked else f"SKIP ({yaml_count}; install PyYAML)"
    print(
        "static audit: PASS "
        f"(Python AST={python_count}, JSON={json_count}, YAML={yaml_status}, "
        f"shell={shell_count}, symlinks={symlink_count}, README links={readme_link_count})"
    )


if __name__ == "__main__":
    main()
