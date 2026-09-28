#!/usr/bin/env python3
"""Exercise publication ancestry checks against a temporary local Git remote."""

import os
from pathlib import Path
import shutil
import subprocess
import tempfile

ROOT = Path(__file__).resolve().parents[2]
SCRIPT = Path("workflows/publishing/upload_github.sh")


def main():
    real_git = shutil.which("git")
    assert real_git
    env = dict(os.environ, GIT_CONFIG_GLOBAL=os.devnull, GIT_CONFIG_NOSYSTEM="1",
               GIT_AUTHOR_NAME="Fixture", GIT_AUTHOR_EMAIL="fixture@example.invalid",
               GIT_COMMITTER_NAME="Fixture", GIT_COMMITTER_EMAIL="fixture@example.invalid")

    def git(cwd, *args):
        return subprocess.run([real_git, *args], cwd=cwd, env=env, text=True,
                              capture_output=True, check=True).stdout.strip()

    with tempfile.TemporaryDirectory(prefix="alam-github-preflight-") as temporary:
        base = Path(temporary)
        remote, writer, checkout = (base / name for name in ("remote.git", "writer", "checkout"))
        git(base, "init", "--bare", "--initial-branch=main", str(remote))
        git(base, "init", "--initial-branch=main", str(writer))
        (writer / SCRIPT).parent.mkdir(parents=True)
        shutil.copyfile(ROOT / SCRIPT, writer / SCRIPT)
        git(writer, "add", ".")
        git(writer, "commit", "-m", "Initial release")
        git(writer, "remote", "add", "origin", str(remote))
        git(writer, "push", "origin", "main")
        git(base, "clone", str(remote), str(checkout))

        # Keep transport local while satisfying the publisher's fixed URL gate.
        binaries = base / "bin"
        binaries.mkdir()
        (binaries / "git").write_text(
            '#!/usr/bin/env bash\n'
            'if [[ ${1:-} == remote && ${2:-} == get-url && ${3:-} == origin ]]; then\n'
            '  echo https://github.com/Mark-zjtang/ALAM.git; exit 0\nfi\n'
            'for arg in "$@"; do\n'
            '  if [[ $arg == fetch && ${ALAM_TEST_FETCH_FAIL:-0} == 1 ]]; then exit 42; fi\n'
            '  if [[ $arg == push ]]; then echo "Unexpected push" >&2; exit 43; fi\n'
            'done\nexec "$ALAM_TEST_REAL_GIT" "$@"\n'
        )
        (binaries / "git-lfs").write_text('#!/usr/bin/env bash\necho "git-lfs fixture"\n')
        for path in binaries.iterdir():
            path.chmod(0o755)
        run_env = dict(env, PATH=str(binaries) + os.pathsep + env.get("PATH", ""),
                       ALAM_TEST_REAL_GIT=real_git)

        def preflight(code, expected):
            before = git(checkout, "rev-parse", "HEAD")
            remote_before = git(remote, "rev-parse", "main")
            result = subprocess.run(["bash", str(SCRIPT), "--dry-run"], cwd=checkout,
                                    env=run_env, text=True, capture_output=True)
            output = result.stdout + result.stderr
            assert result.returncode == code, output
            assert expected in output, output
            assert "Not a valid commit name" not in output, output
            assert git(checkout, "rev-parse", "HEAD") == before
            assert git(remote, "rev-parse", "main") == remote_before

        def commit(cwd, filename):
            (cwd / filename).write_text(filename + "\n")
            git(cwd, "add", filename)
            git(cwd, "commit", "-m", filename)

        preflight(0, "Ready to publish")
        commit(writer, "remote-readme-change")
        git(writer, "push", "origin", "main")
        new_remote = git(writer, "rev-parse", "HEAD")
        unknown = subprocess.run([real_git, "cat-file", "-e", new_remote], cwd=checkout,
                                 env=env, capture_output=True)
        assert unknown.returncode != 0, "Fixture must start with an unknown remote commit"
        preflight(2, "Local main is behind GitHub")
        git(checkout, "merge", "--ff-only", "FETCH_HEAD")
        commit(checkout, "local-feature")
        preflight(0, "Ready to publish")
        commit(writer, "another-remote-change")
        git(writer, "push", "origin", "main")
        preflight(2, "different new commits")
        run_env["ALAM_TEST_FETCH_FAIL"] = "1"
        preflight(1, "Could not fetch remote main")

    print("GitHub preflight: PASS (equal, unknown remote, ahead, diverged, fetch failure)")


if __name__ == "__main__":
    main()
