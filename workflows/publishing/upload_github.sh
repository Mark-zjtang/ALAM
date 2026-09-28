#!/usr/bin/env bash
# Push the audited ALAM code release to its GitHub repository.
set -Eeuo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

if [[ $# -gt 1 || ( $# -eq 1 && $1 != --dry-run ) ]]; then
  echo 'Usage: bash workflows/publishing/upload_github.sh [--dry-run]' >&2
  exit 64
fi

[[ $(git rev-parse --show-toplevel) == "$ROOT" ]] || {
  echo 'Run from the ALAM release repository.' >&2
  exit 2
}
[[ $(git branch --show-current) == main ]] || {
  echo 'The release must be on the main branch.' >&2
  exit 2
}
[[ $(git remote get-url origin) == https://github.com/Mark-zjtang/ALAM.git ]] || {
  echo 'Unexpected origin; refusing to push.' >&2
  exit 2
}
[[ -z $(git status --porcelain) ]] || {
  echo 'Commit or remove uncommitted changes before publishing.' >&2
  exit 2
}
command -v git-lfs >/dev/null || {
  echo 'Git LFS is required for the bundled LIBERO assets.' >&2
  exit 2
}
git lfs version >/dev/null

local_sha=$(git rev-parse HEAD)
remote_sha=$(git ls-remote --heads origin main | awk '{print $1}')
if [[ -n $remote_sha && $remote_sha != "$local_sha" ]]; then
  echo 'Remote main differs from this release; inspect it before publishing. No force push is used.' >&2
  exit 2
fi

if [[ ${1:-} == --dry-run ]]; then
  printf 'Ready to publish %s to https://github.com/Mark-zjtang/ALAM\n' "$local_sha"
  exit 0
fi
if [[ -n $remote_sha ]]; then
  echo 'This release is already on GitHub.'
  exit 0
fi
[[ -t 0 && -t 1 ]] || {
  echo 'Run this script in an interactive terminal for GitHub authentication.' >&2
  exit 2
}

# Bypass stale IDE askpass sockets; Git may use a working credential helper or
# prompt in the terminal. At the password prompt, enter a GitHub access token.
unset GIT_ASKPASS SSH_ASKPASS VSCODE_GIT_ASKPASS_NODE VSCODE_GIT_ASKPASS_MAIN VSCODE_GIT_IPC_HANDLE
export GIT_TERMINAL_PROMPT=1
echo 'Pushing ALAM code and Git LFS assets. If prompted, use your GitHub username and access token.'
git push -u origin main

published_sha=$(git ls-remote --heads origin main | awk '{print $1}')
[[ $published_sha == "$local_sha" ]] || {
  echo 'Push returned, but remote main does not match the local release.' >&2
  exit 1
}
printf 'Published and verified: %s\n' "$published_sha"
