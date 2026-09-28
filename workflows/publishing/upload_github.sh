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

# Keep transport workarounds local to this publisher. Git protocol v1 also
# avoids a truncated v2 ref advertisement on unreliable HTTP/2 connections.
# Disable credential helpers for this command so an entered token is not saved.
git_remote() {
  git -c http.version=HTTP/1.1 -c protocol.version=1 -c credential.helper= "$@"
}

remote_main_sha() {
  local attempt refs
  for attempt in 1 2 3; do
    if refs=$(git_remote ls-remote --heads origin main); then
      printf '%s\n' "$refs" | awk 'NR == 1 { print $1 }'
      return 0
    fi
    if (( attempt < 3 )); then
      printf 'Remote lookup failed; retrying (%s/3)...\n' "$attempt" >&2
      sleep "$attempt"
    fi
  done
  echo 'Could not verify remote main.' >&2
  return 1
}

local_sha=$(git rev-parse HEAD)
remote_sha=$(remote_main_sha)
if [[ -n $remote_sha && $remote_sha != "$local_sha" ]] &&
   ! git merge-base --is-ancestor "$remote_sha" "$local_sha"; then
  echo 'Remote main is not an ancestor of this release; inspect it before publishing. No force push is used.' >&2
  exit 2
fi

if [[ ${1:-} == --dry-run ]]; then
  printf 'Ready to publish %s to https://github.com/Mark-zjtang/ALAM\n' "$local_sha"
  exit 0
fi
if [[ $remote_sha == "$local_sha" ]]; then
  echo 'This release is already on GitHub.'
  exit 0
fi
[[ -t 0 && -t 1 ]] || {
  echo 'Run this script in an interactive terminal for GitHub authentication.' >&2
  exit 2
}

# Bypass stale IDE askpass sockets and prompt in the terminal. At the password
# prompt, enter a GitHub access token; the command disables credential helpers.
unset GIT_ASKPASS SSH_ASKPASS VSCODE_GIT_ASKPASS_NODE VSCODE_GIT_ASKPASS_MAIN VSCODE_GIT_IPC_HANDLE
unset GIT_TRACE GIT_TRACE_CURL GIT_CURL_VERBOSE GIT_TRACE_PACKET GIT_TRACE2 GIT_TRACE2_EVENT GIT_TRACE2_PERF
export GIT_TERMINAL_PROMPT=1
echo 'Pushing ALAM code and Git LFS assets. If prompted, use your GitHub username and access token.'
# These release assets are not managed with LFS file locking. Skip only the
# lock-verification API that can time out; the LFS objects still upload.
git_remote -c 'lfs.https://github.com/Mark-zjtang/ALAM.git/info/lfs.locksverify=false' \
  push -u origin main

published_sha=$(remote_main_sha)
[[ $published_sha == "$local_sha" ]] || {
  echo 'Push returned, but remote main does not match the local release.' >&2
  exit 1
}
printf 'Published and verified: %s\n' "$published_sha"
