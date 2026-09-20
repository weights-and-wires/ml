#!/usr/bin/env bash
# Deploys the commit you name to the VM, or HEAD when you name nothing.
#
#   ./scripts/deploy.sh          # deploy the current HEAD
#   ./scripts/deploy.sh <ref>    # deploy that commit or tag
#
# The repository is private and the organisation disables deploy keys, so the
# VM holds no GitHub credential. This ships the committed tree with
# `git archive` over the SSH connection instead, which sends exactly what is
# committed and nothing from the working directory.
#
# `.env` on the VM is never touched: it holds the session secret and the RunPod
# key, and neither is in the repository.

set -euo pipefail

HOST="${ML_HOST:-ifkash@vm.ifkash.dev}"
KEY="${ML_SSH_KEY:-$HOME/.ssh/id_ed25519_kh}"
REMOTE_DIR="${ML_REMOTE_DIR:-docs/ml}"
REF="${1:-HEAD}"
SHA="$(git rev-parse "$REF")"

if ! git diff --quiet || ! git diff --cached --quiet; then
  echo "Warning: the working directory has changes that will not be deployed." >&2
fi

echo "Shipping ${SHA:0:12} to $HOST:~/$REMOTE_DIR"

ssh -i "$KEY" "$HOST" "mkdir -p ~/$REMOTE_DIR"

# --delete would take `.env` and the data volume's bind points with it, so the
# tree is replaced file by file and only tracked paths are overwritten.
git archive --format=tar "$SHA" \
  | ssh -i "$KEY" "$HOST" "tar -x -C ~/$REMOTE_DIR"

echo "Building"
ssh -i "$KEY" "$HOST" "cd ~/$REMOTE_DIR && GIT_SHA=$SHA docker compose up -d --build" 2>&1 \
  | tail -20

echo "Waiting for the health check to report $SHA"
for _ in $(seq 1 60); do
  reported="$(ssh -i "$KEY" "$HOST" \
    "curl -fsS http://127.0.0.1:8093/api/health 2>/dev/null" \
    | python3 -c "import json,sys; print(json.load(sys.stdin).get('commit',''))" 2>/dev/null || true)"
  if [ "$reported" = "$SHA" ]; then
    echo "Live at ${SHA:0:12}"
    exit 0
  fi
  sleep 5
done

echo "The app did not report $SHA within five minutes. Check the container logs:" >&2
echo "  ssh -i $KEY $HOST 'cd ~/$REMOTE_DIR && docker compose logs --tail 60'" >&2
exit 1
