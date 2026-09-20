#!/usr/bin/env bash
# Deploys the commit you name to the VM, or HEAD when you name nothing.
#
#   ./scripts/deploy.sh          # deploy the current HEAD
#   ./scripts/deploy.sh <sha>    # deploy that commit
#
# The VM holds a clone at ~/docs/ml. This pushes, then fetches and rebuilds
# there, then waits for the health check to report the commit it just sent.

set -euo pipefail

HOST="${ML_HOST:-ifkash@vm.ifkash.dev}"
KEY="${ML_SSH_KEY:-$HOME/.ssh/id_ed25519_kh}"
REMOTE_DIR="${ML_REMOTE_DIR:-~/docs/ml}"
SHA="${1:-$(git rev-parse HEAD)}"

echo "Deploying ${SHA:0:12} to $HOST:$REMOTE_DIR"

git push origin HEAD

ssh -i "$KEY" "$HOST" bash -s <<REMOTE
set -euo pipefail
cd $REMOTE_DIR
git fetch --all --prune
git checkout --detach $SHA
export GIT_SHA=$SHA
docker compose up -d --build
REMOTE

echo "Waiting for the health check"
for _ in \$(seq 1 60); do
  reported=\$(ssh -i "$KEY" "$HOST" \
    "curl -fsS http://127.0.0.1:8093/api/health" 2>/dev/null \
    | python3 -c "import json,sys; print(json.load(sys.stdin).get('commit',''))" 2>/dev/null || true)
  if [ "\$reported" = "$SHA" ]; then
    echo "Live at ${SHA:0:12}"
    exit 0
  fi
  sleep 5
done

echo "The app did not report $SHA within five minutes. Check the container logs." >&2
exit 1
