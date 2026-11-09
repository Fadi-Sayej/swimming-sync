#!/usr/bin/env bash
# Start Label Studio for milestone 2 labeling: http://localhost:8080
# Login is in data/raw/labelstudio/credentials.txt (not in git). Stop with Ctrl+C.
set -euo pipefail
cd "$(dirname "$0")/.."
ROOT="$PWD"
CRED="$ROOT/data/raw/labelstudio/credentials.txt"
LS="$HOME/.venvs/label-studio/bin/label-studio"
[ -x "$LS" ] || { echo "Label Studio not installed. Run: python3 -m venv ~/.venvs/label-studio && ~/.venvs/label-studio/bin/pip install label-studio"; exit 1; }

export LABEL_STUDIO_BASE_DATA_DIR="$ROOT/data/raw/labelstudio"   # database with all your labels
export LABEL_STUDIO_LOCAL_FILES_SERVING_ENABLED=true
export LABEL_STUDIO_LOCAL_FILES_DOCUMENT_ROOT="$ROOT/data/raw/frames/m2"
export LABEL_STUDIO_DISABLE_SIGNUP_WITHOUT_LINK=true
export LABEL_STUDIO_ENABLE_LEGACY_API_TOKEN=true   # lets swimsync.dataset.labelstudio import tasks
if [ -f "$CRED" ]; then
  export LABEL_STUDIO_USERNAME="$(awk '/^email/{print $2}' "$CRED")"
  export LABEL_STUDIO_PASSWORD="$(awk '/^password/{print $2}' "$CRED")"
  export LABEL_STUDIO_USER_TOKEN="$(awk '/^api_token/{print $2}' "$CRED")"
fi
exec "$LS" start --port 8080 --no-browser "$@"
