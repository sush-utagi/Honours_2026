#!/usr/bin/env bash

# This script opens the first 10 input images from a JSON file.
# Usage: ./scripts/inspect_input_images.sh /path/to/prompts.json

set -euo pipefail

if ! command -v jq >/dev/null 2>&1; then
  echo "jq is required (install via 'brew install jq' or your package manager)." >&2
  exit 1
fi

json_path=${1:-}
if [[ -z "$json_path" || ! -f "$json_path" ]]; then
  echo "Usage: $0 /path/to/prompts.json" >&2
  exit 1
fi

jq -r '.samples[].init_image // empty' "$json_path" | head -n 10 | while IFS= read -r img; do
  if [[ -f "$img" ]]; then
    echo "Opening $img"
    open "$img"
  else
    echo "Skipping missing file: $img"
  fi
done
