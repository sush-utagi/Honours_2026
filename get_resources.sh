#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
DATA_DIR="${ROOT_DIR}/diffusion-model/data"

FORCE=0
while [[ $# -gt 0 ]]; do
  case "$1" in
    --force|-f) FORCE=1; shift ;;
    -h|--help)
      cat <<'EOF'
Usage: ./download_sd15_resources.sh [--force]

Downloads Stable Diffusion v1.5 checkpoint + tokenizer files into diffusion-model/data.

If the Hugging Face repo is gated for you, export an access token:
  export HF_TOKEN=hf_...
EOF
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      exit 2
      ;;
  esac
done

mkdir -p "${DATA_DIR}"

if command -v curl >/dev/null 2>&1; then
  HTTP_GET=(curl -L --fail --retry 3 --retry-delay 2 -C -)
elif command -v wget >/dev/null 2>&1; then
  HTTP_GET=(wget -q --show-progress --tries=3 --timeout=30 -c -O)
else
  echo "Need curl or wget installed." >&2
  exit 1
fi

download() {
  local url="$1"
  local out="$2"
  local tmp="${out}.part"

  if [[ "${FORCE}" -eq 0 && -s "${out}" ]]; then
    echo "OK  ${out} (already exists)"
    return 0
  fi

  echo "GET ${url}"
  if [[ "${HTTP_GET[0]}" == "curl" ]]; then
    if [[ -n "${HF_TOKEN:-}" ]]; then
      "${HTTP_GET[@]}" -H "Authorization: Bearer ${HF_TOKEN}" -o "${tmp}" "${url}"
    else
      "${HTTP_GET[@]}" -o "${tmp}" "${url}"
    fi
  else
    # wget form is: wget ... -O <file> <url>
    if [[ -n "${HF_TOKEN:-}" ]]; then
      "${HTTP_GET[@]}" "${tmp}" --header="Authorization: Bearer ${HF_TOKEN}" "${url}"
    else
      "${HTTP_GET[@]}" "${tmp}" "${url}"
    fi
  fi
  mv -f "${tmp}" "${out}"
  echo "OK  ${out}"
}

HF_BASE="https://huggingface.co/stable-diffusion-v1-5/stable-diffusion-v1-5/resolve/main"

download "${HF_BASE}/tokenizer/vocab.json" "${DATA_DIR}/vocab.json"
download "${HF_BASE}/tokenizer/merges.txt" "${DATA_DIR}/merges.txt"
download "${HF_BASE}/v1-5-pruned-emaonly.ckpt" "${DATA_DIR}/v1-5-pruned-emaonly.ckpt"

echo "Done. Files are in: ${DATA_DIR}"
