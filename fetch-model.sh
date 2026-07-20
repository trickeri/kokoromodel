#!/usr/bin/env bash
# Fetch the Kokoro v1.0 ONNX model + voice pack into models/.
# Sudo-free; just downloads. Sizes: model ~325 MB (fp32), voices ~27 MB.
#   ./fetch-model.sh         # fp32 model (best quality, plenty of VRAM on the 4090)
#   ./fetch-model.sh fp16    # half-precision model (smaller/faster, near-identical)
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEST="$HERE/models"
mkdir -p "$DEST"

VARIANT="${1:-fp32}"
BASE="https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0"
case "$VARIANT" in
    fp32) MODEL_FILE="kokoro-v1.0.onnx" ;;
    fp16) MODEL_FILE="kokoro-v1.0.fp16.onnx" ;;
    *) echo "usage: $0 [fp32|fp16]" >&2; exit 1 ;;
esac

fetch() {  # url dest
    [ -s "$2" ] && { echo "have $(basename "$2")"; return; }
    echo "fetching $(basename "$2") ..."
    curl -fL --retry 3 -o "$2" "$1"
}

fetch "$BASE/$MODEL_FILE" "$DEST/$MODEL_FILE"
fetch "$BASE/voices-v1.0.bin" "$DEST/voices-v1.0.bin"
echo "kokoro model ready in $DEST ($VARIANT)" >&2
