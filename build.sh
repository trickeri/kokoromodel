#!/usr/bin/env bash
# Build kokoromodel's Python venv.
#   ./build.sh          # cuda (kokoro-onnx via onnxruntime-gpu, default)
#   ./build.sh cpu      # CPU-only onnxruntime
#   ./build.sh torch    # PyTorch kokoro in venv-torch (per-phoneme durations for
#                       # viseme lip-sync; select at runtime with KK_ENGINE=torch)
# Sudo-free: phonemization uses the pip-bundled espeak-ng (espeakng-loader).
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BACKEND="${1:-cuda}"

# torch engine: a SEPARATE venv-torch (uv + py3.12; the onnx venv stays on the
# system python). Kept apart so the two engines never share deps. Needs uv + a
# CUDA 12.x driver for the GPU wheels.
if [ "$BACKEND" = torch ]; then
    command -v uv >/dev/null || { echo "torch engine needs 'uv' (https://astral.sh/uv)" >&2; exit 1; }
    uv venv --python 3.12 "$HERE/venv-torch"
    VENV="$HERE/venv-torch"
    VIRTUAL_ENV="$VENV" uv pip install torch --index-url https://download.pytorch.org/whl/cu124
    VIRTUAL_ENV="$VENV" uv pip install kokoro "misaki[en]" soundfile
    # misaki's English g2p wants the small spaCy model
    VIRTUAL_ENV="$VENV" uv pip install \
        "en-core-web-sm @ https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-3.8.0/en_core_web_sm-3.8.0-py3-none-any.whl"
    echo "kokoromodel venv-torch ready — run with KK_ENGINE=torch" >&2
    exit 0
fi

[ "$BACKEND" = cuda ] || [ "$BACKEND" = cpu ] || { echo "usage: $0 [cuda|cpu|torch]" >&2; exit 1; }

python3 -m venv "$HERE/venv"
PIP="$HERE/venv/bin/pip"
"$PIP" install -q --upgrade pip
"$PIP" install -q kokoro-onnx espeakng-loader

if [ "$BACKEND" = cuda ]; then
    # kokoro-onnx pulls the CPU onnxruntime, which shadows onnxruntime-gpu and
    # hides CUDAExecutionProvider. Drop it and reinstall the GPU build clean.
    "$PIP" uninstall -y onnxruntime onnxruntime-gpu >/dev/null 2>&1 || true
    "$PIP" install -q --force-reinstall --no-deps onnxruntime-gpu
fi

echo "$BACKEND" > "$HERE/backend.txt"
echo "kokoromodel venv ready ($BACKEND)" >&2
