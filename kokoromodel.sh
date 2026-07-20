#!/usr/bin/env bash
# kokoromodel — system-wide text-to-speech daemon. Runs the Python server in its
# venv with Kokoro v1.0 warm in VRAM. Placement (gpu|ram) is read from the
# model-manager device file and maps to the ONNX execution provider.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

[ -f "$HERE/config.env" ] && . "$HERE/config.env"

export KK_HOST="${KK_HOST:-127.0.0.1}"
export KK_PORT="${KK_PORT:-48490}"
export KK_BACKEND="${KK_BACKEND:-$( [ -f "$HERE/backend.txt" ] && cat "$HERE/backend.txt" || echo cuda )}"

# Device placement (gpu|ram) — written by the model-manager dock, read on launch.
# "ram" forces the CPU ONNX Runtime provider (Kokoro runs on CPU + system RAM).
DEVICE_FILE="${XDG_CACHE_HOME:-$HOME/.cache}/modelmanager/kokoromodel.device"
KK_DEVICE="$( [ -f "$DEVICE_FILE" ] && cat "$DEVICE_FILE" || echo gpu )"
[ "$KK_DEVICE" = ram ] && export KK_BACKEND=cpu

export KK_MODEL="${KK_MODEL:-$HERE/models/kokoro-v1.0.onnx}"
export KK_VOICES="${KK_VOICES:-$HERE/models/voices-v1.0.bin}"

# Engine: onnx (default, kokoro-onnx, bounded VRAM) | torch (PyTorch kokoro in
# venv-torch, exposes per-phoneme durations for viseme lip-sync). Reversible: set
# KK_ENGINE=torch in config.env and restart; unset to go back to onnx.
export KK_ENGINE="${KK_ENGINE:-onnx}"
if [ "$KK_ENGINE" = torch ]; then
    VENV="$HERE/venv-torch"
    export VIRTUAL_ENV="$VENV"   # misaki/spacy runtime helpers look for it
else
    VENV="$HERE/venv"
fi

[ -x "$VENV/bin/python" ] || { echo "venv missing ($KK_ENGINE) — run ./build.sh" >&2; exit 1; }
[ -s "$KK_MODEL" ] && [ -s "$KK_VOICES" ] || { echo "model missing — run ./fetch-model.sh" >&2; exit 1; }

echo "kokoromodel: ${KK_MODEL##*/} on $KK_HOST:$KK_PORT (backend=$KK_BACKEND engine=$KK_ENGINE)" >&2
exec "$VENV/bin/python" "$HERE/src/server.py"
