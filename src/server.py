"""kokoromodel — system-wide text-to-speech daemon.

Kokoro v1.0 warm in VRAM, exposed over local HTTP so any app (voiceagent, the
honeycomb taskbar, scripts) can POST text and get speech back. Backend
(cuda | cpu) is chosen by env at launch via the model-manager device contract.

Routes (mirrors segmodel's shape):
  POST /speak?voice=&speed=&lang=    JSON {text,voice?,speed?,lang?} OR raw text
                                     body -> audio/wav (24 kHz mono PCM16)
  GET  /voices                       JSON: available voice names
  GET  /status                       JSON: model, backend, loaded, port, voices
  POST /load  /activate              ensure model resident; JSON status
  POST /unload /park                 free the model; JSON status
"""
from __future__ import annotations
import base64
import io
import json
import os
import sys
import wave
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlparse

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

HOST = os.environ.get("KK_HOST", "127.0.0.1")
PORT = int(os.environ.get("KK_PORT", "48490"))
BACKEND = os.environ.get("KK_BACKEND", "cuda")
# onnx (default: kokoro-onnx, bounded VRAM) | torch (PyTorch kokoro, exposes per-phoneme
# durations for viseme lip-sync). Fully reversible: flip KK_ENGINE and restart.
KK_ENGINE = os.environ.get("KK_ENGINE", "onnx")
MODEL = os.environ.get("KK_MODEL", "")
VOICES = os.environ.get("KK_VOICES", "")
DEF_VOICE = os.environ.get("KK_VOICE", "af_bella")
DEF_SPEED = float(os.environ.get("KK_SPEED", "1.05"))
DEF_LANG = os.environ.get("KK_LANG", "en-us")
DEF_PITCH = float(os.environ.get("KK_PITCH", "4"))

if KK_ENGINE == "torch":
    from engine_torch import make_torch_engine  # noqa: E402
    ENGINE = make_torch_engine(MODEL, VOICES, backend=BACKEND, default_voice=DEF_VOICE,
                               default_speed=DEF_SPEED, default_lang=DEF_LANG,
                               default_pitch=DEF_PITCH)
else:
    from engine import make_engine  # noqa: E402
    ENGINE = make_engine(BACKEND, MODEL, VOICES, default_voice=DEF_VOICE,
                         default_speed=DEF_SPEED, default_lang=DEF_LANG,
                         default_pitch=DEF_PITCH)


def _wav_bytes(samples: np.ndarray, sr: int) -> bytes:
    """float32 [-1,1] mono -> 16-bit PCM WAV (stdlib, no soundfile dep)."""
    pcm = np.clip(np.asarray(samples, dtype=np.float32), -1.0, 1.0)
    pcm = (pcm * 32767.0).astype("<i2")
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(int(sr))
        w.writeframes(pcm.tobytes())
    return buf.getvalue()


def _status() -> dict:
    return {"service": "kokoromodel", "model": MODEL, "backend": BACKEND,
            "engine": KK_ENGINE, "aligned": hasattr(ENGINE, "speak_aligned"),
            "loaded": ENGINE.loaded(), "host": HOST, "port": PORT,
            "default_voice": DEF_VOICE, "default_pitch": DEF_PITCH}


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *a):  # quiet
        pass

    def _json(self, obj, code=200):
        data = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        path = urlparse(self.path).path
        if path == "/status":
            self._json(_status())
        elif path == "/voices":
            self._json({"voices": ENGINE.voices()})
        else:
            self._json({"error": "not found"}, 404)

    def do_POST(self):
        parsed = urlparse(self.path)
        path, q = parsed.path, parse_qs(parsed.query)
        if path in ("/load", "/activate"):
            ENGINE.load(); return self._json(_status())
        if path in ("/unload", "/park"):
            ENGINE.unload(); return self._json(_status())
        if path != "/speak":
            return self._json({"error": "not found"}, 404)

        n = int(self.headers.get("Content-Length", "0"))
        body = self.rfile.read(n) if n else b""
        ctype = self.headers.get("Content-Type", "")

        text = voice = speed = lang = pitch = None
        if "application/json" in ctype:
            try:
                d = json.loads(body or b"{}")
            except json.JSONDecodeError:
                return self._json({"error": "bad JSON"}, 400)
            text = d.get("text"); voice = d.get("voice")
            speed = d.get("speed"); lang = d.get("lang"); pitch = d.get("pitch")
        else:
            text = body.decode("utf-8", "replace")
        # query params override / supply
        voice = (q.get("voice") or [voice])[0]
        speed = (q.get("speed") or [speed])[0]
        lang = (q.get("lang") or [lang])[0]
        pitch = (q.get("pitch") or [pitch])[0]

        if not text or not text.strip():
            return self._json({"error": "no text"}, 400)

        align = None
        try:
            if hasattr(ENGINE, "speak_aligned"):
                samples, sr, align = ENGINE.speak_aligned(
                    text.strip(), voice=voice, speed=speed, lang=lang, pitch=pitch)
            else:
                samples, sr = ENGINE.speak(text.strip(), voice=voice, speed=speed,
                                           lang=lang, pitch=pitch)
        except Exception as e:  # noqa: BLE001
            return self._json({"error": f"{type(e).__name__}: {e}"}, 500)

        data = _wav_bytes(samples, sr)
        self.send_response(200)
        self.send_header("Content-Type", "audio/wav")
        self.send_header("Content-Length", str(len(data)))
        # Per-phoneme timing (torch engine only): base64(JSON [{ph,start,end},...]) so
        # a client can drive viseme lip-sync. Absent on the onnx engine.
        if align:
            self.send_header("X-Kokoro-Align",
                             base64.b64encode(json.dumps(align).encode()).decode())
        self.end_headers()
        self.wfile.write(data)


def main():
    if not MODEL or not VOICES:
        print("kokoromodel: set KK_MODEL and KK_VOICES", file=sys.stderr)
        sys.exit(1)
    ENGINE.load()
    print(f"kokoromodel: {os.path.basename(MODEL)} on {HOST}:{PORT}  [backend={BACKEND}]",
          file=sys.stderr)
    ThreadingHTTPServer((HOST, PORT), Handler).serve_forever()


if __name__ == "__main__":
    main()
