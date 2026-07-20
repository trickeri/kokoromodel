"""kokoromodel inference engine — Kokoro v1.0 TTS via ONNX Runtime.

One voice pack + model warm in VRAM (or system RAM in CPU placement). Mirrors
the segmodel engine contract:
    load() / unload() / loaded() -> bool
    speak(text, voice, speed, lang) -> (samples_float32, sample_rate)
    voices() -> [str]

Backend is "cuda" (CUDAExecutionProvider) by default, or "cpu" when the
model-manager dock moves this model to RAM. kokoro-onnx selects its execution
provider from the ONNX_PROVIDER env var, which kokoromodel.sh sets per placement.

Phonemization uses espeakng-loader (a pip-bundled espeak-ng binary + data), so
no system espeak-ng install / sudo is required.
"""
from __future__ import annotations
import os
import shutil
import subprocess
import tempfile
import threading
import wave

import numpy as np


def _pitch_shift(samples: np.ndarray, sr: int, semitones: float) -> np.ndarray:
    """Shift pitch by N semitones, preserving duration, via the rubberband CLI.

    Returns float32 [-1,1]. If rubberband is missing, returns input unchanged.
    """
    rb = shutil.which("rubberband")
    if not rb or not semitones:
        return samples
    pcm = (np.clip(np.asarray(samples, np.float32), -1, 1) * 32767).astype("<i2")
    with tempfile.TemporaryDirectory() as d:
        src, dst = os.path.join(d, "in.wav"), os.path.join(d, "out.wav")
        with wave.open(src, "wb") as w:
            w.setnchannels(1); w.setsampwidth(2); w.setframerate(int(sr))
            w.writeframes(pcm.tobytes())
        try:
            subprocess.run([rb, "-p", str(semitones), "-q", src, dst],
                           check=True, stdout=subprocess.DEVNULL,
                           stderr=subprocess.DEVNULL)
            with wave.open(dst, "rb") as r:
                raw = r.readframes(r.getnframes())
        except (subprocess.CalledProcessError, OSError, wave.Error):
            return samples
    out = np.frombuffer(raw, "<i2").astype(np.float32) / 32767.0
    return out


def _wire_espeak() -> None:
    """Point phonemizer at the pip-bundled espeak-ng (no system install)."""
    try:
        import espeakng_loader
        from phonemizer.backend.espeak.wrapper import EspeakWrapper
        EspeakWrapper.set_library(espeakng_loader.get_library_path())
        EspeakWrapper.set_data_path(espeakng_loader.get_data_path())
    except Exception:  # noqa: BLE001 — fall back to system espeak if present
        pass


class KokoroTTS:
    def __init__(self, model_path: str, voices_path: str, backend: str = "cuda",
                 default_voice: str = "af_heart", default_speed: float = 1.0,
                 default_lang: str = "en-us", default_pitch: float = 0.0):
        self.model_path = model_path
        self.voices_path = voices_path
        self.backend = backend
        self.default_voice = default_voice
        self.default_speed = default_speed
        self.default_lang = default_lang
        # Semitones to shift after synthesis (voice identity, e.g. af_bella +4).
        # Duration-preserving, via the rubberband CLI. The Carla FX rack stacks
        # robotify/autotune/effects on top of this.
        self.default_pitch = default_pitch
        # cuda -> CUDA EP; cpu (RAM placement) -> CPU EP. kokoro-onnx reads this.
        os.environ.setdefault(
            "ONNX_PROVIDER",
            "CPUExecutionProvider" if backend == "cpu" else "CUDAExecutionProvider",
        )
        self._kokoro = None
        self._lock = threading.Lock()

    def load(self) -> bool:
        with self._lock:
            if self._kokoro is not None:
                return True
            _wire_espeak()
            import onnxruntime as rt
            from kokoro_onnx import Kokoro
            # kokoro-onnx builds its InferenceSession with bare provider NAMES and
            # no options, so the CUDA EP runs all defaults: a kNextPowerOfTwo memory
            # arena + EXHAUSTIVE cuDNN conv workspace search. Kokoro's input length
            # varies every utterance, so each new/longer shape ratchets the arena up
            # and it never releases -> VRAM crept to ~4.5 GB over a day. We build the
            # session ourselves with the arena bounded and hand it in via from_session.
            so = rt.SessionOptions()
            so.enable_cpu_mem_arena = False   # don't hold a growing host-side arena
            so.enable_mem_pattern = False     # off for dynamic shapes (no over-plan)
            if self.backend == "cpu":
                providers = ["CPUExecutionProvider"]
            else:
                cuda_opts = {
                    # grow by exactly what's requested (no power-of-two inflation)
                    "arena_extend_strategy": "kSameAsRequested",
                    # heuristic conv search -> no giant exhaustive scratch buffers
                    "cudnn_conv_algo_search": "HEURISTIC",
                }
                # Optional hard ceiling, e.g. KK_GPU_MEM_MB=1536. Kokoro's phoneme
                # length is already capped, so memory is bounded without this.
                cap = os.environ.get("KK_GPU_MEM_MB", "").strip()
                if cap.isdigit() and int(cap) > 0:
                    cuda_opts["gpu_mem_limit"] = str(int(cap) * 1024 * 1024)
                providers = [("CUDAExecutionProvider", cuda_opts),
                             "CPUExecutionProvider"]
            sess = rt.InferenceSession(self.model_path, sess_options=so,
                                       providers=providers)
            self._kokoro = Kokoro.from_session(sess, self.voices_path)
            return True

    def unload(self):
        with self._lock:
            self._kokoro = None

    def loaded(self) -> bool:
        return self._kokoro is not None

    def voices(self):
        if not self.loaded():
            self.load()
        try:
            return sorted(self._kokoro.get_voices())
        except Exception:  # noqa: BLE001
            return []

    def speak(self, text: str, voice: str | None = None,
              speed: float | None = None, lang: str | None = None,
              pitch: float | None = None):
        if not self.loaded():
            self.load()
        v = voice or self.default_voice
        s = float(speed) if speed is not None else self.default_speed
        lg = lang or self.default_lang
        p = float(pitch) if pitch is not None else self.default_pitch
        with self._lock:
            samples, sr = self._kokoro.create(text, voice=v, speed=s, lang=lg)
        if p:
            samples = _pitch_shift(samples, sr, p)
        return samples, sr


def make_engine(backend: str, model_path: str, voices_path: str, **kw):
    if backend not in ("cuda", "cpu"):
        raise ValueError(f"unknown backend: {backend!r} (cuda|cpu)")
    return KokoroTTS(model_path, voices_path, backend=backend, **kw)
