"""kokoromodel PyTorch engine — Kokoro v1.0 via the hexgrad `kokoro` KPipeline.

Same contract as engine.py (load/unload/loaded/voices/speak) so the server can use
either, PLUS speak_aligned(), which also returns per-phoneme timing (start/end
seconds) from the model's duration predictor (`Result.pred_dur`). That alignment is
what the ONNX export can't give — it's used for viseme lip-sync. Selected with
KK_ENGINE=torch; the ONNX engine stays the default so this is fully reversible.

Voice/pitch/speed and the rubberband pitch shift match engine.py exactly, and the
weights are the same Kokoro v1.0, so the voice is identical — only the runtime differs.
"""
from __future__ import annotations
import os
import threading

import numpy as np

from engine import _pitch_shift  # reuse the rubberband duration-preserving shift

SAMPLE_RATE = 24000
# pred_dur is in 600-sample frames (24000/600 = 40 frames/sec), so seconds = frames/40.
_FRAMES_PER_SEC = 40.0


class KokoroTorchTTS:
    def __init__(self, repo_id: str = "hexgrad/Kokoro-82M", backend: str = "cuda",
                 default_voice: str = "af_bella", default_speed: float = 1.0,
                 default_lang: str = "en-us", default_pitch: float = 0.0, **_):
        self.repo_id = repo_id
        self.device = "cpu" if backend == "cpu" else "cuda"
        self.default_voice = default_voice
        self.default_speed = default_speed
        self.default_pitch = default_pitch
        self._pipe = None
        self._lock = threading.Lock()

    def load(self) -> bool:
        with self._lock:
            if self._pipe is not None:
                return True
            # bound the torch CUDA allocator (the ONNX build was tuned for VRAM; keep
            # parity here so the torch engine doesn't balloon the arena)
            os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
            from kokoro import KModel, KPipeline
            model = KModel(repo_id=self.repo_id).to(self.device).eval()
            # lang_code 'a' = American English (misaki g2p, matches en-us)
            self._pipe = KPipeline(lang_code="a", repo_id=self.repo_id, model=model)
            return True

    def unload(self):
        with self._lock:
            self._pipe = None
        try:
            import torch
            torch.cuda.empty_cache()
        except Exception:  # noqa: BLE001
            pass

    def loaded(self) -> bool:
        return self._pipe is not None

    def voices(self):
        # Kokoro v1.0 voices (HF-hosted; downloaded lazily by load_voice). Static list
        # to avoid a network hit on GET /voices — matches the onnx pack's names.
        return sorted([
            "af_heart", "af_bella", "af_sarah", "af_nicole", "af_sky",
            "am_michael", "am_adam", "am_echo", "am_liam", "am_onyx",
            "bf_emma", "bf_isabella", "bm_george", "bm_lewis",
        ])

    def _synth(self, text: str, voice: str, speed: float):
        """Run the pipeline, concatenate audio, and build a per-phoneme timeline
        from each segment's pred_dur. Returns (samples float32, sr, align) where
        align = [{"ph": <phoneme char>, "start": s, "end": s}, ...] in seconds."""
        results = list(self._pipe(text, voice=voice, speed=speed))
        audios: list[np.ndarray] = []
        align: list[dict] = []
        t_off = 0.0
        for r in results:
            audio = r.audio
            if audio is None:
                continue
            a = audio.detach().cpu().numpy().astype(np.float32)
            audios.append(a)
            ps = r.phonemes or ""
            pd = r.pred_dur
            if pd is not None and ps:
                pd = [int(x) for x in pd.detach().cpu().tolist()]
                # pd[0] = leading pad, pd[1..len(ps)] = one per phoneme char, pd[-1] = pad
                t = t_off + (pd[0] / _FRAMES_PER_SEC if pd else 0.0)
                for i, ph in enumerate(ps):
                    d = pd[i + 1] / _FRAMES_PER_SEC if (i + 1) < len(pd) else 0.0
                    align.append({"ph": ph, "start": round(t, 3), "end": round(t + d, 3)})
                    t += d
            t_off += len(a) / float(SAMPLE_RATE)
        samples = np.concatenate(audios) if audios else np.zeros(0, np.float32)
        return samples, SAMPLE_RATE, align

    def speak_aligned(self, text: str, voice: str | None = None,
                      speed: float | None = None, lang: str | None = None,
                      pitch: float | None = None):
        if not self.loaded():
            self.load()
        v = voice or self.default_voice
        s = float(speed) if speed is not None else self.default_speed
        p = float(pitch) if pitch is not None else self.default_pitch
        with self._lock:
            samples, sr, align = self._synth(text, v, s)
        if p:
            samples = _pitch_shift(samples, sr, p)  # duration-preserving -> timings hold
        return samples, sr, align

    def speak(self, text: str, voice: str | None = None,
              speed: float | None = None, lang: str | None = None,
              pitch: float | None = None):
        samples, sr, _ = self.speak_aligned(text, voice=voice, speed=speed,
                                            lang=lang, pitch=pitch)
        return samples, sr


def make_torch_engine(model_path: str, voices_path: str, backend: str = "cuda", **kw):
    # model_path/voices_path are ignored (torch kokoro pulls weights from HF by
    # repo_id) but kept in the signature to mirror engine.make_engine().
    return KokoroTorchTTS(backend=backend, **kw)
