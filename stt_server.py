"""Speech-to-text sidecar for transluent.

Protocol (stdin, binary): repeated frames of
    u32 LE  sample_count
    f32 LE  samples[sample_count]   (mono, 16 kHz)
Output (stdout): one JSON object per line.
"""
import argparse
import glob
import json
import os
import site
import struct
import sys
import time


def out(obj):
    sys.stdout.write(json.dumps(obj, ensure_ascii=False) + "\n")
    sys.stdout.flush()


def add_cuda_dlls():
    dirs = list(site.getsitepackages())
    try:
        dirs.append(site.getusersitepackages())
    except Exception:
        pass
    for sp in dirs:
        for d in glob.glob(os.path.join(sp, "nvidia", "*", "bin")):
            try:
                os.add_dll_directory(d)
            except Exception:
                pass
            os.environ["PATH"] = d + os.pathsep + os.environ.get("PATH", "")


def read_exact(stream, n):
    buf = bytearray()
    while len(buf) < n:
        chunk = stream.read(n - len(buf))
        if not chunk:
            return None
        buf.extend(chunk)
    return bytes(buf)


# Typical Whisper hallucinations on silence / music.
BLACKLIST = (
    "субтитры",
    "продолжение следует",
    "спасибо за просмотр",
    "редактор субтитров",
    "dimatorzok",
    "thanks for watching",
    "подписывайтесь на канал",
    "amara.org",
)


def is_garbage(text):
    t = text.strip().lower()
    if len(t) < 2:
        return True
    return any(b in t for b in BLACKLIST)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="medium")
    p.add_argument("--language", default="ru")
    p.add_argument("--device", default="cuda")
    p.add_argument("--prompt", default="")
    args = p.parse_args()

    add_cuda_dlls()
    import numpy as np
    from faster_whisper import WhisperModel

    lang = None if args.language in ("", "auto") else args.language
    t0 = time.time()
    device = args.device
    try:
        model = WhisperModel(args.model, device=device, compute_type="float16" if device == "cuda" else "int8")
        # Warm-up: forces CUDA libraries to load now instead of on the first phrase.
        list(model.transcribe(np.zeros(16000, dtype=np.float32), language=lang or "en")[0])
    except Exception as e:  # fall back to CPU
        out({"type": "log", "text": f"CUDA недоступна ({e}), переключаюсь на CPU"})
        device = "cpu"
        model = WhisperModel(args.model, device="cpu", compute_type="int8")
    out({"type": "ready", "device": device, "model": args.model, "load_s": round(time.time() - t0, 1)})

    stdin = sys.stdin.buffer
    while True:
        hdr = read_exact(stdin, 4)
        if hdr is None:
            break
        (n,) = struct.unpack("<I", hdr)
        data = read_exact(stdin, n * 4)
        if data is None:
            break
        audio = np.frombuffer(data, dtype=np.float32)
        t = time.time()
        try:
            segments, _info = model.transcribe(
                audio,
                language=lang,
                beam_size=3,
                vad_filter=True,
                condition_on_previous_text=False,
                initial_prompt=args.prompt or None,
            )
            parts = []
            for s in segments:
                if s.no_speech_prob > 0.6 and s.avg_logprob < -1.0:
                    continue
                if not is_garbage(s.text):
                    parts.append(s.text.strip())
            text = " ".join(parts).strip()
            if text:
                out({"type": "text", "text": text, "audio_s": round(len(audio) / 16000, 1),
                     "took_s": round(time.time() - t, 2)})
        except Exception as e:
            out({"type": "log", "text": f"ошибка распознавания: {e}"})


if __name__ == "__main__":
    main()
