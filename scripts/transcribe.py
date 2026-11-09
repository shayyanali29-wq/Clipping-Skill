#!/usr/bin/env python3
"""Transcribe a long video with faster-whisper (free, local) and measure audio energy.

Usage: python transcribe.py VIDEO --out WORKDIR [--model auto|small.en|medium.en|large-v3] [--language en]

Writes into WORKDIR:
  audio.wav        16 kHz mono audio
  transcript.json  {"segments":[{"start","end","text","words":[{"w","s","e"}]}]}
  transcript.txt   human/agent-readable, one line per segment: [H:MM:SS] text
  energy.json      {"rms":[per-second loudness], "peaks":[seconds of loudest moments]}
"""
import argparse
import json
import subprocess
import sys
import wave
from pathlib import Path


def fmt(t):
    t = int(t)
    return f"{t // 3600}:{t % 3600 // 60:02d}:{t % 60:02d}"


def energy(wav_path, out_path):
    import numpy as np

    rms = []
    with wave.open(str(wav_path)) as wf:
        sr = wf.getframerate()
        while True:
            raw = wf.readframes(sr)  # 1 second at a time (low memory)
            if len(raw) < sr:  # drop the partial last second
                break
            x = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
            rms.append(float(np.sqrt((x * x).mean())))
    if not rms:
        out_path.write_text(json.dumps({"rms": [], "peaks": []}))
        return
    arr = np.array(rms)
    # smooth over 3 s, then take the loudest, well-separated seconds
    sm = np.convolve(arr, np.ones(3) / 3, mode="same")
    order = np.argsort(-sm)
    peaks = []
    for i in order:
        if all(abs(int(i) - p) > 20 for p in peaks):
            peaks.append(int(i))
        if len(peaks) >= 60:
            break
    out_path.write_text(json.dumps({"rms": [round(v, 4) for v in rms], "peaks": sorted(peaks)}))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("video")
    ap.add_argument("--out", required=True)
    ap.add_argument("--model", default="auto")
    ap.add_argument("--language", default="en")
    a = ap.parse_args()

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    wav = out / "audio.wav"
    if not wav.exists():
        subprocess.run(
            ["ffmpeg", "-y", "-i", a.video, "-vn", "-ac", "1", "-ar", "16000", "-c:a", "pcm_s16le", str(wav)],
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )

    energy(wav, out / "energy.json")

    from faster_whisper import WhisperModel

    device, compute = "cpu", "int8"
    try:
        import ctranslate2

        if ctranslate2.get_cuda_device_count() > 0:
            device, compute = "cuda", "float16"
    except Exception:
        pass
    name = a.model
    if name == "auto":
        name = "large-v3" if device == "cuda" else "small.en"
    print(f"[transcribe] model={name} device={device}", file=sys.stderr)

    model = WhisperModel(name, device=device, compute_type=compute)
    segs_iter, info = model.transcribe(
        str(wav),
        language=a.language,
        word_timestamps=True,
        vad_filter=True,
        beam_size=5 if device == "cuda" else 1,
        condition_on_previous_text=False,
    )
    total = info.duration or 1
    segments = []
    for s in segs_iter:
        words = [{"w": w.word.strip(), "s": round(w.start, 3), "e": round(w.end, 3)} for w in (s.words or [])]
        segments.append({"start": round(s.start, 3), "end": round(s.end, 3), "text": s.text.strip(), "words": words})
        print(f"[transcribe] {s.end / total * 100:5.1f}%", file=sys.stderr, end="\r")
    print("", file=sys.stderr)

    (out / "transcript.json").write_text(json.dumps({"duration": total, "segments": segments}))
    lines = [f"[{fmt(s['start'])}] {s['text']}" for s in segments]
    (out / "transcript.txt").write_text("\n".join(lines), encoding="utf-8")
    print(f"[transcribe] done: {len(segments)} segments, {fmt(total)} of audio")


if __name__ == "__main__":
    main()
