#!/usr/bin/env python3
"""Render one vertical 1080x1920 short from a long video.

Usage:
  python render_clip.py VIDEO SPEC.json --transcript WORKDIR/transcript.json --out OUTDIR

SPEC.json (one clip):
{
  "id": "01_why-most-people-stay-broke",
  "segments": [[612.4, 655.9], [1201.0, 1219.5]],   # source-time ranges, stitched in order
  "hook_text": "Most people stay broke for ONE reason",   # optional on-screen hook (first ~3 s)
  "layout": "auto",            # auto | face | center | blur   (auto: face-tracking if faces found, else blurred-background)
  "zooms": [4.2, 9.8],         # optional clip-time seconds for punch-ins (default: automatic)
  "focus": [[0, 0.35], [12, 0.7]]   # optional manual horizontal focus: [clip_time, x 0..1] (overrides face tracking)
}

Output: OUTDIR/<id>.mp4 (+ OUTDIR/_work/<id>/ intermediates, safe to delete)
Needs: ffmpeg, ffprobe, opencv-python(-headless), numpy.
"""
import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE))
import make_captions as mc  # noqa: E402

OUT_W, OUT_H = 1080, 1920
FPS = 30
EASE = 0.18


def load_config():
    cfg = {
        "font": "Arial Black", "fontsize": 84, "max_height": 1080, "crf": 18,
        "preset": "veryfast", "sfx": True, "zoom": True, "layout": "auto", "sfx_volume": 0.30,
    }
    p = ROOT / "config.json"
    if p.exists():
        cfg.update(json.loads(p.read_text(encoding="utf-8")))
    return cfg


def run(cmd, cwd=None):
    r = subprocess.run([str(c) for c in cmd], cwd=cwd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if r.returncode != 0:
        raise RuntimeError("command failed: " + " ".join(map(str, cmd)) + "\n" + r.stderr[-1800:])
    return r


def probe_height(video):
    r = run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries", "stream=height", "-of", "json", video])
    return int(json.loads(r.stdout)["streams"][0]["height"])


# ---------------------------------------------------------------- cutting
def cut_segments(video, segments, work, cfg):
    h = min(probe_height(video), int(cfg["max_height"]))
    h -= h % 2
    parts = []
    for i, (s, e) in enumerate(segments):
        d = e - s
        if d < 0.15:
            continue
        f = work / f"seg_{i}.mp4"
        run(["ffmpeg", "-y", "-ss", f"{s:.3f}", "-i", video, "-t", f"{d:.3f}",
             "-vf", f"fps={FPS},scale=-2:{h}",
             "-af", f"afade=t=in:d=0.03,afade=t=out:st={max(d - 0.03, 0):.3f}:d=0.03,aresample=48000",
             "-ac", "2", "-c:v", "libx264", "-preset", "veryfast", "-crf", "16", "-pix_fmt", "yuv420p",
             "-c:a", "aac", "-b:a", "192k", f.name], cwd=work)
        parts.append((f.name, d))
    if not parts:
        raise RuntimeError("no usable segments")
    raw = work / "clip_raw.mp4"
    if len(parts) == 1:
        shutil.copy(work / parts[0][0], raw)
    else:
        (work / "list.txt").write_text("".join(f"file '{n}'\n" for n, _ in parts), encoding="utf-8")
        run(["ffmpeg", "-y", "-f", "concat", "-safe", "0", "-i", "list.txt", "-c", "copy", "clip_raw.mp4"], cwd=work)
    offsets, t = [], 0.0
    for _, d in parts:
        offsets.append(t)
        t += d
    return raw, offsets, t


# ---------------------------------------------------------------- face tracking
class Faces:
    def __init__(self):
        name = "haarcascade_frontalface_default.xml"
        paths = []
        try:
            paths.append(Path(cv2.data.haarcascades) / name)
        except Exception:
            pass
        paths += list(Path(cv2.__file__).resolve().parent.rglob(name))
        self.cc = None
        for p in paths:
            cc = cv2.CascadeClassifier(str(p))
            if p.exists() and not cc.empty():
                self.cc = cc
                break
        if self.cc is None:
            print("[render] WARNING: face cascade not found (pip install opencv-python-headless); "
                  "falling back to center/blur layouts", file=sys.stderr)

    def detect(self, frame):
        if self.cc is None:
            return []
        h, w = frame.shape[:2]
        sc = 640.0 / w if w > 640 else 1.0
        g = cv2.cvtColor(cv2.resize(frame, (int(w * sc), int(h * sc))), cv2.COLOR_BGR2GRAY)
        g = cv2.equalizeHist(g)
        found = self.cc.detectMultiScale(g, 1.1, 6, minSize=(28, 28))
        return [((x + fw / 2) / sc, (y + fh / 2) / sc, fw * fh / (sc * sc)) for (x, y, fw, fh) in found]


class Cam:
    """Smoothed camera centre that follows the main speaker's face."""

    def __init__(self, W, H):
        self.W, self.H = W, H
        self.x = self.tx = W / 2
        self.y = self.ty = H / 2
        self.cur = None
        self.cnt = 0

    def update(self, faces, snap):
        if snap:
            self.cur, self.cnt = None, 0
        if not faces:
            if snap:
                self.tx, self.ty = self.W / 2, self.H / 2
                self.x, self.y = self.tx, self.ty
            return
        big = max(faces, key=lambda f: f[2])
        if self.cur is None:
            self.cur = big
        else:
            near = min(faces, key=lambda f: abs(f[0] - self.cur[0]))
            if near is not big and big[2] > 1.5 * near[2]:
                self.cnt += 1
                if self.cnt >= 3:
                    self.cur, self.cnt = big, 0
                else:
                    self.cur = near
            else:
                self.cnt, self.cur = 0, near
        fx, fy = self.cur[0], self.cur[1]
        if snap or abs(fx - self.tx) > 0.04 * self.W:
            self.tx = fx
        if snap or abs(fy - self.ty) > 0.06 * self.H:
            self.ty = fy
        if snap:
            self.x, self.y = self.tx, self.ty

    def step(self):
        self.x += (self.tx - self.x) * 0.10
        self.y += (self.ty - self.y) * 0.10


def decide_layout(raw, faces):
    cap = cv2.VideoCapture(str(raw))
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    hits = tot = 0
    for k in range(12):
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(n * (k + 0.5) / 12))
        ok, fr = cap.read()
        if ok:
            tot += 1
            hits += 1 if faces.detect(fr) else 0
    cap.release()
    return "face" if tot and hits / tot >= 0.4 else "blur"


# ---------------------------------------------------------------- zoom + sfx plan
def smooth_zoom(t, events):
    lvl = 1.0
    for et, el in events:
        if t < et:
            break
        if t < et + EASE:
            u = (t - et) / EASE
            u = u * u * (3 - 2 * u)
            return lvl + (el - lvl) * u
        lvl = el
    return lvl


def plan_zoom(words, boundaries, dur, user_times, enabled):
    if not enabled:
        return []
    if user_times:
        times = sorted(float(t) for t in user_times)
    else:
        times = list(boundaries)
        last = -9.0
        for i, w in enumerate(words[:-1]):
            if w["w"][-1:] in ".?!":
                t = words[i + 1]["s"]
                if t - last >= 4.5 and t > 1.5 and t < dur - 1.5 and all(abs(t - b) > 1.0 for b in boundaries):
                    times.append(t)
                    last = t
        times = sorted(set(times))
    cycle = [1.15, 1.0, 1.10, 1.0]
    return [(t, cycle[i % len(cycle)]) for i, t in enumerate(times) if t < dur - 0.6]


def ensure_sfx():
    d = ROOT / "assets" / "sfx"
    d.mkdir(parents=True, exist_ok=True)
    w = d / "whoosh.wav"
    if not w.exists():
        run(["ffmpeg", "-y", "-f", "lavfi", "-i", "anoisesrc=d=0.45:c=pink:a=0.6", "-af",
             "highpass=f=250,lowpass=f=5500,afade=t=in:st=0:d=0.22,afade=t=out:st=0.22:d=0.23",
             "-ar", "48000", "-ac", "2", w])
    return w


# ---------------------------------------------------------------- video pass
def render_video(raw, out_silent, layout, zoom_events, focus, dur, cfg):
    cap = cv2.VideoCapture(str(raw))
    W, H = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    faces = Faces()
    if layout == "auto":
        layout = cfg["layout"] if cfg["layout"] != "auto" else decide_layout(raw, faces)
        cap.release()
        cap = cv2.VideoCapture(str(raw))
    print(f"[render] layout={layout} source={W}x{H}")

    if W / H >= 9 / 16:
        base_h, base_w = float(H), H * 9 / 16
    else:
        base_w, base_h = float(W), W * 16 / 9
    cam = Cam(W, H)
    step = max(1, int(FPS * 0.4))
    prev_small = None

    enc = subprocess.Popen(
        ["ffmpeg", "-y", "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{OUT_W}x{OUT_H}", "-r", str(FPS),
         "-i", "-", "-an", "-c:v", "libx264", "-preset", "veryfast", "-crf", "14", "-pix_fmt", "yuv420p", str(out_silent)],
        stdin=subprocess.PIPE, stderr=subprocess.DEVNULL)

    i = 0
    while True:
        ok, fr = cap.read()
        if not ok:
            break
        t = i / FPS
        z = smooth_zoom(t, zoom_events) * (1 + 0.04 * min(t / max(dur, 1), 1))

        if layout == "blur":
            cw = int(H * 9 / 16) if W / H >= 9 / 16 else W
            x0 = max(0, (W - cw) // 2)
            bg = cv2.resize(fr[:, x0:x0 + cw], (270, 480))
            bg = cv2.GaussianBlur(bg, (0, 0), 9)
            bg = cv2.resize(bg, (OUT_W, OUT_H), interpolation=cv2.INTER_LINEAR)
            bg = (bg * 0.55).astype(np.uint8)
            zw, zh = W / z, H / z
            cx, cy = W / 2, H / 2
            x1, y1 = int(cx - zw / 2), int(cy - zh / 2)
            fg = fr[y1:y1 + int(zh), x1:x1 + int(zw)]
            fh = int(OUT_W * fg.shape[0] / fg.shape[1])
            fg = cv2.resize(fg, (OUT_W, fh), interpolation=cv2.INTER_CUBIC)
            top = max(0, (OUT_H - fh) // 2 - 80)
            frame = bg
            frame[top:top + fh] = fg[: OUT_H - top]
        else:
            if layout == "face":
                snap = False
                tiny = cv2.resize(cv2.cvtColor(fr, cv2.COLOR_BGR2GRAY), (32, 18)).astype(np.int16)
                if prev_small is not None and np.abs(tiny - prev_small).mean() > 22:
                    snap = True
                prev_small = tiny
                if focus:
                    xn = focus[0][1]
                    for ft, fx in focus:
                        if t >= ft:
                            xn = fx
                    cam.tx = cam.x = float(xn) * W
                elif i % step == 0 or snap:
                    cam.update(faces.detect(fr), snap)
                cam.step()
                cx, cy = cam.x, cam.y
            else:
                cx, cy = W / 2, H / 2
            cw, ch = base_w / z, base_h / z
            cy_eff = cy + (0.08 * ch if layout == "face" and z > 1.0 else 0)
            x1 = int(min(max(cx - cw / 2, 0), W - cw))
            y1 = int(min(max(cy_eff - ch / 2, 0), H - ch))
            crop = fr[y1:y1 + int(ch), x1:x1 + int(cw)]
            frame = cv2.resize(crop, (OUT_W, OUT_H), interpolation=cv2.INTER_CUBIC)

        enc.stdin.write(frame.tobytes())
        i += 1
    cap.release()
    enc.stdin.close()
    enc.wait()
    return layout


# ---------------------------------------------------------------- audio mix
def mix_audio(raw, zoom_events, cfg, work):
    out = work / "mix.m4a"
    sfx = []
    last = -9.0
    prev = 1.0
    if cfg["sfx"]:
        for t, lvl in zoom_events:
            if lvl > prev and t - last >= 3.0:
                sfx.append(max(0.0, t - 0.08))
                last = t
            prev = lvl
    if not sfx:
        run(["ffmpeg", "-y", "-i", raw, "-vn", "-af", "loudnorm=I=-14:TP=-1.5:LRA=11", "-c:a", "aac", "-b:a", "192k", out])
        return out
    w = ensure_sfx()
    cmd = ["ffmpeg", "-y", "-i", raw]
    for _ in sfx:
        cmd += ["-i", w]
    parts, labels = [], ["[0:a]"]
    for k, t in enumerate(sfx, start=1):
        ms = int(t * 1000)
        parts.append(f"[{k}:a]adelay={ms}|{ms},volume={cfg['sfx_volume']}[s{k}]")
        labels.append(f"[s{k}]")
    fc = ";".join(parts) + ";" + "".join(labels) + f"amix=inputs={len(labels)}:normalize=0:duration=first,loudnorm=I=-14:TP=-1.5:LRA=11[a]"
    try:
        run(cmd + ["-filter_complex", fc, "-map", "[a]", "-c:a", "aac", "-b:a", "192k", out])
    except RuntimeError:
        fc = fc.replace(":normalize=0", "")
        run(cmd + ["-filter_complex", fc, "-map", "[a]", "-c:a", "aac", "-b:a", "192k", out])
    return out


# ---------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("video")
    ap.add_argument("spec")
    ap.add_argument("--transcript", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--keep-work", action="store_true")
    a = ap.parse_args()

    cfg = load_config()
    a.video = str(Path(a.video).resolve())  # ffmpeg runs with cwd=work, so paths must be absolute
    spec = json.loads(Path(a.spec).read_text(encoding="utf-8"))
    tr = json.loads(Path(a.transcript).read_text(encoding="utf-8"))
    out_dir = Path(a.out).resolve()
    work = out_dir / "_work" / spec["id"]
    work.mkdir(parents=True, exist_ok=True)

    raw, offsets, dur = cut_segments(a.video, spec["segments"], work, cfg)
    words = mc.clip_words(tr, spec["segments"])
    boundaries = [o for o in offsets[1:]]
    zoom_events = plan_zoom(words, boundaries, dur, spec.get("zooms"), cfg["zoom"])

    silent = work / "silent.mp4"
    layout = render_video(raw, silent, spec.get("layout", "auto"), zoom_events, spec.get("focus"), dur, cfg)

    margin_v = 520 if layout == "blur" else 600
    (work / "captions.ass").write_text(
        mc.build_ass(words, spec.get("hook_text"), dur, margin_v, cfg["font"], cfg["fontsize"]), encoding="utf-8")

    mix = mix_audio(raw, zoom_events, cfg, work)

    vf = "ass=captions.ass"
    fonts = ROOT / "assets" / "fonts"
    if fonts.exists() and any(fonts.iterdir()):
        shutil.copytree(fonts, work / "fonts", dirs_exist_ok=True)
        vf += ":fontsdir=fonts"
    vf += ",unsharp=5:5:0.5"
    final = out_dir / f"{spec['id']}.mp4"
    # cwd=work with relative filenames avoids Windows path-escaping problems in the ass filter
    run(["ffmpeg", "-y", "-i", "silent.mp4", "-i", "mix.m4a", "-vf", vf, "-c:v", "libx264",
         "-preset", cfg["preset"], "-crf", str(cfg["crf"]), "-pix_fmt", "yuv420p", "-c:a", "aac", "-b:a", "192k",
         "-movflags", "+faststart", "-shortest", str(final.resolve())], cwd=work)
    if not a.keep_work:
        for f in work.glob("seg_*.mp4"):
            f.unlink()
        for f in ("clip_raw.mp4", "silent.mp4", "list.txt"):
            (work / f).unlink(missing_ok=True)
    print(f"[render] done: {final} ({dur:.1f}s, layout={layout}, zooms={len(zoom_events)})")


if __name__ == "__main__":
    main()
