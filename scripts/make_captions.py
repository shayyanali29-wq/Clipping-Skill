#!/usr/bin/env python3
"""Build word-by-word "pop" captions (ASS subtitles) for a vertical short.

Library use (from render_clip.py):
    words = clip_words(transcript, segments)            # clip-time words
    ass   = build_ass(words, hook_text, duration, margin_v, font, fontsize)

CLI use:
    python make_captions.py transcript.json spec.json out.ass [--margin-v 600]
"""
import argparse
import json
import re
import textwrap

YELLOW = r"&H0000FFFF&"  # ASS colours are BGR: this is yellow
WHITE = r"&H00FFFFFF&"


def clip_words(transcript, segments):
    """Return [{'s','e','w'}] in clip time for the given source [start,end] segments."""
    words_all = [w for seg in transcript["segments"] for w in seg.get("words", [])]
    out, offset = [], 0.0
    for s, e in segments:
        for w in words_all:
            if w["s"] >= s - 0.05 and w["e"] <= e + 0.15:
                ws = offset + max(0.0, w["s"] - s)
                we = offset + min(e - s, w["e"] - s)
                if we - ws < 0.05:
                    we = ws + 0.05
                txt = w["w"].strip()
                if txt:
                    out.append({"s": ws, "e": we, "w": txt})
        offset += e - s
    return out


def _tc(t):
    t = max(0.0, t)
    cs = int(round(t * 100))
    h, cs = divmod(cs, 360000)
    m, cs = divmod(cs, 6000)
    s, cs = divmod(cs, 100)
    return f"{h}:{m:02d}:{s:02d}.{cs:02d}"


def _clean(w):
    w = re.sub(r"[.,;:]+$", "", w)  # keep ? ! and apostrophes, drop other trailing punctuation
    w = w.replace("{", "").replace("}", "").replace("\\", "")
    return w.upper()


def group_words(words, max_words=3, max_chars=17, gap=0.6):
    groups, cur = [], []
    for w in words:
        if cur:
            chars = sum(len(x["w"]) + 1 for x in cur) + len(w["w"])
            if len(cur) >= max_words or chars > max_chars or w["s"] - cur[-1]["e"] > gap:
                groups.append(cur)
                cur = []
        cur.append(w)
        if re.search(r"[.?!,;:]$", w["w"]):
            groups.append(cur)
            cur = []
    if cur:
        groups.append(cur)
    return groups


def build_ass(words, hook_text=None, duration=None, margin_v=600, font="Arial Black", fontsize=84):
    head = f"""[Script Info]
ScriptType: v4.00+
PlayResX: 1080
PlayResY: 1920
WrapStyle: 2
ScaledBorderAndShadow: yes

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: Cap,{font},{fontsize},&H00FFFFFF,&H000000FF,&H00000000,&H64000000,-1,0,0,0,100,100,0,0,1,8,3,2,60,60,{margin_v},1
Style: Hook,{font},60,&H00FFFFFF,&H000000FF,&H00141414,&H00000000,-1,0,0,0,100,100,0,0,3,18,0,8,90,90,300,1

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
"""
    lines = []
    if hook_text:
        txt = textwrap.fill(hook_text.strip().upper(), 24).replace("\n", r"\N")
        end = min(3.2, duration) if duration else 3.2
        lines.append(f"Dialogue: 1,{_tc(0)},{_tc(end)},Hook,,0,0,0,,{{\\fad(120,200)}}{txt}")

    groups = group_words(words)
    for gi, g in enumerate(groups):
        next_start = groups[gi + 1][0]["s"] if gi + 1 < len(groups) else (duration or g[-1]["e"] + 0.5)
        for i, w in enumerate(g):
            start = w["s"]
            end = g[i + 1]["s"] if i + 1 < len(g) else min(w["e"] + 0.3, next_start)
            if end <= start:
                end = start + 0.08
            parts = []
            for j, x in enumerate(g):
                color = YELLOW if j == i else WHITE
                parts.append(f"{{\\c{color}}}{_clean(x['w'])}")
            pop = r"{\fscx88\fscy88\t(0,90,\fscx100\fscy100)}" if i == 0 else ""
            lines.append(f"Dialogue: 0,{_tc(start)},{_tc(end)},Cap,,0,0,0,,{pop}{' '.join(parts)}")
    return head + "\n".join(lines) + "\n"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("transcript")
    ap.add_argument("spec")
    ap.add_argument("out")
    ap.add_argument("--margin-v", type=int, default=600)
    ap.add_argument("--font", default="Arial Black")
    ap.add_argument("--fontsize", type=int, default=84)
    a = ap.parse_args()
    tr = json.load(open(a.transcript, encoding="utf-8"))
    spec = json.load(open(a.spec, encoding="utf-8"))
    words = clip_words(tr, spec["segments"])
    dur = sum(e - s for s, e in spec["segments"])
    open(a.out, "w", encoding="utf-8").write(
        build_ass(words, spec.get("hook_text"), dur, a.margin_v, a.font, a.fontsize)
    )


if __name__ == "__main__":
    main()
