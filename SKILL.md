---
name: shorts-clipper
description: Turn a long-form video (podcast, interview, lecture, stream, gameplay, vlog, webinar, motivational talk) into many finished, self-contained 9:16 Shorts (about 30-90 s) with word-by-word captions, punch-in zooms and subtle SFX, plus titles, descriptions and hashtags. Use when the user gives a video file and asks for clips, shorts, reels, highlights or "clipping". Fully autonomous; do not ask questions.
---

# shorts-clipper

Goal: the user monetizes Shorts (main platform: **YouTube Shorts**; also TikTok and Reels) in top-tier English-speaking markets (US, UK, AU, CA). Every clip must earn views: strong hook, no dead air, clear payoff, readable captions. Quality beats count, but find **every** moment that clears the bar.

## Operating rules
- **Do not ask the user questions.** They provide a video path; you do everything. Assume YouTube Shorts unless they name another platform. If something is ambiguous, pick the sensible default, note it in `report.md`, and keep going.
- Free tools only. Everything runs locally: ffmpeg, faster-whisper, OpenCV.
- Language: English only. Non-English audio: stop and say so in one line.
- Settings live in `config.json` next to this file. Read it first; the user may have edited it.
- Never touch the original video. Work in `out/<video-name>/`.

## 0. Setup (once per machine)
Check, install only what is missing:
```
ffmpeg -version && ffprobe -version     # Windows: winget install Gyan.FFmpeg   (or choco install ffmpeg)
python -m pip install faster-whisper opencv-python-headless numpy
```
- Use `opencv-python-headless` 4.x (it ships the face cascade the tracker needs). If the render log says "face cascade not found", reinstall it that way.
- The user's laptop may be slow and short on disk. Whisper defaults to `small.en` on CPU (`large-v3` only if a CUDA GPU exists). For a 2-hour video on CPU, transcription can take a long while: run it in the background, show progress, and use `tiny.en`/`base.en` only if the user asks for speed. Check free disk space (need roughly 3x the video size).
- Run scripts with `python` (Windows) or `python3`. Scripts use `pathlib`; do not hand-write bash-only paths on Windows.
- Captions font: Arial Black is used by default (present on Windows). For a stronger look, try once to put an OFL font (Montserrat ExtraBold or Anton) in `assets/fonts/` and set `"font"` in config to its family name. If the download fails, keep the default.

## 1. Transcribe
```
python scripts/transcribe.py "<video>" --out out/<name>/work
```
Produces `transcript.json` (word timestamps), `transcript.txt` (`[H:MM:SS] text` lines) and `energy.json` (per-second loudness plus loudest moments).

## 2. Find the moments (this is where views are won)
Read the whole `transcript.txt`, in order, in ~15-minute windows (use subagents for windows if you have them; otherwise go sequentially). Keep a running `out/<name>/work/candidates.json`. Then do a global second pass to rank, dedupe, and cut weak ones.

**First, classify the video** and adapt what "good" means:
| Type | What wins |
|---|---|
| Business / self-improvement / motivational / lectures | A single actionable idea, framework, or hard truth; strong opening claim; numbers; contrarian takes. Emotional peaks for motivational talks. Don't force controversy where there is none. |
| Interviews / podcasts | Surprising claims, confessions, stories with a twist, disagreements, "I've never said this before" moments, sharp one-liners. |
| Streamers / reactions / IRL (e.g. high-energy creators) | Peaks of energy and reaction: use `energy.json` peaks as leads, then confirm in the transcript. Funny, chaotic, shocking, wholesome moments; short setup plus big payoff; keep the original audio punchy. |
| Gameplay | Clutch plays, fails, jump scares, rage/hype reactions, funny commentary. Usually blur layout. |
| Vlogs / webinars | The one standout story, tip, or reveal per section. |

**Score each candidate 0-10** on: hook (first 3 s), self-contained, payoff/completeness, emotion/novelty, shareability (quotable, comment-bait). Keep a clip only if hook >= 7 **and** self-contained >= 8 **and** average >= 7. Never pad to hit a number. As a rough guide, strong talking content yields about 0.5-1.5 keepers per 10 minutes; a weak video may yield few. Report what you rejected and why in one line each.

**Self-contained is non-negotiable.** A stranger must understand the clip with zero context:
- Start on a complete thought that states or implies the topic. Never start on "and", "so", "but", "he/she/that/this" with no referent, or mid-sentence.
- If the setup is far away, stitch it: `segments: [[setup_start, setup_end], [payoff_start, payoff_end]]` (non-consecutive parts are allowed when they make sense together, e.g. question then answer). Keep stitches to 2-3 parts.
- End right after the payoff or punchline, on a completed sentence. No "thanks for having me", no trailing filler, no cliffhanger that needs the next minute.
- Trim the first words if they are throat-clearing ("so, um, basically"). Start at the first strong word of the hook; keep audio roughly 0.1 s of lead-in.

**Length:** sweet spot 30-90 s. Allowed 20-120 s (config `hard_range_seconds`). Go longer only if cutting would hurt the story, and never past ~3 min. Prefer a tight 35 s over a padded 80 s.

**Always skip:** sponsor reads and ad breaks ("this video is sponsored", "use code", "link in the description", "brought to you by"), housekeeping ("like and subscribe"), intros/outros, long silences, and personal information such as phone numbers or home addresses. Profanity, politics and religion are allowed if the moment is strong; do not add anything defamatory or misleading.

**Snap boundaries to word timestamps** from `transcript.json` (start at a word's `s`, minus ~0.1 s; end at a word's `e`, plus ~0.25 s).

## 3. Write one spec per clip
`out/<name>/work/specs/NN_slug.json`:
```json
{
  "id": "01_why-most-people-stay-broke",
  "segments": [[612.4, 655.9]],
  "hook_text": "Most people stay broke for ONE reason",
  "layout": "auto"
}
```
- `hook_text` (optional but recommended): 4-9 words shown in a box for the first ~3 s. Honest to the content, curiosity-driven. Skip it if the first spoken line is already a great hook.
- `layout`: `auto` (default: face-tracking crop if a face is found, blurred-background letterbox otherwise), or force `face`, `center`, `blur`. Gameplay/screen content: `blur`. Two-person wide shots: face tracking follows the larger face; if it follows the wrong person, add `"focus": [[clip_time_s, x_0_to_1], ...]`.
- `zooms` (optional): clip-time seconds for punch-ins. Default is automatic: a zoom on stitched-cut joins and every ~5 s on sentence starts. For emotional beats, add your own.

## 4. Render
```
python scripts/render_clip.py "<video>" out/<name>/work/specs/NN_slug.json --transcript out/<name>/work/transcript.json --out out/<name>/clips
```
What it does (do not re-implement): cuts frame-accurately, crops to 1080x1920, follows the speaker's face with smoothing and snaps on shot changes, adds slow drift plus eased punch-in zooms, mixes quiet whoosh SFX on zooms/joins, normalizes loudness (-14 LUFS), burns word-by-word captions (3 words max, current word highlighted yellow, pop-in) placed above the Shorts UI zone, and encodes H.264/AAC with faststart. Render clips one by one; if a render fails, read the error, fix the spec or environment, retry once, and continue with the rest.

Do not add logos, watermarks, intros, outros or brand colors (the user does not want them). Do not add copyrighted music. If you want background music, use only royalty-free tracks the user placed in `assets/music/`.

## 5. QA every clip before delivering (mandatory)
1. `ffprobe`: 1080x1920, has audio, duration within range.
2. Extract frames at ~1 s, middle, and ~2 s before the end (`ffmpeg -ss T -i clip.mp4 -frames:v 1 f.png`) and look at them: face not cut off, captions fully visible, not overlapping the hook box, not clipped at the edges, readable.
3. Check the first and last 3 s of the transcript text for the clip: no dangling start or end, no sponsor text.
4. If a clip fails, fix the spec (different boundaries, layout, `focus`) and re-render. Drop the clip if it cannot be made good.

## 6. Metadata and report
For each clip write to `clips.json` and `report.md` (ranked best to worst, with score):
- `title`: <= 70 chars, front-load the hook, honest to the clip (no bait that the clip does not deliver). Vary patterns: question, bold claim, "X vs Y", number.
- `description`: 1-2 lines of context plus the source credit if the video title/channel is known (e.g. "Clip from <title> by <channel>").
- `hashtags`: 3-5 relevant, always including `#shorts` for YouTube.
- Source timestamps, duration, score, and a one-line reason it should perform.
- Platform notes: YouTube Shorts is the default. If the user names TikTok/Reels, keep the same file and adapt the title/hashtags only.

`report.md` also lists rejected moments, settings used, and any problems. Final message to the user: where the clips are, how many, the top 3 by score, and any warnings. Keep it short.

## 7. Rights and monetization (state once in report.md, briefly, then proceed)
Clipping other people's content can be copyright-claimed, and YouTube may refuse or limit monetization on reused content that adds little original value. Adding captions, hooks and editing helps, but permission from the creator is what protects the channel. Mention, in two lines, that clipping programs or written permission from the creator are the safest route, and record any permission note the user provides. Do not lecture and do not refuse because of this.

## Output layout
```
out/<name>/
  clips/NN_slug.mp4       finished shorts
  clips.json              metadata for all clips
  report.md               ranked list, rejected moments, notes
  work/                   transcript, energy, candidates, specs (safe to delete)
```
