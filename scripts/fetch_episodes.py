"""Pull real Threadguy captions from YouTube and convert them into the
Episode JSON the podcast search ingests.

Uses yt-dlp to download auto-generated VTT subtitles (no video download),
parses each caption cue into a {t, text} segment, dedupes YouTube's rolling
auto-caption overlap, and writes data/episodes.json.

    .venv/bin/python scripts/fetch_episodes.py VIDEO_ID [VIDEO_ID ...]
    .venv/bin/python scripts/fetch_episodes.py --latest 5   # newest N MB eps

If YouTube blocks the automated download (bot check / 429), either pass your
browser cookies:

    .venv/bin/python scripts/fetch_episodes.py --cookies chrome VIDEO_ID

or download the .vtt yourself (browser extension / online tool) and convert
the local files — no network needed:

    .venv/bin/python scripts/fetch_episodes.py --vtt ep9.vtt=VIDEO_ID ...

Then ingest:
    curl -X POST localhost:8100/v1/podcast/ingest \\
      -H 'content-type: application/json' -d @data/episodes.json
"""

import html
import json
import os
import re
import subprocess
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
YTDLP = str(ROOT / ".venv" / "bin" / "yt-dlp")
CHANNEL = "https://www.youtube.com/@notthreadguy/videos"
OUT = ROOT / "data" / "episodes.json"

_TS = re.compile(r"(\d{2}):(\d{2}):(\d{2})\.(\d{3})")
_TAG = re.compile(r"<[^>]+>")
_BLEEP = re.compile(r"\[\s*_+\s*\]")           # YouTube profanity bleep [ __ ]
_SPEAKER_MARK = re.compile(r"\s*>>+\s*")        # auto-caption speaker change
_WS = re.compile(r"\s+")


def _clean(text: str) -> str:
    text = html.unescape(text)
    text = _BLEEP.sub("[expletive]", text)
    text = _SPEAKER_MARK.sub(" ", text)
    return _WS.sub(" ", text).strip()


def _seconds(h: str, m: str, s: str, ms: str) -> float:
    return int(h) * 3600 + int(m) * 60 + int(s) + int(ms) / 1000


def _new_tail(prev_words: list[str], cue_words: list[str]) -> list[str]:
    """Return only the words in `cue_words` that aren't already covered by the
    overlap with the end of `prev_words`. YouTube auto-captions roll a window
    forward, so each cue repeats a prefix of already-emitted text; we find the
    largest k where the last k emitted words equal the first k cue words and
    keep only what follows."""
    max_overlap = min(len(prev_words), len(cue_words))
    for k in range(max_overlap, 0, -1):
        if prev_words[-k:] == cue_words[:k]:
            return cue_words[k:]
    return cue_words


def parse_vtt(vtt: str) -> list[dict]:
    """Parse a WebVTT file into deduped {t, text} segments, stripping YouTube's
    rolling-window overlap at word granularity."""
    segments: list[dict] = []
    emitted: list[str] = []  # trailing context for overlap detection
    for block in vtt.split("\n\n"):
        lines = block.strip().splitlines()
        ts_line = next((ln for ln in lines if "-->" in ln), None)
        if not ts_line:
            continue
        m = _TS.search(ts_line)
        if not m:
            continue
        start = _seconds(*m.groups())
        # The cue body is the lines AFTER the timestamp. (A WebVTT cue
        # identifier, if present, precedes the timestamp — never after it.)
        # Taking post-timestamp lines preserves fully-numeric spoken lines
        # like "2026" or "100" that a blanket isdigit() filter would drop —
        # which matters a lot for a finance podcast full of years and prices.
        ts_idx = lines.index(ts_line)
        body = _clean(" ".join(
            _TAG.sub("", ln) for ln in lines[ts_idx + 1:] if ln.strip()
        ))
        if not body:
            continue
        cue_words = body.split()
        new_words = _new_tail(emitted[-40:], cue_words)
        if not new_words:
            continue
        segments.append({"t": round(start, 2), "text": " ".join(new_words)})
        emitted.extend(new_words)
    return segments


def coalesce(segments: list[dict], min_gap: float = 6.0) -> list[dict]:
    """Merge tiny caption fragments into ~sentence-level segments so each
    carries a meaningful timestamp instead of two words."""
    out: list[dict] = []
    for seg in segments:
        if out and seg["t"] - out[-1]["t"] < min_gap:
            out[-1]["text"] = f'{out[-1]["text"]} {seg["text"]}'.strip()
        else:
            out.append(dict(seg))
    return out


def from_vtt_file(path: Path, video_id: str, title: str | None = None) -> dict | None:
    """Convert a local .vtt caption file into an Episode (no network)."""
    raw = parse_vtt(path.read_text(encoding="utf-8", errors="ignore"))
    segments = coalesce(raw)
    if not segments:
        print(f"  ! empty transcript in {path}")
        return None
    print(f"  ✓ {video_id}: {len(segments)} segments — {title or path.name}")
    return {
        "episode_id": video_id,
        "title": title or path.stem,
        "url": f"https://www.youtube.com/watch?v={video_id}",
        "platform": "youtube",
        "segments": segments,
    }


def fetch(
    video_id: str, cookies_browser: str | None = None, attempts: int = 3
) -> dict | None:
    url = f"https://www.youtube.com/watch?v={video_id}"
    # YouTube gates caption downloads behind a JS challenge + bot check.
    # --remote-components pulls yt-dlp's solver; --cookies-from-browser and a
    # JS runtime (deno, on PATH) get past the bot check. See README.
    cookie_args = ["--cookies-from-browser", cookies_browser] if cookies_browser else []
    common = ["--remote-components", "ejs:github", *cookie_args]
    env = {**os.environ, "PATH": f"/opt/homebrew/bin:{os.environ.get('PATH', '')}"}
    title = video_id
    for attempt in range(1, attempts + 1):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            # Single invocation: downloads captions AND prints the title, so we
            # only hit YouTube once per episode (two calls trips rate limits).
            proc = subprocess.run(
                [YTDLP, "--skip-download", "--write-auto-sub", "--write-sub",
                 "--sub-lang", "en", "--sub-format", "vtt", *common,
                 # --print alone implies simulate (nothing downloads!);
                 # --no-simulate makes it print the title AND write files.
                 "--print", "%(title)s", "--no-simulate",
                 "-o", str(tmp_path / "%(id)s.%(ext)s"), url],
                capture_output=True, text=True, check=False, env=env,
            )
            title = (proc.stdout.strip().splitlines() or [title])[0]
            vtts = list(tmp_path.glob("*.vtt"))
            if vtts:
                segments = coalesce(
                    parse_vtt(vtts[0].read_text(encoding="utf-8", errors="ignore"))
                )
                if segments:
                    print(f"  ✓ {video_id}: {len(segments)} segments — {title}")
                    return {
                        "episode_id": video_id, "title": title, "url": url,
                        "platform": "youtube", "segments": segments,
                    }
        if attempt < attempts:
            wait = 20 * attempt  # back off — YouTube throttles rapid pulls
            print(f"  … {video_id} throttled (try {attempt}/{attempts}), "
                  f"waiting {wait}s")
            time.sleep(wait)
    print(f"  ! no captions for {video_id} ({title}) after {attempts} tries")
    return None


def latest_ids(n: int, cookies_browser: str | None = None) -> list[str]:
    cookie_args = ["--cookies-from-browser", cookies_browser] if cookies_browser else []
    env = {**os.environ, "PATH": f"/opt/homebrew/bin:{os.environ.get('PATH', '')}"}
    out = subprocess.run(
        [YTDLP, "--flat-playlist", "--print", "%(id)s",
         "--remote-components", "ejs:github", *cookie_args, CHANNEL],
        capture_output=True, text=True, check=False, env=env,
    )
    ids = [ln.strip() for ln in out.stdout.splitlines() if ln.strip()][:n]
    if not ids:
        err = (out.stderr.strip().splitlines() or ["(no stderr)"])[-1]
        print(f"  ! channel listing failed — {err[:140]}")
    return ids


def main(argv: list[str]) -> int:
    if not argv:
        print(__doc__)
        return 1

    # Local-file mode: --vtt path=VIDEO_ID [path=VIDEO_ID ...]
    if argv[0] == "--vtt":
        episodes = []
        for spec in argv[1:]:
            path_str, _, vid = spec.partition("=")
            ep = from_vtt_file(Path(path_str), vid or Path(path_str).stem)
            if ep:
                episodes.append(ep)
        OUT.write_text(json.dumps(episodes, indent=2, ensure_ascii=False))
        total = sum(len(e["segments"]) for e in episodes)
        print(f"\nWrote {len(episodes)} episodes ({total} segments) -> {OUT}")
        return 0

    cookies = None
    if argv and argv[0] == "--cookies":
        cookies, argv = argv[1], argv[2:]

    if argv and argv[0] == "--latest":
        ids = latest_ids(int(argv[1]) if len(argv) > 1 else 3, cookies)
    else:
        ids = argv

    # Resume support: keep episodes already fetched in a previous run.
    episodes: list[dict] = []
    if OUT.exists():
        try:
            episodes = json.loads(OUT.read_text())
        except json.JSONDecodeError:
            episodes = []
    done = {e["episode_id"] for e in episodes}
    ids = [v for v in ids if v not in done]

    print(f"Fetching {len(ids)} episode(s) ({len(done)} already cached)...")
    for i, vid in enumerate(ids):
        ep = fetch(vid, cookies)
        if ep:
            episodes.append(ep)
            # Incremental save — a long run that dies keeps its progress.
            OUT.write_text(json.dumps(episodes, indent=2, ensure_ascii=False))
        if i < len(ids) - 1:
            time.sleep(12)  # be polite; rapid pulls trip YouTube throttling

    OUT.write_text(json.dumps(episodes, indent=2, ensure_ascii=False))
    total = sum(len(e["segments"]) for e in episodes)
    print(f"\nWrote {len(episodes)} episodes ({total} segments) -> {OUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
