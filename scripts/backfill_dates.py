"""Fill in each episode's publish date.

    python scripts/backfill_dates.py [--file data/episodes.json] [--force]

The Episode schema has carried `published_at` since the start but nothing
ever wrote it: the fetcher asks yt-dlp only for the title. Without a date
every chunk is timeless, so a view from a year ago and last week's
correction rank purely on wording and get averaged into something nobody
ever said.

A flat playlist listing reports upload_date as NA — the date only comes
from per-video metadata, so this is one lookup per episode. Captions are
never re-downloaded.

Idempotent: episodes that already have a date are skipped unless --force.
"""

import argparse
import json
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
YTDLP = str(ROOT / ".venv" / "bin" / "yt-dlp")


def upload_date(video_id: str, timeout: int = 60) -> str | None:
    """YYYY-MM-DD, or None if unavailable."""
    try:
        out = subprocess.run(
            [YTDLP, "--skip-download", "--no-warnings",
             "--print", "%(upload_date)s",
             f"https://www.youtube.com/watch?v={video_id}"],
            capture_output=True, text=True, timeout=timeout,
        )
    except subprocess.TimeoutExpired:
        return None
    raw = (out.stdout or "").strip().splitlines()
    raw = raw[-1].strip() if raw else ""
    if len(raw) != 8 or not raw.isdigit():      # yt-dlp gives YYYYMMDD or NA
        return None
    return f"{raw[:4]}-{raw[4:6]}-{raw[6:]}"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--file", type=Path, default=ROOT / "data" / "episodes.json")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    episodes = json.loads(args.file.read_text())
    todo = [e for e in episodes if args.force or not e.get("published_at")]
    print(f"{len(episodes)} episodes, {len(todo)} needing a date")

    filled = failed = 0
    for i, ep in enumerate(todo, 1):
        date = upload_date(ep["episode_id"])
        if date:
            ep["published_at"] = date
            filled += 1
            print(f"  [{i}/{len(todo)}] {date}  {ep['title'][:52]}")
        else:
            failed += 1
            print(f"  [{i}/{len(todo)}] ??????????  {ep['title'][:52]}  (no date)")
        # Write after every lookup: a run interrupted at episode 19 should
        # keep the first eighteen.
        args.file.write_text(json.dumps(episodes, ensure_ascii=False))

    have = sum(1 for e in episodes if e.get("published_at"))
    print(f"\n{filled} filled, {failed} failed — {have}/{len(episodes)} now dated")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
