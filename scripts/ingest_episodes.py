"""Replace the podcast index with whatever is in data/episodes.json.

Clears the 'podcast' namespace, then ingests episode by episode so
progress is incremental — a crash mid-run keeps every episode already
upserted, and Pinecone stats show progress live. Run after
scripts/fetch_episodes.py.

    .venv/bin/python scripts/ingest_episodes.py
"""

import asyncio
import json
import logging
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.podcast import NAMESPACE, PodcastIndex  # noqa: E402
from app.schemas import Episode  # noqa: E402

DATA = Path(__file__).resolve().parent.parent / "data" / "episodes.json"

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")


def log(msg: str) -> None:
    print(msg, flush=True)  # visible immediately even when redirected


async def main() -> int:
    episodes = [Episode(**e) for e in json.loads(DATA.read_text())]
    if not episodes:
        log("data/episodes.json is empty — run fetch_episodes.py first.")
        return 1

    idx = PodcastIndex()
    try:
        idx.index.delete(delete_all=True, namespace=NAMESPACE)
        log(f"cleared namespace '{NAMESPACE}'")
    except Exception as exc:  # namespace may not exist yet — fine
        log(f"(namespace clear skipped: {exc})")

    total = 0
    started = time.monotonic()
    for i, episode in enumerate(episodes, 1):
        count = await idx.ingest([episode])
        total += count
        elapsed = int(time.monotonic() - started)
        log(f"[{i}/{len(episodes)}] +{count} windows ({total} total, "
            f"{elapsed}s) — {episode.title[:60]}")

    log(f"DONE: {total} windows from {len(episodes)} episodes")
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
