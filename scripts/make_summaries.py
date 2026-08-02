"""Generate a summary for every indexed episode.

    .venv/bin/python scripts/make_summaries.py [--force] [--limit N]

Reads data/episodes.json (written by fetch_episodes.py) and writes
data/summaries.json. Resumable: an episode that already has a summary is
skipped unless --force, so a partial run costs nothing to finish. Each
episode is saved as it completes rather than at the end — a crash on
episode 9 should not throw away the first eight.
"""

import argparse
import asyncio
import json
import sys
from pathlib import Path

from anthropic import AsyncAnthropic

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.config import get_settings  # noqa: E402
from app.schemas import Episode  # noqa: E402
from app.summaries import SummaryStore  # noqa: E402

EPISODES_PATH = Path(__file__).resolve().parent.parent / "data" / "episodes.json"


async def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--force", action="store_true",
                        help="regenerate episodes that already have a summary")
    parser.add_argument("--limit", type=int, default=None,
                        help="stop after N episodes")
    args = parser.parse_args()

    if not EPISODES_PATH.exists():
        print(f"no {EPISODES_PATH} — run fetch_episodes.py first")
        return 1

    episodes = [Episode(**row) for row in json.loads(EPISODES_PATH.read_text())]
    store = SummaryStore()
    client = AsyncAnthropic(api_key=get_settings().anthropic_api_key)

    todo = [e for e in episodes if args.force or e.episode_id not in store]
    if args.limit:
        todo = todo[: args.limit]
    print(f"{len(episodes)} episodes, {len(todo)} to summarise")

    failures = 0
    for i, ep in enumerate(todo, 1):
        try:
            text = await store.generate(client, ep)
        except Exception as exc:  # noqa: BLE001
            # One bad episode should not abort the batch; the rerun picks
            # it up because nothing was written for it.
            failures += 1
            print(f"  [{i}/{len(todo)}] FAILED {ep.title[:52]}: {exc}")
            continue
        store.save()
        print(f"  [{i}/{len(todo)}] {len(text.split()):>4} words  {ep.title[:52]}")

    print(f"\nwrote {len(store.all())} summaries; {failures} failed")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
