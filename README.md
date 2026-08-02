# Threadguy Search

Ask [Threadguy's](https://www.youtube.com/@notthreadguy) streams anything in
plain English and jump to the exact second it was said.

A three-hour video is unsearchable. You either remember which stream something
was in and scrub for it, or you give up. This indexes the transcripts so you
can ask a question instead.

## How it works

A RAG pipeline over the auto-generated captions:

1. **Pull the captions** with `yt-dlp` (no video download). YouTube's
   auto-captions repeat words in a rolling window, so the parser dedupes at
   word granularity and keeps the **timestamp** on every line. That timestamp
   is what later lets an answer link to the exact second.
2. **Chunk** each stream into windows of consecutive lines, each carrying its
   start time.
3. **Embed** each window with Voyage and store it in Pinecone.
4. **Answer** by embedding the question, retrieving the closest windows, and
   asking Claude to answer *only* from them — citing the timestamp, never
   inventing quotes. If it isn't in the streams, it says so.

The window is embedded together with its stream title. A stream's subject is
often in the title and barely spoken aloud, so embedding the passage alone
made whole topics unreachable. The stored excerpt stays as the transcript, so
a reader still sees what was actually said.

## Summaries

Search only helps someone who already knows what to ask. `make_summaries.py`
condenses each episode to a two-minute read where every bullet carries a
timestamp, so the page turns them into links straight into the video.

It is a one-off batch job per episode and resumable — an episode that already
has a summary is skipped, so a failed run costs nothing to finish. A summary
that hits the token ceiling is refused rather than stored: a write cut off
mid-sentence still looks valid in the JSON, and nothing downstream would
catch it. Transcripts stay out of the runtime image; only the summaries ship.

## Isolation

This shares a Pinecone index with a sibling project but uses its own
namespace (`threadguy`), and runs as its own service. Neither catalogue can
surface in the other's results, and that is enforced by the deployment
boundary rather than by a filter someone has to remember.

## Running it

```bash
python -m venv .venv && .venv/bin/pip install -r requirements.txt
cp .env.example .env          # ANTHROPIC_API_KEY, VOYAGE_API_KEY, PINECONE_API_KEY

.venv/bin/python scripts/fetch_episodes.py --latest 10   # captions -> data/episodes.json
.venv/bin/python scripts/ingest_episodes.py              # embed -> Pinecone
.venv/bin/python scripts/make_summaries.py               # 2-min read per episode
.venv/bin/uvicorn app.main:app --reload
```

Then open http://localhost:8000.

On Voyage's free tier (3 requests/minute) a ten-stream ingest takes roughly
twelve minutes. That is a rate limit, not a bug.

## API

| | |
|---|---|
| `POST /v1/search` | answer plus timestamped hits |
| `POST /v1/search/stream` | the same, streamed as SSE |
| `GET /v1/episodes` | what's indexed, with each episode's summary |
| `POST /v1/ingest` | admin only, requires `X-Admin-Token` |
| `GET /healthz` | liveness |

Public endpoints carry a per-IP rate limit, a bot-wide ceiling, and a daily
request budget, so an unattended spike costs a few dollars rather than a
surprise bill.
