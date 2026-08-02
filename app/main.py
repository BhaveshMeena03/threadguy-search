"""API layer — search over a single YouTube channel's back catalogue.

Endpoints:
    POST /v1/search         -> answer + timestamped hits
    POST /v1/search/stream  -> Server-Sent Events token stream
    POST /v1/ingest         -> (admin) index episodes
    GET  /v1/episodes       -> what's indexed
    GET  /healthz           -> liveness probe

A separate service from the Market Bubble one on purpose. Same idea, different
channel, and a different Pinecone namespace — so neither can ever return the
other's content. The isolation is a deployment boundary rather than a filter
somebody has to remember to apply.
"""

import json
import logging
from contextlib import asynccontextmanager
from datetime import UTC, datetime
from pathlib import Path

import anthropic
from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, RedirectResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from voyageai import error as voyage_error

from .podcast import REFUSAL_ANSWER, PodcastIndex
from .schemas import Episode, PodcastSearchRequest, PodcastSearchResponse
from .security import (
    daily_budget,
    global_rate_limit,
    public_rate_limit,
    require_admin,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

STATS: dict = {
    "started_at": datetime.now(UTC).isoformat(timespec="seconds"),
    "searches": 0,
}


def _track(kind: str, **fields) -> None:
    STATS[kind] = STATS.get(kind, 0) + 1
    logger.info("ANALYTICS %s", json.dumps({"event": kind, **fields}))


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.podcast = PodcastIndex()
    yield


app = FastAPI(title="Threadguy Search", version="1.0.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["POST", "GET"],
    allow_headers=["content-type"],
)

_ROOT = Path(__file__).resolve().parent.parent
app.mount("/demo", StaticFiles(directory=_ROOT / "demo", html=True), name="demo")


def get_podcast(request: Request) -> PodcastIndex:
    return request.app.state.podcast


@app.exception_handler(voyage_error.RateLimitError)
async def _voyage_rate_limit(request: Request, exc: voyage_error.RateLimitError):
    logger.warning("Voyage rate limit on %s", request.url.path)
    return JSONResponse(
        status_code=503,
        content={"detail": "Search is busy right now — try again in a moment."},
        headers={"Retry-After": "10"},
    )


@app.exception_handler(voyage_error.VoyageError)
async def _voyage_error(request: Request, exc: voyage_error.VoyageError):
    logger.error("Voyage error on %s: %s", request.url.path, exc)
    return JSONResponse(status_code=502, content={"detail": "Embedding provider error."})


@app.exception_handler(Exception)
async def _unhandled(request: Request, exc: Exception):
    logger.exception("Unhandled error on %s", request.url.path)
    return JSONResponse(
        status_code=500, content={"detail": "Something went wrong. Please retry."}
    )


@app.get("/", include_in_schema=False)
async def root() -> RedirectResponse:
    return RedirectResponse(url="/demo/podcast.html")


@app.get("/healthz")
async def healthz() -> dict:
    return {"status": "ok"}


@app.get("/v1/stats")
async def stats() -> dict:
    return {**STATS, "daily_budget": daily_budget.state()}


@app.post(
    "/v1/search",
    response_model=PodcastSearchResponse,
    dependencies=[
        Depends(public_rate_limit),
        Depends(global_rate_limit),
        Depends(daily_budget),
    ],
)
async def search(
    body: PodcastSearchRequest,
    podcast: PodcastIndex = Depends(get_podcast),
) -> PodcastSearchResponse:
    _track("searches", q=body.query[:120])
    try:
        return await podcast.search(body.query, top_k=body.top_k)
    except anthropic.RateLimitError as exc:
        raise HTTPException(429, "Rate limited; retry shortly.") from exc
    except anthropic.APIError as exc:
        logger.error("Anthropic error on search: %s", type(exc).__name__)
        raise HTTPException(502, "Model provider error.") from exc


@app.post(
    "/v1/search/stream",
    dependencies=[
        Depends(public_rate_limit),
        Depends(global_rate_limit),
        Depends(daily_budget),
    ],
)
async def search_stream(
    body: PodcastSearchRequest,
    podcast: PodcastIndex = Depends(get_podcast),
) -> StreamingResponse:
    _track("searches", q=body.query[:120], stream=True)
    hits = await podcast.retrieve(body.query, body.top_k)

    async def event_source():
        try:
            # A bare array, not an object. The page passes this event's parsed
            # payload straight to renderHits, so wrapping it in {"hits": ...}
            # silently rendered "no matching moments" while the answer streamed
            # in fine — the timestamps, which are the whole point, vanished.
            payload = json.dumps([h.model_dump() for h in hits])
            yield f"event: hits\ndata: {payload}\n\n"
            if not hits:
                yield f"event: refusal\ndata: {json.dumps({'text': REFUSAL_ANSWER})}\n\n"
                return
            async for delta in podcast.answer_stream(body.query, hits):
                yield f"data: {json.dumps({'text': delta})}\n\n"
            yield "event: done\ndata: {}\n\n"
        except Exception as exc:  # noqa: BLE001
            # Headers are already flushed, so an uncaught error would leave
            # the client hanging with no terminal event.
            logger.exception("Search stream failure: %s", exc)
            yield f"event: error\ndata: {json.dumps({'detail': 'stream failed'})}\n\n"

    return StreamingResponse(
        event_source(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@app.post("/v1/ingest", dependencies=[Depends(require_admin)])
async def ingest(
    episodes: list[Episode],
    podcast: PodcastIndex = Depends(get_podcast),
) -> dict:
    return {"windows_indexed": await podcast.ingest(episodes)}


@app.get("/v1/episodes")
async def episodes() -> list[dict]:
    """What's indexed, from the manifest written at ingest time."""
    path = _ROOT / "data" / "episodes.json"
    if not path.exists():
        return []
    return [
        {"episode_id": e["episode_id"], "title": e["title"], "url": e["url"]}
        for e in json.loads(path.read_text())
    ]
