"""Shared embedding helper: batches large inputs and retries rate limits.

Voyage free-tier keys are limited to 3 requests/min and 10K tokens/min, so
bulk ingestion must chunk its calls and back off politely. Paid keys sail
through the same code path with fewer, larger batches.
"""

import asyncio
import logging
from collections import OrderedDict

import voyageai
from voyageai import error as voyage_error

logger = logging.getLogger(__name__)

# Small LRU cache for single-text query embeddings. Query embeddings are
# deterministic per (text, model, dimension), and demo chips / popular
# questions repeat heavily — caching them dodges the free-tier 3 RPM limit
# for the common case. Ingestion (many unique texts) does not use this.
_QUERY_CACHE: "OrderedDict[tuple, list[float]]" = OrderedDict()
_QUERY_CACHE_MAX = 512


async def rerank_order(
    client: voyageai.AsyncClient,
    query: str,
    documents: list[str],
    *,
    top_k: int,
    model: str,
) -> list[int] | None:
    """Re-score retrieved documents for actual relevance to the query.

    Returns the document indices in best-first order, or None on any
    failure (rate limit, API change) — the caller falls back to the
    original vector-similarity order. Reranking improves quality; it must
    never break search.
    """
    try:
        result = await client.rerank(
            query=query, documents=documents, model=model, top_k=top_k
        )
        return [r.index for r in result.results]
    except Exception as exc:  # noqa: BLE001 — quality upgrade, never fatal
        logger.warning("rerank skipped (%s): %s", type(exc).__name__, exc)
        return None


async def embed_query(
    client: voyageai.AsyncClient,
    text: str,
    *,
    model: str,
    dimension: int,
) -> list[float]:
    """Embed a single query string, cached. Falls back to embed_texts."""
    key = (text.strip(), model, dimension)
    cached = _QUERY_CACHE.get(key)
    if cached is not None:
        _QUERY_CACHE.move_to_end(key)
        return cached
    vector = (
        await embed_texts(
            client, [text], model=model, dimension=dimension, input_type="query"
        )
    )[0]
    _QUERY_CACHE[key] = vector
    if len(_QUERY_CACHE) > _QUERY_CACHE_MAX:
        _QUERY_CACHE.popitem(last=False)
    return vector

# ~24K chars ≈ 6-7K tokens — safely under the free tier's 10K TPM per call.
BATCH_CHAR_BUDGET = 24_000
MAX_RETRIES = 6


def _batches(texts: list[str], char_budget: int) -> list[list[str]]:
    batches: list[list[str]] = []
    current: list[str] = []
    size = 0
    for text in texts:
        if current and size + len(text) > char_budget:
            batches.append(current)
            current, size = [], 0
        current.append(text)
        size += len(text)
    if current:
        batches.append(current)
    return batches


async def embed_texts(
    client: voyageai.AsyncClient,
    texts: list[str],
    *,
    model: str,
    dimension: int,
    input_type: str,
) -> list[list[float]]:
    """Embed `texts`, transparently batching and retrying 429s with backoff."""
    embeddings: list[list[float]] = []
    batches = _batches(texts, BATCH_CHAR_BUDGET)
    for i, batch in enumerate(batches):
        for attempt in range(1, MAX_RETRIES + 1):
            try:
                result = await client.embed(
                    texts=batch,
                    model=model,
                    input_type=input_type,
                    output_dimension=dimension,
                )
                embeddings.extend(result.embeddings)
                break
            except voyage_error.RateLimitError:
                if attempt == MAX_RETRIES:
                    raise
                wait = min(25 * attempt, 90)
                logger.info(
                    "Voyage rate limit (batch %d/%d, try %d) — waiting %ds",
                    i + 1, len(batches), attempt, wait,
                )
                await asyncio.sleep(wait)
        if len(batches) > 1 and i < len(batches) - 1:
            await asyncio.sleep(21)  # free tier: 3 requests/minute
    return embeddings
