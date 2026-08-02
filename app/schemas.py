"""Pydantic models shared across the API, retriever, and agent layers."""

from enum import StrEnum
from typing import Literal

from pydantic import BaseModel, Field, field_validator

# Ceiling on a replayed conversation, summed across turns. See the validator
# on ChatRequest.history for why the per-turn limits were not enough.
MAX_HISTORY_CHARS = 60_000


class SourceType(StrEnum):
    """The three unstructured corpora the concierge is grounded in."""

    DOCS = "docs"          # Markdown product documentation
    PODCAST = "podcast"    # stream transcripts
    TWEET = "tweet"        # unused here; kept so stored records parse


class ChatTurn(BaseModel):
    role: Literal["user", "assistant"]
    content: str = Field(..., max_length=16000)


class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1, max_length=8000)
    # Prior turns of the conversation; the API layer is stateless so the
    # client replays history. Keeping history byte-identical between turns
    # is what lets the prompt cache pay off.
    history: list[ChatTurn] = Field(default_factory=list, max_length=40)
    # Optional metadata pre-filter, e.g. {"source_type": "docs"} to answer
    # only from documentation, or {"episode": "ep-42"}.
    filters: dict[str, str | int | bool] | None = None
    session_id: str | None = None
    # Ask for a chat-sized answer. Chat clients set this; the web page
    # does not, because it has room to read a thorough one.
    brief: bool = False

    @field_validator("history")
    @classmethod
    def _bound_total_history(cls, turns: list[ChatTurn]) -> list[ChatTurn]:
        """Cap the whole conversation, not just each turn.

        The per-field limits alone allowed 40 turns of 16,000 characters:
        640,000 characters, roughly 160k tokens, on every request. Input is
        billed per token, so that made a single caller's spend a function of
        what they chose to send rather than of the rate limit.

        Real clients are nowhere near it. The web page replays at most 20
        entries, the Discord bot 6, and an answer runs about 2,000 characters,
        so a genuine conversation is well under 60,000. Anything larger is a
        client bug or someone probing the bill.
        """
        total = sum(len(t.content) for t in turns)
        if total > MAX_HISTORY_CHARS:
            raise ValueError(
                f"history too large: {total} characters, limit {MAX_HISTORY_CHARS}"
            )
        return turns


class RetrievedChunk(BaseModel):
    id: str
    text: str
    score: float
    source_type: SourceType
    metadata: dict


class ChatResponse(BaseModel):
    answer: str
    sources: list[RetrievedChunk]
    model: str
    refused: bool = False
    usage: dict | None = None


class IngestDocument(BaseModel):
    """One raw document handed to the ingestion pipeline."""

    source_type: SourceType
    source_id: str                 # doc slug / episode id / tweet id
    text: str
    title: str | None = None
    author: str | None = None
    url: str | None = None
    published_at: str | None = None  # ISO 8601


# --- Stream search -----------------------------------------

class TranscriptSegment(BaseModel):
    """One timestamped line of a podcast transcript (as it comes out of a
    YouTube/Whisper caption file)."""

    t: float = Field(..., ge=0)   # start time in seconds
    text: str
    speaker: str | None = None


class Episode(BaseModel):
    """One indexed stream with a timestamped transcript."""

    episode_id: str
    title: str
    url: str                       # base watch URL (YouTube/Spotify)
    platform: Literal["youtube", "spotify", "other"] = "youtube"
    published_at: str | None = None
    segments: list[TranscriptSegment]


class PodcastHit(BaseModel):
    """A retrieved transcript window, deep-linked to the moment in the episode."""

    episode_id: str
    title: str
    start_seconds: float
    timestamp: str                 # "14:32"
    deep_link: str                 # url that jumps to start_seconds
    text: str
    score: float


class PodcastSearchRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=2000)
    top_k: int | None = None


class PodcastSearchResponse(BaseModel):
    answer: str
    hits: list[PodcastHit]
    model: str
    refused: bool = False
