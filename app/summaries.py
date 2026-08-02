"""Per-episode summaries — a 2-minute read of a 3-hour stream.

Search only helps someone who already knows what to ask. A summary gives
the rest of them somewhere to start, and because every claim carries a
bracketed timestamp the page turns into links straight into the video.

Summarising is a one-off batch job per episode, not something that runs on
a request, so the results live in a small JSON file that ships with the
image. Transcripts stay out of it — they are two orders of magnitude
larger and nothing serves them.
"""

import json
import logging
from pathlib import Path

from anthropic import AsyncAnthropic

from .config import get_settings
from .schemas import Episode

logger = logging.getLogger(__name__)

_ROOT = Path(__file__).resolve().parent.parent
SUMMARY_PATH = _ROOT / "data" / "summaries.json"

# Timestamps must come out as [h:mm:ss] or [m:ss] — the page linkifies
# exactly that shape and passes anything else through as literal text.
# Bold and bullets are the only other markup its renderer understands.
SUMMARY_PROMPT = """\
You are writing show notes for one episode of Threadguy's stream, from its \
transcript. Someone who has not watched should be able to read this in two \
minutes and know what happened.

Format, exactly:
- Open with one plain sentence saying what the episode was about. No heading.
- Then 4-7 bullets, each starting with a bolded 2-4 word label, like:
  - **Leopold blowup** — what was said, in one or two sentences. [14:39]
- Every bullet ends with a timestamp in square brackets: [h:mm:ss] for \
anything past an hour, [m:ss] otherwise. Use the timestamps in the \
transcript — never estimate one.

Rules:
1. Use ONLY the transcript. Do not add background, context, or anything you \
know from elsewhere. If the transcript is too garbled to summarise, say so \
in one sentence and stop.
2. Paraphrase. Never invent a quote or attribute a view to someone who did \
not express it.
3. Cover what was actually discussed at length, not passing mentions.
4. This is informational. Never add a buy/sell recommendation or a price \
prediction of your own, even if the hosts made one — reporting that they \
said it is fine, endorsing it is not.
5. No preamble, no sign-off, no "in this episode". Start with the sentence.

Transcript follows."""


def _transcript_text(episode: Episode) -> str:
    """Flatten segments into timestamped lines the model can cite."""
    lines = []
    for seg in episode.segments:
        sec = int(seg.t)
        h, rem = divmod(sec, 3600)
        m, s = divmod(rem, 60)
        stamp = f"{h}:{m:02d}:{s:02d}" if h else f"{m}:{s:02d}"
        lines.append(f"[{stamp}] {seg.text}")
    return "\n".join(lines)


class SummaryStore:
    """Load, generate and persist episode summaries."""

    def __init__(self, path: Path = SUMMARY_PATH) -> None:
        self._path = path
        self._settings = get_settings()
        self._entries: dict[str, dict] = {}
        if path.exists():
            for row in json.loads(path.read_text()):
                self._entries[row["episode_id"]] = row

    def __contains__(self, episode_id: str) -> bool:
        return bool(self._entries.get(episode_id, {}).get("summary"))

    def all(self) -> list[dict]:
        return list(self._entries.values())

    def save(self) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._path.write_text(
            json.dumps(list(self._entries.values()), indent=2, ensure_ascii=False)
        )

    def _request(self, episode: Episode) -> dict:
        model = self._settings.summary_model
        request: dict = {
            "model": model,
            "max_tokens": self._settings.summary_max_tokens,
            # The instructions are identical across episodes, so caching the
            # system block means only the transcript is billed at full rate
            # from the second episode onward.
            "system": [
                {
                    "type": "text",
                    "text": SUMMARY_PROMPT,
                    "cache_control": {"type": "ephemeral"},
                }
            ],
            "messages": [
                {"role": "user", "content": _transcript_text(episode)}
            ],
        }
        if "haiku" not in model and not model.startswith("claude-fable"):
            # Condensing a transcript is recall, not reasoning.
            request["output_config"] = {"effort": "low"}
            request["thinking"] = {"type": "disabled"}
        return request

    async def generate(self, client: AsyncAnthropic, episode: Episode) -> str:
        message = await client.messages.create(**self._request(episode))
        text = "".join(
            block.text for block in message.content if block.type == "text"
        ).strip()
        if not text:
            raise ValueError(f"empty summary for {episode.episode_id}")
        if message.stop_reason == "max_tokens":
            # A summary cut off mid-sentence still looks plausible in the
            # JSON, so nothing downstream would ever catch it. Refuse to
            # store it and let the rerun pick the episode up again.
            raise ValueError(
                f"summary truncated at max_tokens for {episode.episode_id} "
                f"— raise summary_max_tokens (currently "
                f"{self._settings.summary_max_tokens})"
            )
        self._entries[episode.episode_id] = {
            "episode_id": episode.episode_id,
            "title": episode.title,
            "url": episode.url,
            "summary": text,
        }
        return text
