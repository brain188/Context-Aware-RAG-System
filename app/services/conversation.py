"""Per-session conversation memory (Redis) and follow-up rewriting.

Sessions are capped Redis lists with a sliding TTL. Redis failures degrade to
"no history" (WARNING) rather than failing the query. Logs carry the session
id, never turn content.
"""

from pydantic import BaseModel, ValidationError
from redis.asyncio import Redis
from redis.exceptions import RedisError

from app.config import Settings, get_logger
from app.llm import LLMClient, LLMError
from app.prompts import load_prompt

logger = get_logger(__name__)

_KEY_PREFIX = "rag:session:"


class Turn(BaseModel):
    """One question/answer exchange."""

    question: str
    answer: str


class ConversationStore:
    """Store and fetch recent turns per session_id."""

    def __init__(self, redis: Redis, settings: Settings) -> None:
        self._redis = redis
        self._max_turns = settings.conversation_max_turns
        self._ttl = settings.redis_ttl_seconds

    async def history(self, session_id: str, request_id: str) -> list[Turn]:
        """Return the session's recent turns, oldest first ([] if none or Redis is down)."""
        try:
            raw = await self._redis.lrange(_KEY_PREFIX + session_id, -self._max_turns, -1)
        except RedisError as exc:
            logger.warning(
                "Session history unavailable (Redis: %s); answering without it.",
                exc,
                extra={"request_id": request_id},
            )
            return []
        turns = []
        for item in raw:
            try:
                turns.append(Turn.model_validate_json(item))
            except ValidationError:
                logger.warning(
                    "Skipping malformed turn in session=%s.",
                    session_id,
                    extra={"request_id": request_id},
                )
        return turns

    async def append(self, session_id: str, turn: Turn, request_id: str) -> None:
        """Add *turn*, keep only the newest turns, and refresh the session TTL."""
        key = _KEY_PREFIX + session_id
        try:
            async with self._redis.pipeline(transaction=True) as pipe:
                pipe.rpush(key, turn.model_dump_json())
                pipe.ltrim(key, -self._max_turns, -1)
                pipe.expire(key, self._ttl)
                await pipe.execute()
        except RedisError as exc:
            logger.warning(
                "Could not save turn for session=%s (Redis: %s).",
                session_id,
                exc,
                extra={"request_id": request_id},
            )
            return
        logger.info("Saved turn for session=%s.", session_id, extra={"request_id": request_id})


def _format_history(turns: list[Turn]) -> str:
    return "\n".join(f"User: {t.question}\nAssistant: {t.answer}" for t in turns)


class QueryRewriter:
    """Rewrite a follow-up question into a standalone one using prior turns."""

    def __init__(self, llm: LLMClient, settings: Settings) -> None:
        self._llm = llm
        self._max_tokens = settings.llm_rewrite_max_tokens
        self._version = settings.prompt_version

    async def rewrite(self, question: str, history: list[Turn], request_id: str) -> str:
        """Return a standalone version of *question*; unchanged when there is no history.

        An LLM failure degrades to the original question (WARNING): retrieval
        still works for most follow-ups, just less precisely.
        """
        if not history:
            return question
        system = load_prompt("rewrite_system", self._version).render()
        prompt = load_prompt("rewrite_user", self._version).render(
            history=_format_history(history), question=question
        )
        try:
            reply = await self._llm.complete(
                system=system, prompt=prompt, max_tokens=self._max_tokens, request_id=request_id
            )
        except LLMError as exc:
            logger.warning(
                "Follow-up rewrite failed (%s); using the question as asked.",
                exc,
                extra={"request_id": request_id},
            )
            return question
        rewritten = " ".join(reply.split()).strip("\"'")
        if not rewritten:
            return question
        logger.info(
            "Follow-up rewrite: turns=%d changed=%s",
            len(history),
            rewritten != question,
            extra={"request_id": request_id},
        )
        return rewritten
