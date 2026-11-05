"""Rewrite the search query after an ambiguous grade, so a second retrieval can find what was missing.

Not the follow-up rewriter (services/conversation.py), which makes a question
standalone. This one keeps the user's question fixed and changes only what is
sent to the search: the grader's reason says what the first excerpts lacked.
"""

from app.config import Settings, get_logger
from app.llm import LLMClient, LLMError
from app.prompts import PromptNotFoundError, load_prompt

logger = get_logger(__name__)


class SearchQueryRewriter:
    """Ask the fast model for one better search query."""

    def __init__(self, llm: LLMClient, settings: Settings) -> None:
        self._llm = llm
        self._max_tokens = settings.llm_rewrite_max_tokens
        self._version = settings.prompt_version

    async def rewrite(
        self, question: str, tried: list[str], missing: str, request_id: str
    ) -> str | None:
        """Return a new search query, or None when there is no usable new one.

        *tried* holds the queries already searched, oldest first (the question
        itself first). None (logged, never raised) when the prompt version has
        no rewrite template, the LLM call fails, or the reply repeats a query in
        *tried*: the stage then keeps the excerpts it has.
        """
        try:
            system = load_prompt("retrieval_rewrite_system", self._version).render()
            prompt = load_prompt("retrieval_rewrite_user", self._version).render(
                question=question, query=tried[-1], missing=missing or "not stated"
            )
        except PromptNotFoundError:
            logger.warning(
                "Prompt version %s has no retrieval_rewrite template; not retrying the search.",
                self._version,
                extra={"request_id": request_id},
            )
            return None
        try:
            reply = await self._llm.generate(
                system=system, prompt=prompt, max_tokens=self._max_tokens, request_id=request_id
            )
        except LLMError as exc:
            logger.warning(
                "Search query rewrite failed (%s); keeping the excerpts found so far.",
                exc,
                extra={"request_id": request_id},
            )
            return None
        query = " ".join(reply.text.split()).strip("\"'")
        if not query or query.casefold() in {q.casefold() for q in tried}:
            logger.info("Search query rewrite gave nothing new.", extra={"request_id": request_id})
            return None
        return query
