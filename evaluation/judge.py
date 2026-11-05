"""Optional LLM judge: faithfulness and answer relevance (RAGAS-style) for one answer.

Scores an answer against the excerpts it was generated from. Each call costs
one LLM request, so run_eval only uses it with --judge. A judge from the same
model family as the generator tends to be lenient, so the judge role defaults
to another provider (LLM_JUDGE_PROVIDER); --judge-model overrides its model.
The verdict uses the provider's structured (JSON-schema) output.
"""

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

from pydantic import BaseModel, Field

from app.config import get_logger
from app.llm import LLMClient, LLMOutputError
from app.prompts import PromptTemplate

logger = get_logger(__name__)

_PROMPTS_DIR = Path(__file__).resolve().parent / "prompts"
JUDGE_MAX_TOKENS = 600


class JudgeReply(BaseModel):
    """The judge model's structured verdict."""

    faithfulness: float = Field(ge=0.0, le=1.0)
    answer_relevance: float = Field(ge=0.0, le=1.0)
    unsupported_claims: list[str] = Field(default_factory=list)


@dataclass(frozen=True)
class Excerpt:
    """One source the answer was generated from."""

    label: str  # "Book → Chapter → p. 12", for the judge's context only
    text: str


@lru_cache(maxsize=4)
def _template(name: str) -> PromptTemplate:
    text = (_PROMPTS_DIR / f"{name}.txt").read_text(encoding="utf-8").strip()
    return PromptTemplate(name=name, version="eval", text=text)


def build_judge_prompt(question: str, answer: str, excerpts: list[Excerpt]) -> tuple[str, str]:
    """Return (system, user) prompts, numbering excerpts [S1].. as the generator saw them."""
    sources = (
        "\n\n".join(f"[S{i}] {e.label}\n{e.text}" for i, e in enumerate(excerpts, start=1))
        or "(no excerpts)"
    )
    user = _template("judge_user").render(question=question, sources=sources, answer=answer)
    return _template("judge_system").render(), user


class AnswerJudge:
    """Ask an LLM to score one answer."""

    def __init__(self, llm: LLMClient) -> None:
        self._llm = llm

    @property
    def model(self) -> str:
        return self._llm.model

    async def judge(
        self, question: str, answer: str, excerpts: list[Excerpt], request_id: str
    ) -> JudgeReply | None:
        """Score *answer*; None (logged WARNING) if the reply can't be parsed.

        Raises:
            LLMError: if the judge model cannot be reached.
        """
        system, prompt = build_judge_prompt(question, answer, excerpts)
        try:
            return await self._llm.generate_structured(
                system=system,
                prompt=prompt,
                schema=JudgeReply,
                max_tokens=JUDGE_MAX_TOKENS,
                request_id=request_id,
            )
        except LLMOutputError as exc:
            logger.warning("Unusable judge reply: %s", exc, extra={"request_id": request_id})
            return None
