"""List the Gemini models your GEMINI_API_KEY can call for text generation.

    python -m app.llm.list_models

Use stable ids (no "preview" / "exp") for LLM_GENERATION_MODEL, LLM_FAST_MODEL
and LLM_JUDGE_MODEL. Prints model ids and limits only, never the key.
"""

import asyncio
import sys

from google import genai

from app.config import get_settings


async def _list(api_key: str) -> list[tuple[str, int | None, int | None]]:
    client = genai.Client(api_key=api_key)
    try:
        rows = []
        async for model in await client.aio.models.list():
            if "generateContent" in (model.supported_actions or []):
                name = (model.name or "").removeprefix("models/")
                rows.append((name, model.input_token_limit, model.output_token_limit))
        return sorted(rows)
    finally:
        await client.aio.aclose()


def main() -> None:
    settings = get_settings()
    if not settings.gemini_api_key:
        sys.exit("GEMINI_API_KEY is not set in infra/.env")
    rows = asyncio.run(_list(settings.gemini_api_key))
    width = max((len(name) for name, _, _ in rows), default=10)
    print(f"{'model':<{width}}  {'input':>9}  {'output':>7}")
    for name, input_limit, output_limit in rows:
        tag = "  (preview)" if "preview" in name or "exp" in name else ""
        print(f"{name:<{width}}  {input_limit or '-':>9}  {output_limit or '-':>7}{tag}")


if __name__ == "__main__":
    main()
