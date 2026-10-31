"""Versioned prompt templates, loaded by name + version.

Templates live in app/prompts/templates/<version>/<name>.txt and use
string.Template placeholders ($name), so braces in code or JSON need no
escaping and substituted values (book text) are never re-parsed.
Changing a prompt means adding a new version directory, not editing in place,
so evaluation runs stay comparable.
"""

import re
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from string import Template

_TEMPLATES_DIR = Path(__file__).resolve().parent / "templates"
_SAFE_NAME = re.compile(r"^[a-z0-9_]+$")

# The exact refusal sentence; answer templates instruct the model to use it and
# the output guard recognizes it.
NOT_COVERED_MESSAGE = "This is not covered in your library."


class PromptNotFoundError(LookupError):
    """No template exists for the requested name and version."""


@dataclass(frozen=True)
class PromptTemplate:
    """A loaded template; render() fills its $placeholders."""

    name: str
    version: str
    text: str

    def render(self, **values: str) -> str:
        """Substitute every placeholder.

        Raises:
            ValueError: if a placeholder has no value.
        """
        try:
            return Template(self.text).substitute(values)
        except KeyError as exc:
            raise ValueError(
                f"Prompt '{self.name}' ({self.version}) needs a value for {exc}"
            ) from exc


@lru_cache(maxsize=64)
def load_prompt(name: str, version: str) -> PromptTemplate:
    """Return the template *name* at *version* (e.g. "answer_system", "v1").

    Raises:
        PromptNotFoundError: if the name/version is malformed or has no file.
    """
    if not _SAFE_NAME.match(name) or not _SAFE_NAME.match(version):
        raise PromptNotFoundError(f"Invalid prompt name/version: {name!r} / {version!r}")
    path = _TEMPLATES_DIR / version / f"{name}.txt"
    if not path.is_file():
        raise PromptNotFoundError(
            f"No prompt '{name}' for version '{version}' (expected {path}); check PROMPT_VERSION"
        )
    return PromptTemplate(name=name, version=version, text=path.read_text(encoding="utf-8").strip())
