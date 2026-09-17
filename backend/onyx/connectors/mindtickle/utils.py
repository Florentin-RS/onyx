import json
from collections.abc import Callable, Iterable
from typing import Any, TypeVar

from onyx.file_processing.html_utils import parse_html_page_basic

# `asset_type` values seen live are `MEDIA_TYPE_DOCUMENT_<KIND>`; the docs also
# list bare kinds such as `PDF`. Both spellings map to the same extension.
_ASSET_TYPE_EXTENSIONS: dict[str, str] = {
    "PDF": ".pdf",
    "DOCUMENT_PDF": ".pdf",
    "DOC": ".docx",
    "WORD": ".docx",
    "DOCUMENT_WORD": ".docx",
    "PPT": ".pptx",
    "DOCUMENT_PPT": ".pptx",
    "XLS": ".xlsx",
    "DOCUMENT_XLS": ".xlsx",
}
_ASSET_TYPE_PREFIX = "MEDIA_TYPE_"

T = TypeVar("T")


def html_to_text(html: str | None) -> str:
    """Strip Mindtickle's HTML fragments down to text."""
    if not html or not html.strip():
        return ""
    return parse_html_page_basic(html).strip()


def asset_type_to_extension(asset_type: str | None) -> str | None:
    """Map a Mindtickle asset type to a file extension Onyx can extract text from."""
    if not asset_type:
        return None
    kind = asset_type.strip().upper()
    if kind.startswith(_ASSET_TYPE_PREFIX):
        kind = kind[len(_ASSET_TYPE_PREFIX) :]
    return _ASSET_TYPE_EXTENSIONS.get(kind)


def _words_to_text(words: Iterable[Any]) -> str:
    parts: list[str] = []
    for word in words:
        if isinstance(word, dict):
            value = word.get("word") or word.get("text")
            if isinstance(value, str) and value.strip():
                parts.append(value.strip())
        elif isinstance(word, str) and word.strip():
            parts.append(word.strip())
    return " ".join(parts)


def transcript_to_text(raw: bytes) -> str:
    """Turn a Mindtickle transcript download into plain text.

    Documents come back as plain text. Videos come back as JSON with a word list
    under `results.words`, each word carrying `word`, `start_time`, `end_time`.
    """
    text = raw.decode("utf-8", errors="replace").strip()
    if not text:
        return ""
    if text[0] not in "{[":
        return text
    try:
        data = json.loads(text)
    except ValueError:
        return text

    if isinstance(data, list):
        return _words_to_text(data)
    if not isinstance(data, dict):
        return text

    results = data.get("results")
    if isinstance(results, dict) and isinstance(results.get("words"), list):
        return _words_to_text(results["words"])
    if isinstance(data.get("words"), list):
        return _words_to_text(data["words"])
    for key in ("transcript", "text"):
        value = data.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return ""


def filter_by_name(
    items: list[T],
    name_of: Callable[[T], str],
    allowed_names: list[str] | None,
    excluded_names: list[str] | None,
) -> list[T]:
    """Apply an allow-list and an exclude-list of names, case-insensitively.

    An empty allow-list means everything is allowed.
    """
    allowed = {name.strip().lower() for name in allowed_names or [] if name.strip()}
    excluded = {name.strip().lower() for name in excluded_names or [] if name.strip()}
    selected: list[T] = []
    for item in items:
        key = name_of(item).strip().lower()
        if allowed and key not in allowed:
            continue
        if key in excluded:
            continue
        selected.append(item)
    return selected
