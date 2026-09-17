"""Text extraction for Mindtickle training modules.

A module's learning objects are mostly authored HTML courses (Articulate Rise)
packaged as SCORM zips, plus uploaded documents, videos with transcripts, quiz
questions and iframe embeds. This module turns each of them into plain text.
"""

import base64
import io
import json
import re
import zipfile
from collections.abc import Callable
from typing import Any
from urllib.parse import urlparse

from onyx.connectors.mindtickle.models import MindtickleLearningObject
from onyx.connectors.mindtickle.utils import html_to_text, transcript_to_text
from onyx.file_processing.extract_file_text import extract_file_text
from onyx.utils.logger import setup_logger

logger = setup_logger()

# Module types whose learning objects the API exposes. Checklists, coaching and
# reinforcement modules answer HTTP 400 "Module type not supported".
SUPPORTED_MODULE_TYPES: frozenset[str] = frozenset({"UPDATE", "COURSE", "ASSESSMENT"})

MAX_PACKAGE_BYTES = 200 * 1024 * 1024
MAX_DOCUMENT_BYTES = 50 * 1024 * 1024
MAX_TRANSCRIPT_BYTES = 20 * 1024 * 1024
# Upper bound on the bytes read out of a zip, so a hostile package cannot exhaust memory.
_MAX_ZIP_MEMBER_BYTES = 50 * 1024 * 1024
_MAX_ZIP_TOTAL_BYTES = 200 * 1024 * 1024

_RISE_RUNTIME_DATA_MEMBER = "scormcontent/runtime-data.js"
_RISE_JSONP_RE = re.compile(r'__jsonp\(\s*"runtime-data\.js"\s*,\s*"([^"]+)"')
# Keys in Rise's course JSON that carry author-written text, in reading order.
_RISE_TEXT_KEYS: tuple[str, ...] = (
    "title",
    "heading",
    "subheading",
    "paragraph",
    "description",
    "caption",
    "text",
    "question",
    "answer",
    "feedback",
    "srcdoc",
)
_RISE_SKIP_KEYS: frozenset[str] = frozenset(
    {"settings", "theme", "fonts", "media", "coverImage", "labelSet", "exportSettings"}
)
_SCORM_IGNORED_PREFIXES = ("scormdriver/", "lib/", "scormcontent/lib/")
_DOCUMENT_MEDIA_TYPES: frozenset[str] = frozenset({"document", "text document"})
_TRANSCRIBABLE_MEDIA_TYPES: frozenset[str] = frozenset(
    {"video", "audio", "video recording", "screen recording", "voiceover slideshow"}
)

Downloader = Callable[[str, int], bytes | None]


def _title_from_manifest(archive: zipfile.ZipFile) -> str:
    for name in archive.namelist():
        if name.endswith("imsmanifest.xml"):
            manifest = _read_member(archive, name).decode("utf-8", errors="replace")
            match = re.search(r"<title>(.*?)</title>", manifest, re.S)
            if match:
                return html_to_text(match.group(1))
    return ""


def _read_member(archive: zipfile.ZipFile, name: str) -> bytes:
    info = archive.getinfo(name)
    if info.file_size > _MAX_ZIP_MEMBER_BYTES:
        raise ValueError(f"zip member {name} is too large ({info.file_size} bytes)")
    return archive.read(name)


def rise_runtime_data_to_text(runtime_data_js: str) -> str:
    """Decode Articulate Rise's `runtime-data.js` and return the course text.

    The file is a JSONP call whose single argument is base64-encoded JSON with
    the course, its lessons and their content blocks.
    """
    match = _RISE_JSONP_RE.search(runtime_data_js)
    if not match:
        return ""
    encoded = match.group(1)
    decoded = base64.b64decode(encoded + "=" * (-len(encoded) % 4))
    data = json.loads(decoded)
    course = data.get("course") if isinstance(data, dict) else None
    if not isinstance(course, dict):
        return ""

    lines: list[str] = []
    title = html_to_text(str(course.get("title") or ""))
    if title:
        lines.append(title)
    description = html_to_text(str(course.get("description") or ""))
    if description:
        lines.append(description)

    for lesson in course.get("lessons") or []:
        if not isinstance(lesson, dict):
            continue
        lesson_title = html_to_text(str(lesson.get("title") or ""))
        if lesson_title:
            lines.append(f"\n## {lesson_title}")
        for block in lesson.get("items") or []:
            block_lines: list[str] = []
            _collect_rise_text(block, block_lines)
            lines.extend(block_lines)
    return "\n".join(line for line in lines if line.strip()).strip()


def _collect_rise_text(node: Any, out: list[str]) -> None:
    if isinstance(node, dict):
        for key, value in node.items():
            if key in _RISE_SKIP_KEYS:
                continue
            if key in _RISE_TEXT_KEYS and isinstance(value, str):
                text = html_to_text(value)
                if text and text not in out[-3:]:
                    out.append(text)
            elif isinstance(value, (dict, list)):
                _collect_rise_text(value, out)
    elif isinstance(node, list):
        for item in node:
            _collect_rise_text(item, out)


def scorm_zip_to_text(package: bytes) -> str:
    """Extract course text from a SCORM package.

    Articulate Rise packages carry all text in one JSON blob. Other authoring
    tools ship plain HTML pages, which are stripped to text as a fallback.
    """
    with zipfile.ZipFile(io.BytesIO(package)) as archive:
        total = sum(info.file_size for info in archive.infolist())
        if total > _MAX_ZIP_TOTAL_BYTES:
            raise ValueError(f"SCORM package expands to {total} bytes; skipping")

        names = archive.namelist()
        if _RISE_RUNTIME_DATA_MEMBER in names:
            runtime = _read_member(archive, _RISE_RUNTIME_DATA_MEMBER)
            text = rise_runtime_data_to_text(runtime.decode("utf-8", errors="replace"))
            if text:
                return text

        parts: list[str] = []
        title = _title_from_manifest(archive)
        if title:
            parts.append(title)
        for name in names:
            lowered = name.lower()
            if not lowered.endswith((".html", ".htm", ".xhtml")):
                continue
            if any(lowered.startswith(prefix) for prefix in _SCORM_IGNORED_PREFIXES):
                continue
            page = html_to_text(
                _read_member(archive, name).decode("utf-8", errors="replace")
            )
            if page:
                parts.append(page)
        return "\n\n".join(parts).strip()


def _embed_host(htmlsrc: str) -> str | None:
    match = re.search(r'src=["\']([^"\']+)', htmlsrc)
    if not match:
        return None
    return urlparse(match.group(1)).netloc or None


def _file_name_from_url(url: str, fallback: str) -> str:
    path = urlparse(url).path
    name = path.rsplit("/", 1)[-1]
    return name if "." in name else fallback


def _quiz_text(lo: MindtickleLearningObject) -> str:
    lines: list[str] = []
    question = html_to_text(lo.question)
    if question:
        lines.append(question)
    for option in lo.options:
        label = html_to_text(option.option or option.text or option.question)
        if option.answer:
            answer = html_to_text(option.answer)
            lines.append(f"- {label}: {answer}" if label else f"- {answer}")
        elif label:
            lines.append(f"- {label}")
    answer = html_to_text(lo.answer or lo.exact_answer)
    if answer:
        lines.append(f"Answer: {answer}")
    return "\n".join(lines)


def learning_object_to_text(lo: MindtickleLearningObject, download: Downloader) -> str:
    """Return the indexable text of one learning object, or an empty string."""
    if lo.type.lower() != "content":
        return _quiz_text(lo)

    media = lo.media
    if media is None:
        return ""
    media_type = (media.type or "").strip().lower()
    title = html_to_text(media.title)
    url = media.url or ""
    bare_url = url.split("?", 1)[0].lower()

    if bare_url.endswith(".zip"):
        package = download(url, MAX_PACKAGE_BYTES)
        if package is None:
            logger.warning("Mindtickle package %s exceeds size limit", media.id)
            return title
        try:
            return scorm_zip_to_text(package) or title
        except (zipfile.BadZipFile, ValueError, json.JSONDecodeError) as e:
            logger.warning("Mindtickle package %s could not be read: %s", media.id, e)
            return title

    if media.transcription_url:
        raw = download(media.transcription_url, MAX_TRANSCRIPT_BYTES)
        transcript = transcript_to_text(raw) if raw else ""
        if transcript:
            return f"{title}\n{transcript}".strip()

    if media_type in _DOCUMENT_MEDIA_TYPES and url:
        raw = download(url, MAX_DOCUMENT_BYTES)
        if raw is None:
            logger.warning("Mindtickle document %s exceeds size limit", media.id)
            return title
        file_name = _file_name_from_url(url, fallback=f"{title or media.id}.pdf")
        text = extract_file_text(
            io.BytesIO(raw), file_name, break_on_unprocessable=False
        )
        return f"{title}\n{text}".strip() if text else title

    if media.htmlsrc:
        host = _embed_host(media.htmlsrc)
        return f"{title} (embedded content from {host})".strip() if host else title

    if media_type in _TRANSCRIBABLE_MEDIA_TYPES or url:
        # Video without transcript, image, or an embed page: only the title is text.
        return title
    return title
