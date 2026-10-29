"""Parser for Markdown and plain-text manuals. Needs no optional dependencies.

Conventions read from the file:
- Optional front matter between two ``---`` lines, one ``key: value`` per line.
- ``#``, ``##`` and ``###`` headings become title elements with a heading level.
- ``<!-- page: N -->`` comments set the page number for the elements that follow.
- Lines starting with ``-``, ``*`` or ``1.`` become list items.
- Consecutive lines starting with ``|`` become one table element.
- Other blank-line separated blocks become text elements.
"""

import re
from pathlib import Path

from service_manual_rag.utils.logging_config import get_logger

from .pdf_parser import ParsedDocument, ParsedElement

logger = get_logger(__name__)

HEADING = re.compile(r"^(#{1,6})\s+(.+?)\s*#*\s*$")
PAGE_MARKER = re.compile(r"^<!--\s*page:\s*(\d+)\s*-->$")
COMMENT = re.compile(r"^<!--.*-->$")
LIST_ITEM = re.compile(r"^(?:[-*]|\d+\.)\s+(.+)$")


def split_front_matter(text: str) -> tuple[dict[str, str], str]:
    """Return (front matter dict, remaining text)."""
    lines = text.splitlines()
    if not lines or lines[0].strip() != "---":
        return {}, text
    for end, line in enumerate(lines[1:], start=1):
        if line.strip() == "---":
            meta = {}
            for entry in lines[1:end]:
                key, separator, value = entry.partition(":")
                if separator:
                    meta[key.strip()] = value.strip()
            return meta, "\n".join(lines[end + 1 :])
    raise ValueError("Front matter starts with '---' but is never closed")


class _ElementBuilder:
    """Accumulates elements while tracking the current page and paragraph."""

    def __init__(self) -> None:
        self.elements: list[ParsedElement] = []
        self.page: int | None = None
        self.paragraph: list[str] = []
        self.table: list[str] = []

    def add(self, elem_type: str, content: str, metadata: dict | None = None) -> None:
        self.elements.append(
            ParsedElement(
                type=elem_type,
                content=content,
                metadata=metadata or {},
                page_number=self.page,
                category=elem_type,
            )
        )

    def flush(self) -> None:
        if self.paragraph:
            self.add("text", " ".join(self.paragraph))
            self.paragraph = []
        if self.table:
            self.add("table", "\n".join(self.table))
            self.table = []

    def feed(self, raw_line: str) -> None:
        line = raw_line.strip()
        page_match = PAGE_MARKER.match(line)
        if page_match:
            self.flush()
            self.page = int(page_match.group(1))
            return
        if not line or COMMENT.match(line):
            self.flush()
            return
        if line.startswith("|"):
            if self.paragraph:
                self.flush()
            self.table.append(line)
            return
        if self.table:
            self.flush()

        heading = HEADING.match(line)
        if heading:
            self.flush()
            self.add("title", heading.group(2), {"heading_level": len(heading.group(1))})
            return
        item = LIST_ITEM.match(line)
        if item:
            self.flush()
            self.add("list_item", line)
            return
        self.paragraph.append(line)


class MarkdownParser:
    """Parse .md and .txt manuals into the same ParsedDocument the PDF parser returns."""

    supported_suffixes = (".md", ".markdown", ".txt")

    def parse(self, file_path: Path | str) -> ParsedDocument:
        file_path = Path(file_path)
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        front_matter, body = split_front_matter(file_path.read_text(encoding="utf-8"))
        builder = _ElementBuilder()
        for line in body.splitlines():
            builder.feed(line)
        builder.flush()

        pages = [e.page_number for e in builder.elements if e.page_number]
        metadata = {
            "filename": file_path.name,
            "total_pages": max(pages) if pages else 0,
            "total_elements": len(builder.elements),
            **front_matter,
        }
        logger.info(f"Parsed {len(builder.elements)} elements from {file_path.name}")
        return ParsedDocument(
            elements=builder.elements,
            metadata=metadata,
            file_path=str(file_path),
            total_pages=metadata["total_pages"],
        )
