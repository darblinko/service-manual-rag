"""Hierarchical chunking for service manuals.

Four levels:
- Level 1: document (manual metadata)
- Level 2: major sections (Safety, Maintenance, Troubleshooting, ...)
- Level 3: subsections (headings below the major sections)
- Level 4: semantic chunks (groups of paragraphs, lists and tables up to chunk_size tokens)
"""

from dataclasses import dataclass, field
from enum import IntEnum
from pathlib import Path
from typing import Any
from uuid import NAMESPACE_URL, uuid5

from service_manual_rag.utils.logging_config import get_logger

from .pdf_parser import ParsedDocument, ParsedElement

logger = get_logger(__name__)

MAJOR_SECTION_KEYWORDS = (
    "introduction",
    "installation",
    "maintenance",
    "operation",
    "troubleshooting",
    "specifications",
    "safety",
    "parts",
    "service",
    "repair",
)
SECTION_PREVIEW_ELEMENTS = 5
SUBSECTION_PREVIEW_ELEMENTS = 3


def stable_chunk_id(*parts: object) -> str:
    """UUID derived from the chunk's position and content, so re-ingesting a file is idempotent."""
    return str(uuid5(NAMESPACE_URL, "/".join(str(part) for part in parts)))


class ChunkLevel(IntEnum):
    """Chunk hierarchy levels."""

    DOCUMENT = 1
    SECTION = 2
    SUBSECTION = 3
    SEMANTIC = 4


@dataclass
class Chunk:
    """One node of the chunk hierarchy. ``level`` is stored as a plain int."""

    chunk_id: str
    level: int
    content: str
    parent_id: str | None = None
    children_ids: list[str] = field(default_factory=list)
    oem: str = ""
    model: str = ""
    section_name: str | None = None
    page_numbers: list[int] = field(default_factory=list)
    element_type: str = "text"
    file_path: str = ""
    doc_id: str | None = None
    token_count: int = 0
    table_data: dict[str, Any] | None = None

    def __post_init__(self):
        self.level = int(self.level)


@dataclass
class _Group:
    name: str
    elements: list[ParsedElement]


def is_major_section(element: ParsedElement) -> bool:
    """Heading level 1 or 2 when the parser records it, else a keyword match on the title."""
    heading_level = element.metadata.get("heading_level")
    if heading_level is not None:
        return heading_level <= 2
    title = element.content.lower()
    return any(keyword in title for keyword in MAJOR_SECTION_KEYWORDS)


def page_numbers_of(elements: list[ParsedElement]) -> list[int]:
    return sorted({e.page_number for e in elements if e.page_number})


def split_on_titles(elements: list[ParsedElement], is_boundary, default_name: str) -> list[_Group]:
    """Split elements into groups that start at each boundary title.

    Elements before the first boundary form a group named ``default_name``.
    """
    groups: list[_Group] = []
    for element in elements:
        if element.type == "title" and is_boundary(element):
            groups.append(_Group(element.content, [element]))
        elif groups:
            groups[-1].elements.append(element)
        else:
            groups.append(_Group(default_name, [element]))
    return groups


class HierarchicalChunker:
    """Build the four-level chunk hierarchy from a parsed document."""

    def __init__(
        self,
        chunk_size: int = 500,
        chunk_overlap: int = 50,
        encoding_name: str | None = "cl100k_base",
    ):
        """
        Args:
            chunk_size: Target size of level 4 chunks in tokens
            chunk_overlap: When > 0, each new level 4 chunk repeats the previous
                chunk's last element
            encoding_name: tiktoken encoding name, or None to approximate token
                counts from word counts (deterministic and offline)
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.tokenizer = None
        if encoding_name:
            try:
                import tiktoken

                self.tokenizer = tiktoken.get_encoding(encoding_name)
            # tiktoken downloads encodings on first use; fall back when offline.
            except Exception as error:  # noqa: BLE001
                logger.warning(
                    f"Could not load tokenizer {encoding_name} ({error}); "
                    "token counts will be approximated from word counts"
                )

    def chunk_document(
        self, parsed_doc: ParsedDocument, oem: str, model: str, doc_id: str | None = None
    ) -> list[Chunk]:
        """Return chunks of all four levels with parent and child links set."""
        doc_id = doc_id or stable_chunk_id(oem, model, Path(parsed_doc.file_path).name)
        doc_chunk = self._make_chunk(
            chunk_id=doc_id,
            level=ChunkLevel.DOCUMENT,
            content=(
                f"{oem} {model} Service Manual\n"
                f"Total Pages: {parsed_doc.total_pages}\n"
                f"Total Elements: {len(parsed_doc.elements)}\n"
            ),
            oem=oem,
            model=model,
            doc_id=doc_id,
            file_path=parsed_doc.file_path,
        )
        all_chunks = [doc_chunk]

        sections = split_on_titles(parsed_doc.elements, is_major_section, "Introduction")
        for section in sections:
            section_chunk = self._group_chunk(section, ChunkLevel.SECTION, doc_chunk, oem, model)
            all_chunks.append(section_chunk)
            doc_chunk.children_ids.append(section_chunk.chunk_id)

            subsections = split_on_titles(
                section.elements, lambda e: not is_major_section(e), section.name
            )
            for subsection in subsections:
                if all(e.type == "title" for e in subsection.elements):
                    continue
                sub_chunk = self._group_chunk(
                    subsection, ChunkLevel.SUBSECTION, section_chunk, oem, model
                )
                semantic = self._semantic_chunks(subsection, sub_chunk, oem, model)
                sub_chunk.children_ids.extend(c.chunk_id for c in semantic)
                section_chunk.children_ids.append(sub_chunk.chunk_id)
                all_chunks.append(sub_chunk)
                all_chunks.extend(semantic)

        counts = {level.name: sum(c.level == level for c in all_chunks) for level in ChunkLevel}
        logger.info(f"Created {len(all_chunks)} chunks for {oem} {model}: {counts}")
        return all_chunks

    def _group_chunk(
        self, group: _Group, level: ChunkLevel, parent: Chunk, oem: str, model: str
    ) -> Chunk:
        preview = (
            SECTION_PREVIEW_ELEMENTS if level == ChunkLevel.SECTION else SUBSECTION_PREVIEW_ELEMENTS
        )
        content = f"{group.name}\n\n" + "\n\n".join(e.content for e in group.elements[:preview])
        return self._make_chunk(
            chunk_id=stable_chunk_id(parent.chunk_id, int(level), group.name, content),
            level=level,
            content=content,
            parent_id=parent.chunk_id,
            oem=oem,
            model=model,
            section_name=group.name,
            page_numbers=page_numbers_of(group.elements),
            doc_id=parent.doc_id,
            file_path=parent.file_path,
        )

    def _semantic_chunks(
        self, subsection: _Group, parent: Chunk, oem: str, model: str
    ) -> list[Chunk]:
        """Group consecutive elements until adding one would exceed chunk_size."""
        batches: list[list[ParsedElement]] = []
        current: list[ParsedElement] = []
        current_tokens = 0
        for element in subsection.elements:
            element_tokens = self._count_tokens(element.content)
            if current and current_tokens + element_tokens > self.chunk_size:
                batches.append(current)
                current = [current[-1]] if self.chunk_overlap > 0 else []
                current_tokens = sum(self._count_tokens(e.content) for e in current)
            current.append(element)
            current_tokens += element_tokens
        if current:
            batches.append(current)

        return [
            self._semantic_chunk(batch, parent, oem, model, subsection.name) for batch in batches
        ]

    def _semantic_chunk(
        self,
        elements: list[ParsedElement],
        parent: Chunk,
        oem: str,
        model: str,
        section_name: str,
    ) -> Chunk:
        content = "\n\n".join(e.content for e in elements)
        element_type = "text"
        if any(e.type == "table" for e in elements):
            element_type = "table"
        elif any(e.type == "list_item" for e in elements):
            element_type = "procedure"
        return self._make_chunk(
            chunk_id=stable_chunk_id(parent.chunk_id, int(ChunkLevel.SEMANTIC), content),
            level=ChunkLevel.SEMANTIC,
            content=content,
            parent_id=parent.chunk_id,
            oem=oem,
            model=model,
            section_name=section_name,
            page_numbers=page_numbers_of(elements),
            element_type=element_type,
            doc_id=parent.doc_id,
            file_path=parent.file_path,
        )

    def _make_chunk(self, **fields: Any) -> Chunk:
        return Chunk(token_count=self._count_tokens(fields["content"]), **fields)

    def _count_tokens(self, text: str) -> int:
        if self.tokenizer:
            return len(self.tokenizer.encode(text))
        return int(len(text.split()) / 0.75)
