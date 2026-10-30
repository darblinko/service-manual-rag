import pytest

from service_manual_rag.processing import (
    ChunkLevel,
    HierarchicalChunker,
    MarkdownParser,
    ProcessingPipeline,
)

MANUAL = """---
oem: ExampleCo
model: T-1
title: T-1 Test Unit
---
<!-- page: 1 -->
Preface text that sits before the first section.

## Maintenance

### Bearing Lubrication

Grease the bearings every 2,000 hours with 15 g of lithium grease.

1. Stop the unit.
2. Clean the grease fitting.

<!-- page: 2 -->
## Specifications

### Ratings

| Item | Value |
|---|---|
| Rated flow | 200 L/min |
"""


def write_manual(tmp_path, text=MANUAL):
    path = tmp_path / "t-1.md"
    path.write_text(text, encoding="utf-8")
    return path


def test_markdown_parser_reads_front_matter_pages_and_element_types(tmp_path):
    parsed = MarkdownParser().parse(write_manual(tmp_path))

    assert parsed.metadata["oem"] == "ExampleCo"
    assert parsed.metadata["model"] == "T-1"
    assert parsed.total_pages == 2
    types = [e.type for e in parsed.elements]
    assert types.count("title") == 4
    assert types.count("list_item") == 2
    assert types.count("table") == 1
    table = next(e for e in parsed.elements if e.type == "table")
    assert table.page_number == 2
    assert "200 L/min" in table.content


def test_chunker_keeps_content_before_first_section(tmp_path):
    parsed = MarkdownParser().parse(write_manual(tmp_path))
    chunks = HierarchicalChunker(encoding_name=None).chunk_document(parsed, "ExampleCo", "T-1")

    sections = [c for c in chunks if c.level == ChunkLevel.SECTION]
    assert [c.section_name for c in sections] == ["Introduction", "Maintenance", "Specifications"]
    semantic = [c for c in chunks if c.level == ChunkLevel.SEMANTIC]
    assert any("Preface text" in c.content for c in semantic)


def test_semantic_chunks_carry_the_subsection_heading_and_type(tmp_path):
    parsed = MarkdownParser().parse(write_manual(tmp_path))
    chunks = HierarchicalChunker(encoding_name=None).chunk_document(parsed, "ExampleCo", "T-1")

    by_section = {c.section_name: c for c in chunks if c.level == ChunkLevel.SEMANTIC}
    assert by_section["Bearing Lubrication"].element_type == "procedure"
    assert by_section["Ratings"].element_type == "table"
    assert by_section["Ratings"].page_numbers == [2]


def test_chunk_ids_are_deterministic_and_links_are_consistent(tmp_path):
    parsed = MarkdownParser().parse(write_manual(tmp_path))
    chunker = HierarchicalChunker(encoding_name=None)
    first = chunker.chunk_document(parsed, "ExampleCo", "T-1")
    second = chunker.chunk_document(parsed, "ExampleCo", "T-1")

    assert [c.chunk_id for c in first] == [c.chunk_id for c in second]
    ids = {c.chunk_id: c for c in first}
    assert len(ids) == len(first)
    for chunk in first:
        for child_id in chunk.children_ids:
            assert ids[child_id].parent_id == chunk.chunk_id


def test_small_chunk_size_splits_a_subsection(tmp_path):
    parsed = MarkdownParser().parse(write_manual(tmp_path))
    chunks = HierarchicalChunker(chunk_size=10, chunk_overlap=0, encoding_name=None)
    semantic = [
        c
        for c in chunks.chunk_document(parsed, "ExampleCo", "T-1")
        if c.level == ChunkLevel.SEMANTIC and c.section_name == "Bearing Lubrication"
    ]
    assert len(semantic) > 1


def test_pipeline_takes_oem_and_model_from_front_matter(tmp_path):
    _, chunks = ProcessingPipeline().process(write_manual(tmp_path))
    assert {c.oem for c in chunks} == {"ExampleCo"}
    assert {c.model for c in chunks} == {"T-1"}


def test_pipeline_requires_oem_and_model(tmp_path):
    path = write_manual(tmp_path, "## Safety\n\nWear gloves.\n")
    with pytest.raises(ValueError, match="OEM and model"):
        ProcessingPipeline().process(path)
