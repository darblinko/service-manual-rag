"""Document processing: parse a manual and build its chunk hierarchy."""

from collections import Counter
from pathlib import Path

from service_manual_rag.config import get_settings
from service_manual_rag.utils.logging_config import get_logger

from .hierarchical_chunker import Chunk, ChunkLevel, HierarchicalChunker
from .markdown_parser import MarkdownParser
from .pdf_parser import ParsedDocument, PDFParser

logger = get_logger(__name__)


def default_chunker() -> HierarchicalChunker:
    """Chunker configured from settings; offline mode approximates token counts."""
    settings = get_settings()
    return HierarchicalChunker(
        chunk_size=settings.chunk_size_tokens,
        chunk_overlap=settings.chunk_overlap_tokens,
        encoding_name="cl100k_base" if settings.use_openai else None,
    )


class ProcessingPipeline:
    """Parse .pdf, .md or .txt manuals and chunk them."""

    def __init__(
        self,
        pdf_parser: PDFParser | None = None,
        text_parser: MarkdownParser | None = None,
        chunker: HierarchicalChunker | None = None,
    ):
        self.pdf_parser = pdf_parser or PDFParser(strategy="hi_res")
        self.text_parser = text_parser or MarkdownParser()
        self.chunker = chunker or default_chunker()

    def parse(self, file_path: Path | str) -> ParsedDocument:
        file_path = Path(file_path)
        suffix = file_path.suffix.lower()
        if suffix == ".pdf":
            return self.pdf_parser.parse(file_path)
        if suffix in MarkdownParser.supported_suffixes:
            return self.text_parser.parse(file_path)
        raise ValueError(f"Unsupported file type '{suffix}' for {file_path}")

    def process(
        self,
        file_path: Path | str,
        oem: str | None = None,
        model: str | None = None,
        doc_id: str | None = None,
    ) -> tuple[ParsedDocument, list[Chunk]]:
        """Parse and chunk one file.

        ``oem`` and ``model`` default to the file's front matter; one of the two
        sources must provide them.
        """
        parsed_doc = self.parse(file_path)
        oem = oem or parsed_doc.metadata.get("oem")
        model = model or parsed_doc.metadata.get("model")
        if not oem or not model:
            raise ValueError(f"OEM and model are required for {file_path}")

        chunks = self.chunker.chunk_document(parsed_doc, oem, model, doc_id)
        logger.info(
            f"Processed {file_path}: {len(parsed_doc.elements)} elements, {len(chunks)} chunks"
        )
        return parsed_doc, chunks

    @staticmethod
    def get_statistics(parsed_doc: ParsedDocument, chunks: list[Chunk]) -> dict:
        semantic = [c for c in chunks if c.level == ChunkLevel.SEMANTIC]
        token_counts = [c.token_count for c in semantic]
        return {
            "total_elements": len(parsed_doc.elements),
            "total_chunks": len(chunks),
            "chunks_by_level": dict(Counter(c.level for c in chunks)),
            "semantic_chunks_by_type": dict(Counter(c.element_type for c in semantic)),
            "average_tokens_per_chunk": sum(token_counts) / len(token_counts)
            if token_counts
            else 0,
            "total_pages": parsed_doc.total_pages,
        }
