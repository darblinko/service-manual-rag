"""PDF parsing using Unstructured.io (optional ``pdf`` extra)."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from service_manual_rag.utils.logging_config import get_logger

logger = get_logger(__name__)

UNSTRUCTURED_TYPES = {
    "Title": "title",
    "Table": "table",
    "ListItem": "list_item",
    "Image": "image",
    "NarrativeText": "text",
    "TableChunk": "table",
}


@dataclass
class ParsedElement:
    """Parsed document element."""

    type: str  # title, text, table, list_item, image
    content: str
    metadata: dict[str, Any] = field(default_factory=dict)
    page_number: int | None = None
    category: str | None = None


@dataclass
class ParsedDocument:
    """Parsed PDF document."""

    elements: list[ParsedElement]
    metadata: dict[str, Any]
    file_path: str
    total_pages: int = 0


class PDFParser:
    """PDF parser using Unstructured.io."""

    def __init__(
        self,
        strategy: str = "hi_res",
        extract_images: bool = False,
        infer_table_structure: bool = True,
    ):
        """Initialize PDF parser.

        Args:
            strategy: Parsing strategy ('fast', 'hi_res', 'ocr_only')
            extract_images: Whether to extract images
            infer_table_structure: Whether to infer table structure

        Example:
            >>> parser = PDFParser(strategy="hi_res")
            >>> doc = parser.parse("manual.pdf")
        """
        self.strategy = strategy
        self.extract_images = extract_images
        self.infer_table_structure = infer_table_structure

        logger.info(f"Initialized PDFParser with strategy: {strategy}")

    def parse(self, file_path: Path | str) -> ParsedDocument:
        """Parse a PDF file.

        Args:
            file_path: Path to PDF file

        Returns:
            ParsedDocument: Parsed document

        Example:
            >>> parser = PDFParser()
            >>> doc = parser.parse("manual.pdf")
            >>> print(f"Parsed {len(doc.elements)} elements")
        """
        file_path = Path(file_path)

        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        try:
            from unstructured.partition.pdf import partition_pdf
        except ImportError as error:
            raise ImportError(
                "PDF parsing needs the optional dependency: pip install -e '.[pdf]'"
            ) from error

        logger.info(f"Parsing PDF: {file_path}")

        try:
            elements = partition_pdf(
                filename=str(file_path),
                strategy=self.strategy,
                extract_images_in_pdf=self.extract_images,
                infer_table_structure=self.infer_table_structure,
                extract_image_block_types=["Image", "Table"] if self.extract_images else [],
            )

            # Convert to ParsedElement objects
            parsed_elements = []
            for element in elements:
                parsed_elem = self._convert_element(element)
                if parsed_elem:
                    parsed_elements.append(parsed_elem)

            # Extract document metadata
            doc_metadata = self._extract_document_metadata(file_path, elements)

            logger.info(f"Parsed {len(parsed_elements)} elements from {file_path.name}")

            return ParsedDocument(
                elements=parsed_elements,
                metadata=doc_metadata,
                file_path=str(file_path),
                total_pages=doc_metadata.get("total_pages", 0),
            )

        except Exception as e:
            logger.error(f"Error parsing PDF {file_path}: {e}")
            raise

    def _convert_element(self, element: Any) -> ParsedElement | None:
        """Convert Unstructured element to ParsedElement.

        Args:
            element: Unstructured element

        Returns:
            ParsedElement: Converted element or None
        """
        elem_type = UNSTRUCTURED_TYPES.get(type(element).__name__, "text")

        # Extract content
        content = str(element)

        # Extract metadata
        metadata = {}
        if hasattr(element, "metadata"):
            metadata = element.metadata.to_dict() if hasattr(element.metadata, "to_dict") else {}

        # Get page number
        page_number = metadata.get("page_number")

        # Get category
        category = metadata.get("category")

        return ParsedElement(
            type=elem_type,
            content=content,
            metadata=metadata,
            page_number=page_number,
            category=category,
        )

    def _extract_document_metadata(self, file_path: Path, elements: list[Any]) -> dict[str, Any]:
        """Extract document-level metadata.

        Args:
            file_path: PDF file path
            elements: Parsed elements

        Returns:
            dict: Document metadata
        """
        # Get page numbers
        page_numbers = set()
        for elem in elements:
            if hasattr(elem, "metadata") and hasattr(elem.metadata, "page_number"):
                page_num = elem.metadata.page_number
                if page_num:
                    page_numbers.add(page_num)

        total_pages = max(page_numbers) if page_numbers else 0

        # Count element types
        type_counts = {}
        for elem in elements:
            elem_type = type(elem).__name__
            type_counts[elem_type] = type_counts.get(elem_type, 0) + 1

        metadata = {
            "filename": file_path.name,
            "total_pages": total_pages,
            "total_elements": len(elements),
            "element_type_counts": type_counts,
        }

        return metadata

    def extract_tables(self, parsed_doc: ParsedDocument) -> list[ParsedElement]:
        """Extract all tables from parsed document.

        Args:
            parsed_doc: Parsed document

        Returns:
            list: List of table elements
        """
        tables = [elem for elem in parsed_doc.elements if elem.type == "table"]
        logger.debug(f"Extracted {len(tables)} tables")
        return tables

    def extract_titles(self, parsed_doc: ParsedDocument) -> list[ParsedElement]:
        """Extract all titles from parsed document.

        Args:
            parsed_doc: Parsed document

        Returns:
            list: List of title elements
        """
        titles = [elem for elem in parsed_doc.elements if elem.type == "title"]
        logger.debug(f"Extracted {len(titles)} titles")
        return titles

    def get_elements_by_page(
        self, parsed_doc: ParsedDocument, page_number: int
    ) -> list[ParsedElement]:
        """Get all elements from a specific page.

        Args:
            parsed_doc: Parsed document
            page_number: Page number (1-indexed)

        Returns:
            list: List of elements from the page
        """
        elements = [elem for elem in parsed_doc.elements if elem.page_number == page_number]
        logger.debug(f"Found {len(elements)} elements on page {page_number}")
        return elements

    def get_text_content(self, parsed_doc: ParsedDocument) -> str:
        """Get all text content concatenated.

        Args:
            parsed_doc: Parsed document

        Returns:
            str: Concatenated text content
        """
        text_elements = [
            elem.content
            for elem in parsed_doc.elements
            if elem.type in ["text", "title", "list_item"]
        ]
        return "\n\n".join(text_elements)
