"""Download the public-domain PDF corpus, verify each SHA-256 checksum and index it.

The manifest lists works of the United States Government, which are not subject to
copyright in the United States (17 U.S.C. 105). The PDFs are not committed to this
repository. This script downloads them into data/public_corpus/, refuses any file whose
SHA-256 differs from the manifest, and ingests the verified files through the PDF
parser, which needs the ``pdf`` extra.
"""

import argparse
import hashlib
import json
import re
import shutil
import sys
import urllib.error
import urllib.request
from pathlib import Path

from service_manual_rag.config import MissingCredentialError
from service_manual_rag.embeddings import VectorIndexer
from service_manual_rag.processing import ChunkLevel, PDFParser, ProcessingPipeline
from service_manual_rag.utils import setup_logging

DEFAULT_MANIFEST = Path("sample_data/public_domain/manifest.json")
DEFAULT_DEST = Path("data/public_corpus")
REQUIRED_FIELDS = ("id", "title", "publisher", "url", "sha256", "bytes", "oem", "model", "license")
SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")
ID_PATTERN = re.compile(r"^[a-z0-9][a-z0-9-]*$")
DOWNLOAD_TIMEOUT_SECONDS = 120
READ_BLOCK_BYTES = 1 << 20
USER_AGENT = "service-manual-rag-corpus-fetch/0.1"


class ChecksumMismatchError(RuntimeError):
    """Raised when a downloaded file does not match the checksum in the manifest."""


# OSError covers network failures (urllib.error.URLError); ImportError a missing pdf extra.
FETCH_ERRORS = (ChecksumMismatchError, OSError, ValueError, ImportError, MissingCredentialError)


def validate_document(document: dict, position: int) -> None:
    missing = [field for field in REQUIRED_FIELDS if not document.get(field)]
    if missing:
        raise ValueError(f"Manifest document {position} is missing {', '.join(missing)}")
    if not ID_PATTERN.match(document["id"]):
        raise ValueError(f"Manifest id {document['id']!r} must be lower-case letters, digits, -")
    if not document["url"].startswith("https://"):
        raise ValueError(f"Manifest URL for {document['id']} must use https")
    if not SHA256_PATTERN.match(document["sha256"]):
        raise ValueError(f"Manifest sha256 for {document['id']} must be 64 lower-case hex digits")


def load_manifest(path: Path) -> list[dict]:
    """Read and validate the manifest; return its documents."""
    documents = json.loads(path.read_text(encoding="utf-8")).get("documents")
    if not isinstance(documents, list) or not documents:
        raise ValueError(f"{path} must hold a non-empty 'documents' array")
    for position, document in enumerate(documents):
        validate_document(document, position)
    ids = [document["id"] for document in documents]
    if len(set(ids)) != len(ids):
        raise ValueError(f"{path} has duplicate document ids")
    return documents


def sha256_of(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(READ_BLOCK_BYTES), b""):
            digest.update(block)
    return digest.hexdigest()


def download(url: str, target: Path) -> None:
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with (
        urllib.request.urlopen(request, timeout=DOWNLOAD_TIMEOUT_SECONDS) as response,
        target.open("wb") as handle,
    ):
        shutil.copyfileobj(response, handle, READ_BLOCK_BYTES)


def fetch_document(document: dict, dest: Path) -> Path:
    """Return the local path of a verified copy, downloading it when needed.

    Raises:
        ChecksumMismatchError: the downloaded bytes differ from the manifest checksum.
            Nothing is left on disk in that case.
    """
    target = dest / f"{document['id']}.pdf"
    if target.exists() and sha256_of(target) == document["sha256"]:
        print(f"{target.name}: present, checksum verified")
        return target

    dest.mkdir(parents=True, exist_ok=True)
    partial = target.with_name(target.name + ".part")
    try:
        try:
            download(document["url"], partial)
        except urllib.error.HTTPError as error:
            raise OSError(
                f"{document['url']} returned HTTP {error.code}. Download it in a browser, save "
                f"it as {target}, and run this script again; it verifies a file already in place."
            ) from error
        actual = sha256_of(partial)
        if actual != document["sha256"]:
            raise ChecksumMismatchError(
                f"{document['url']} has SHA-256 {actual}; the manifest records "
                f"{document['sha256']}. The file was not kept."
            )
        partial.replace(target)
    finally:
        partial.unlink(missing_ok=True)
    print(f"{target.name}: downloaded {target.stat().st_size:,} bytes, checksum verified")
    return target


def ingest_documents(
    documents_and_paths: list[tuple[dict, Path]],
    pipeline: ProcessingPipeline,
    indexer: VectorIndexer,
    levels: list[int],
) -> dict:
    """Parse each PDF with the pipeline and index the selected chunk levels."""
    selected_levels = [ChunkLevel(level) for level in levels]
    totals = {"files": 0, "chunks": 0, "indexed": 0}
    for document, path in documents_and_paths:
        _, chunks = pipeline.process(path, oem=document["oem"], model=document["model"])
        stats = indexer.index_chunks(chunks, levels_to_index=selected_levels)
        totals["files"] += 1
        totals["chunks"] += stats["total_chunks"]
        totals["indexed"] += stats["indexed_chunks"]
        print(f"{path.name}: {stats['indexed_chunks']} of {stats['total_chunks']} chunks indexed")
    print(f"Indexed {totals['indexed']} chunks from {totals['files']} PDF files")
    return totals


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--dest", type=Path, default=DEFAULT_DEST)
    parser.add_argument(
        "--no-ingest", action="store_true", help="Download and verify only; do not index"
    )
    parser.add_argument(
        "--strategy",
        choices=["hi_res", "fast", "ocr_only", "auto"],
        default=None,
        help="Unstructured partition strategy (default: PDF_STRATEGY)",
    )
    parser.add_argument("--levels", type=int, nargs="+", default=[4], choices=[1, 2, 3, 4])
    parser.add_argument(
        "--reset", action="store_true", help="Recreate the collection before indexing"
    )
    parser.add_argument("--log-level", default="WARNING", help="Logging level (default WARNING)")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    setup_logging(log_level=args.log_level)
    try:
        documents = load_manifest(args.manifest)
        paths = [fetch_document(document, args.dest) for document in documents]
        if args.no_ingest:
            return 0
        pdf_parser = PDFParser(strategy=args.strategy) if args.strategy else None
        indexer = VectorIndexer()
        indexer.ensure_collection(reset=args.reset)
        ingest_documents(
            list(zip(documents, paths, strict=True)),
            ProcessingPipeline(pdf_parser=pdf_parser),
            indexer,
            args.levels,
        )
    except FETCH_ERRORS as error:
        print(f"Error: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
