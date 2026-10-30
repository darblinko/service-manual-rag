"""Offline tests for scripts/fetch_public_corpus.py; downloads are replaced by local bytes."""

import hashlib
import importlib.util
import json
import urllib.error
from pathlib import Path

import pytest

from service_manual_rag.embeddings import VectorIndexer
from service_manual_rag.processing import ParsedDocument, ParsedElement, ProcessingPipeline

REPO_ROOT = Path(__file__).parents[1]
MANIFEST = REPO_ROOT / "sample_data" / "public_domain" / "manifest.json"
PDF_BYTES = b"%PDF-1.4 test document\n"


def load_fetch_script():
    path = REPO_ROOT / "scripts" / "fetch_public_corpus.py"
    spec = importlib.util.spec_from_file_location("fetch_public_corpus", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


fetch = load_fetch_script()


def make_document(**overrides):
    document = {
        "id": "test-doc",
        "title": "Test handbook",
        "publisher": "Test agency",
        "url": "https://example.gov/test.pdf",
        "sha256": hashlib.sha256(PDF_BYTES).hexdigest(),
        "bytes": len(PDF_BYTES),
        "oem": "Agency",
        "model": "TEST-1",
        "license": "Public domain",
    }
    return {**document, **overrides}


def serve_bytes(monkeypatch, payload=PDF_BYTES):
    """Replace the network download with a write of ``payload``; return the call log."""
    calls = []

    def fake_download(url, target):
        calls.append(url)
        target.write_bytes(payload)

    monkeypatch.setattr(fetch, "download", fake_download)
    return calls


def test_manifest_lists_federal_works_with_checksums():
    documents = fetch.load_manifest(MANIFEST)

    assert 2 <= len(documents) <= 3
    for document in documents:
        assert document["url"].startswith("https://www.faa.gov/")
        assert "17 U.S.C. 105" in document["license"]
        assert document["bytes"] > 0


def test_fetch_downloads_and_keeps_a_verified_file(monkeypatch, tmp_path):
    calls = serve_bytes(monkeypatch)

    path = fetch.fetch_document(make_document(), tmp_path)

    assert path == tmp_path / "test-doc.pdf"
    assert path.read_bytes() == PDF_BYTES
    assert calls == ["https://example.gov/test.pdf"]


def test_fetch_rejects_a_checksum_mismatch_and_keeps_nothing(monkeypatch, tmp_path):
    serve_bytes(monkeypatch, payload=b"%PDF-1.4 tampered\n")

    with pytest.raises(fetch.ChecksumMismatchError, match="does not match|manifest records"):
        fetch.fetch_document(make_document(), tmp_path)

    assert list(tmp_path.iterdir()) == []


def test_fetch_skips_the_download_when_the_file_matches(monkeypatch, tmp_path):
    (tmp_path / "test-doc.pdf").write_bytes(PDF_BYTES)
    calls = serve_bytes(monkeypatch)

    fetch.fetch_document(make_document(), tmp_path)

    assert calls == []


def test_fetch_replaces_a_corrupted_local_copy(monkeypatch, tmp_path):
    (tmp_path / "test-doc.pdf").write_bytes(b"truncated")
    calls = serve_bytes(monkeypatch)

    path = fetch.fetch_document(make_document(), tmp_path)

    assert calls == ["https://example.gov/test.pdf"]
    assert path.read_bytes() == PDF_BYTES


def test_fetch_explains_a_refused_download(monkeypatch, tmp_path):
    def refuse(url, target):
        target.write_bytes(b"partial")
        raise urllib.error.HTTPError(url, 403, "Forbidden", hdrs=None, fp=None)

    monkeypatch.setattr(fetch, "download", refuse)

    with pytest.raises(OSError, match="HTTP 403") as raised:
        fetch.fetch_document(make_document(), tmp_path)

    assert str(tmp_path / "test-doc.pdf") in str(raised.value)
    assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize(
    ("override", "message"),
    [
        ({"url": "http://example.gov/test.pdf"}, "https"),
        ({"sha256": "ABC"}, "64 lower-case hex"),
        ({"license": ""}, "missing license"),
        ({"id": "../escape"}, "lower-case letters"),
    ],
)
def test_manifest_validation_rejects_bad_entries(tmp_path, override, message):
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({"documents": [make_document(**override)]}))

    with pytest.raises(ValueError, match=message):
        fetch.load_manifest(manifest)


class StubPdfParser:
    """PDF parser stand-in that returns fixed elements in the Unstructured element shape."""

    def __init__(self):
        self.parsed = []

    def parse(self, file_path):
        self.parsed.append(Path(file_path))
        elements = [
            ParsedElement("title", "Chapter 1 Engine Troubleshooting", page_number=1),
            ParsedElement("text", "Engine fails to start. Check the fuel supply.", page_number=1),
            ParsedElement("title", "Section 2 Lubrication", page_number=2),
            ParsedElement("text", "Low oil pressure. Check the oil level.", page_number=2),
        ]
        return ParsedDocument(elements, {"filename": Path(file_path).name}, str(file_path), 2)


def test_ingest_parses_each_pdf_with_manifest_metadata(tmp_path):
    parser = StubPdfParser()
    indexer = VectorIndexer()
    document = make_document()
    pdf_path = tmp_path / "test-doc.pdf"

    totals = fetch.ingest_documents(
        [(document, pdf_path)], ProcessingPipeline(pdf_parser=parser), indexer, levels=[4]
    )

    assert parser.parsed == [pdf_path]
    assert totals["files"] == 1
    assert totals["indexed"] == indexer.qdrant_manager.count_points() > 0
    hits = indexer.search("oil pressure", limit=1)
    assert hits[0]["payload"]["oem"] == "Agency"
    assert hits[0]["payload"]["model"] == "TEST-1"
    assert hits[0]["payload"]["page_numbers"] == [2]


class RecordingPopulator:
    def __init__(self):
        self.batches = []

    def populate(self, chunks):
        self.batches.append(chunks)
        return {"sections_created": len(chunks)}

    def close(self):
        self.closed = True


def test_ingest_passes_every_chunk_level_to_the_graph_populator(tmp_path):
    populator = RecordingPopulator()
    fetch.ingest_documents(
        [(make_document(), tmp_path / "test-doc.pdf")],
        ProcessingPipeline(pdf_parser=StubPdfParser()),
        VectorIndexer(),
        levels=[4],
        populator=populator,
    )

    assert len(populator.batches) == 1
    assert {int(chunk.level) for chunk in populator.batches[0]} >= {1, 4}


def test_main_without_ingest_downloads_every_manifest_entry(monkeypatch, tmp_path):
    manifest = tmp_path / "manifest.json"
    second = make_document(id="second-doc", url="https://example.gov/second.pdf")
    manifest.write_text(json.dumps({"documents": [make_document(), second]}))
    calls = serve_bytes(monkeypatch)

    status = fetch.main(
        ["--manifest", str(manifest), "--dest", str(tmp_path / "pdf"), "--no-ingest"]
    )

    assert status == 0
    assert calls == ["https://example.gov/test.pdf", "https://example.gov/second.pdf"]
    assert sorted(p.name for p in (tmp_path / "pdf").iterdir()) == [
        "second-doc.pdf",
        "test-doc.pdf",
    ]


def test_main_reports_a_checksum_mismatch(monkeypatch, tmp_path, capsys):
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({"documents": [make_document()]}))
    serve_bytes(monkeypatch, payload=b"other bytes")

    status = fetch.main(["--manifest", str(manifest), "--dest", str(tmp_path), "--no-ingest"])

    assert status == 1
    assert "SHA-256" in capsys.readouterr().err
