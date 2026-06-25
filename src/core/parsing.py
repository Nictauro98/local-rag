import io

from docx import Document
from pypdf import PdfReader
from unstructured.partition.auto import partition


def parse(filename: str, data: bytes) -> str:
    """Extract plain text from a document. Dispatches by file extension."""
    ext = filename.rsplit(".", 1)[-1].lower() if "." in filename else ""

    if ext == "pdf":
        return _parse_pdf(data)
    if ext in ("docx", "doc"):
        return _parse_docx(data)
    if ext == "txt":
        return data.decode("utf-8", errors="replace")

    return _parse_unstructured(data, filename)


def _parse_pdf(data: bytes) -> str:
    reader = PdfReader(io.BytesIO(data))
    return "\n".join(page.extract_text() or "" for page in reader.pages)


def _parse_docx(data: bytes) -> str:
    doc = Document(io.BytesIO(data))
    return "\n".join(p.text for p in doc.paragraphs)


def _parse_unstructured(data: bytes, filename: str) -> str:
    elements = partition(file=io.BytesIO(data), metadata_filename=filename)
    return "\n".join(str(e) for e in elements)
