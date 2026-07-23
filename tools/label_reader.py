import fitz
from pathlib import Path


def read_document(file_path: str) -> str:
    """
    Reads a pharmaceutical label document and extracts its text.

    Currently supported:
    - PDF
    - TXT
    """

    path = Path(file_path)

    if not path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")

    extension = path.suffix.lower()

    if extension == ".pdf":
        return _read_pdf(path)

    elif extension == ".txt":
        return _read_txt(path)

    else:
        raise ValueError(
            f"Unsupported file format: {extension}. "
            "Supported formats are PDF and TXT."
        )


def _read_pdf(file_path: Path) -> str:
    """Extract text from a PDF document."""

    text = []

    with fitz.open(file_path) as document:
        for page in document:
            page_text = page.get_text("text")

            if page_text:
                text.append(page_text)

    extracted_text = "\n".join(text).strip()

    if not extracted_text:
        raise ValueError(
            f"No readable text was found in: {file_path.name}"
        )

    return extracted_text


def _read_txt(file_path: Path) -> str:
    """Read text from a TXT document."""

    return file_path.read_text(
        encoding="utf-8",
        errors="ignore"
    ).strip()