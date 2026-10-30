from io import BytesIO
from pathlib import Path
from zipfile import BadZipFile, ZipFile

import fitz
from docx import Document


def validate_document_content(file_bytes: bytes, file_type: str) -> bool:
    """Check that the payload resembles the claimed PDF or DOCX format."""
    extension = file_type.lower()
    if extension == ".pdf":
        return b"%PDF-" in file_bytes[:1024]

    if extension == ".docx":
        try:
            with ZipFile(BytesIO(file_bytes)) as archive:
                names = set(archive.namelist())
                return {
                    "[Content_Types].xml",
                    "word/document.xml",
                }.issubset(names)
        except (BadZipFile, OSError):
            return False

    return False


def extract_text_from_pdf(file_path: str) -> str:
    document = fitz.open(file_path)

    text_parts = []

    for page in document:
        text_parts.append(page.get_text())

    document.close()

    return "\n".join(text_parts).strip()


def extract_text_from_docx(file_path: str) -> str:
    document = Document(file_path)

    paragraphs = [
        paragraph.text.strip()
        for paragraph in document.paragraphs
        if paragraph.text.strip()
    ]

    return "\n".join(paragraphs)


def extract_text(file_path: str, file_type: str) -> str:
    file_type = file_type.lower()

    if file_type == ".pdf":
        return extract_text_from_pdf(file_path)

    if file_type == ".docx":
        return extract_text_from_docx(file_path)

    raise ValueError(
        "Unsupported file type. Only PDF and DOCX are supported."
    )
