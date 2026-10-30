from io import BytesIO
from zipfile import ZipFile

from backend.app.services.document_parser import validate_document_content


def test_accepts_pdf_signature():
    assert validate_document_content(b"%PDF-1.7\nbody", ".pdf")


def test_rejects_non_pdf_payload_with_pdf_extension():
    assert not validate_document_content(b"not a pdf", ".pdf")


def test_accepts_docx_container_with_required_parts():
    buffer = BytesIO()
    with ZipFile(buffer, "w") as archive:
        archive.writestr("[Content_Types].xml", "<Types/>")
        archive.writestr("word/document.xml", "<document/>")

    assert validate_document_content(buffer.getvalue(), ".docx")


def test_rejects_arbitrary_zip_as_docx():
    buffer = BytesIO()
    with ZipFile(buffer, "w") as archive:
        archive.writestr("payload.txt", "not a Word document")

    assert not validate_document_content(buffer.getvalue(), ".docx")
