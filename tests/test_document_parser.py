import fitz
from docx import Document

from backend.app.services.document_parser import extract_text


def test_extracts_embedded_text_from_pdf(tmp_path):
    path = tmp_path / "sample.pdf"
    document = fitz.open()
    page = document.new_page()
    page.insert_text((72, 72), "Senior Data Engineer")
    document.save(path)
    document.close()

    assert "Senior Data Engineer" in extract_text(str(path), ".pdf")


def test_extracts_paragraph_text_from_docx(tmp_path):
    path = tmp_path / "sample.docx"
    document = Document()
    document.add_paragraph("Role: Product Designer")
    document.add_paragraph("5 years of experience")
    document.save(path)

    extracted = extract_text(str(path), ".docx")
    assert "Role: Product Designer" in extracted
    assert "5 years of experience" in extracted
