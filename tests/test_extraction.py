from unittest.mock import MagicMock, patch

import pytest

from app.services.extraction import ExtractionError, ExtractionService


@pytest.fixture
def service():
    return ExtractionService()


def _make_pdf_doc(pages_text: list[str]):
    """Build a mock fitz.Document (PyMuPDF) that returns specific text per page."""
    mock_pages = []
    for text in pages_text:
        page = MagicMock()
        page.get_text.return_value = text
        mock_pages.append(page)

    doc = MagicMock()
    doc.__iter__ = MagicMock(side_effect=lambda: iter(mock_pages))
    return doc


def test_txt_file_returns_decoded_string(service):
    content = "Hello, this is a plain text document."
    result = service.extract(content.encode("utf-8"), "doc.txt")
    assert result == content


def test_txt_file_preserves_content(service):
    content = "Line one.\n\nLine two.\n\nLine three."
    result = service.extract(content.encode("utf-8"), "notes.txt")
    assert "Line one." in result
    assert "Line two." in result
    assert "Line three." in result


def test_txt_file_strips_leading_trailing_whitespace(service):
    content = "\n\n  actual content here  \n\n"
    result = service.extract(content.encode("utf-8"), "padded.txt")
    assert result == "actual content here"


def test_text_pdf_extracts_single_page(service):
    mock_doc = _make_pdf_doc(["Contract clause one. Payment is due within 30 days."])

    with patch("app.services.extraction.fitz") as mock_fitz:
        mock_fitz.open.return_value = mock_doc
        result = service.extract(b"%PDF-fake", "contract.pdf")

    assert "Payment is due within 30 days" in result


def test_text_pdf_joins_multiple_pages_with_double_newline(service):
    mock_doc = _make_pdf_doc([
        "This is page one content with enough text to exceed the sparse threshold.",
        "This is page two content with enough text to exceed the sparse threshold.",
        "This is page three content with enough text to exceed the sparse threshold.",
    ])

    with patch("app.services.extraction.fitz") as mock_fitz:
        mock_fitz.open.return_value = mock_doc
        result = service.extract(b"%PDF-fake", "multipage.pdf")

    assert "page one" in result
    assert "page two" in result
    assert "page three" in result
    # Pages separated by double newline (paragraph break)
    assert "\n\n" in result


def test_text_pdf_skips_empty_pages(service):
    long_page_one = "Real content on page one with enough characters to exceed the sparse detection threshold."
    long_page_four = "Real content on page four with enough characters to exceed the sparse detection threshold."
    mock_doc = _make_pdf_doc([
        long_page_one,
        "",           # blank page — should be skipped
        "   \n  ",   # whitespace-only page — should be skipped
        long_page_four,
    ])

    with patch("app.services.extraction.fitz") as mock_fitz:
        mock_fitz.open.return_value = mock_doc
        result = service.extract(b"%PDF-fake", "doc.pdf")

    assert "page one" in result
    assert "page four" in result
    assert "\n\n\n" not in result


def test_sparse_text_pdf_falls_back_to_ocr(service):
    # PyMuPDF returns almost nothing — 2 chars, rest empty
    mock_doc = _make_pdf_doc(["ab"] + [""] * 4)
    mock_images = [MagicMock()]  # one fake PIL image

    with patch("app.services.extraction.fitz") as mock_fitz, \
         patch("app.services.extraction._HAS_OCR", True), \
         patch("app.services.extraction.pdf2image") as mock_pdf2img, \
         patch("app.services.extraction.pytesseract") as mock_tess:

        mock_fitz.open.return_value = mock_doc
        mock_pdf2img.convert_from_bytes.return_value = mock_images
        mock_tess.image_to_string.return_value = "OCR extracted text from scanned page."

        result = service.extract(b"%PDF-fake", "scanned.pdf")

    assert "OCR extracted text" in result
    mock_pdf2img.convert_from_bytes.assert_called_once()
    mock_tess.image_to_string.assert_called_once()


def test_rich_text_pdf_does_not_call_ocr(service):
    page_text = "A" * 500
    mock_doc = _make_pdf_doc([page_text, page_text])

    with patch("app.services.extraction.fitz") as mock_fitz, \
         patch("app.services.extraction._HAS_OCR", True), \
         patch("app.services.extraction.pdf2image") as mock_pdf2img:

        mock_fitz.open.return_value = mock_doc
        result = service.extract(b"%PDF-fake", "text_doc.pdf")

    mock_pdf2img.convert_from_bytes.assert_not_called()
    assert "A" * 100 in result


def test_png_image_uses_ocr(service):
    fake_image_bytes = b"\x89PNG\r\n\x1a\n" + b"\x00" * 100  # PNG magic bytes + padding

    with patch("app.services.extraction._HAS_OCR", True), \
         patch("PIL.Image.open") as mock_open, \
         patch("app.services.extraction.pytesseract") as mock_tess:

        mock_open.return_value = MagicMock()
        mock_tess.image_to_string.return_value = "Text found in image."

        result = service.extract(fake_image_bytes, "scan.png")

    assert "Text found in image" in result
    mock_tess.image_to_string.assert_called_once()


def test_jpeg_image_uses_ocr(service):
    fake_bytes = b"\xff\xd8\xff" + b"\x00" * 50  # JPEG magic bytes

    with patch("app.services.extraction._HAS_OCR", True), \
         patch("PIL.Image.open") as mock_open, \
         patch("app.services.extraction.pytesseract") as mock_tess:

        mock_open.return_value = MagicMock()
        mock_tess.image_to_string.return_value = "JPEG OCR content."

        result = service.extract(fake_bytes, "photo.jpg")

    assert "JPEG OCR content" in result


def test_empty_bytes_raises_extraction_error(service):
    with pytest.raises(ExtractionError, match="empty"):
        service.extract(b"", "empty.pdf")


def test_unsupported_extension_raises_extraction_error(service):
    with pytest.raises(ExtractionError, match=r"\.docx"):
        service.extract(b"fake content", "report.docx")


def test_unsupported_extension_error_lists_supported_formats(service):
    with pytest.raises(ExtractionError) as exc_info:
        service.extract(b"data", "file.exe")
    error_msg = str(exc_info.value)
    assert ".pdf" in error_msg
    assert ".txt" in error_msg


def test_pdf_with_no_extractable_text_raises(service):
    mock_doc = _make_pdf_doc([""])  # PyMuPDF gets nothing

    with patch("app.services.extraction.fitz") as mock_fitz, \
         patch("app.services.extraction._HAS_OCR", True), \
         patch("app.services.extraction.pdf2image") as mock_pdf2img, \
         patch("app.services.extraction.pytesseract") as mock_tess:

        mock_fitz.open.return_value = mock_doc
        mock_pdf2img.convert_from_bytes.return_value = [MagicMock()]
        mock_tess.image_to_string.return_value = ""  # OCR also gets nothing

        with pytest.raises(ExtractionError, match="No text"):
            service.extract(b"%PDF-fake", "blank.pdf")


def test_clean_text_removes_form_feeds(service):
    content = "Page one.\fPage two."
    result = service._clean_text(content)
    assert "\f" not in result
    assert "Page one." in result
    assert "Page two." in result


def test_clean_text_collapses_excessive_blank_lines(service):
    content = "Paragraph one.\n\n\n\n\nParagraph two."
    result = service._clean_text(content)
    assert "\n\n\n" not in result
    assert "Paragraph one." in result
    assert "Paragraph two." in result
