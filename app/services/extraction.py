"""Text extraction service — converts raw file bytes to a plain text string."""
import io
import logging
from pathlib import Path

import fitz  # PyMuPDF — wraps the MuPDF C library
from PIL import Image

logger = logging.getLogger(__name__)

# Lazy imports — pdf2image and pytesseract require system packages (tesseract, poppler).
# If those aren't installed, a module-level import would crash the entire application
# even for tenants who only upload text PDFs.
try:
    import pdf2image
    import pytesseract
    _HAS_OCR = True
except ImportError:
    _HAS_OCR = False
    logger.warning(
        "pdf2image or pytesseract not installed. "
        "OCR extraction (scanned PDFs, images) will not be available."
    )

# Threshold: if PyMuPDF extracts fewer than this many chars/page on average,
# assume the PDF is scanned and fall back to OCR.
_MIN_CHARS_PER_PAGE = 50

_SUPPORTED_EXTENSIONS = {".pdf", ".txt", ".png", ".jpg", ".jpeg"}


class ExtractionError(Exception):
    """
    Raised when text cannot be extracted from a file.

    Causes: unsupported format, empty file, scanned PDF without OCR deps, corrupt file.
    """


class ExtractionService:
    """Converts raw file bytes to a plain text string. Synchronous — call via asyncio.to_thread."""

    def extract(self, file_bytes: bytes, filename: str) -> str:
        """
        Route to the correct extractor by file extension.

        Returns a clean text string with whitespace normalized.
        Raises ExtractionError for unsupported formats, empty files, or unavailable OCR.
        """
        if not file_bytes:
            raise ExtractionError(f"File '{filename}' is empty.")

        ext = Path(filename).suffix.lower()

        if ext not in _SUPPORTED_EXTENSIONS:
            raise ExtractionError(
                f"Unsupported file type '{ext}'. "
                f"Supported: {', '.join(sorted(_SUPPORTED_EXTENSIONS))}"
            )

        logger.info("Extracting text from '%s' (%d bytes)", filename, len(file_bytes))

        if ext == ".pdf":
            text = self._extract_pdf(file_bytes)
        elif ext == ".txt":
            text = self._extract_txt(file_bytes)
        else:
            # .png, .jpg, .jpeg → direct OCR
            text = self._extract_image(file_bytes)

        text = self._clean_text(text)

        if not text:
            raise ExtractionError(
                f"No text could be extracted from '{filename}'. "
                "The file may be empty, corrupt, or contain only images."
            )

        logger.info("Extracted %d characters from '%s'", len(text), filename)
        return text

    def _extract_pdf(self, file_bytes: bytes) -> str:
        """
        Try PyMuPDF text extraction; fall back to OCR if avg chars/page < threshold.

        Average is computed over non-empty pages so blank separator pages don't
        incorrectly push a text PDF below the scanned-PDF threshold.
        """
        raw_text, page_count = self._extract_text_pdf(file_bytes)
        avg_chars = len(raw_text.strip()) / max(page_count, 1)

        if avg_chars < _MIN_CHARS_PER_PAGE:
            logger.info(
                "PDF text extraction produced %.1f chars/page (threshold: %d). "
                "Falling back to OCR.",
                avg_chars,
                _MIN_CHARS_PER_PAGE,
            )
            return self._extract_ocr_pdf(file_bytes)

        return raw_text

    def _extract_text_pdf(self, file_bytes: bytes) -> tuple[str, int]:
        """
        Extract text from a text-based PDF using PyMuPDF (fitz).

        Returns: (extracted_text, non_empty_page_count).
        sort=True gives reading-order output and handles rotated text correctly.
        """
        doc = fitz.open(stream=file_bytes, filetype="pdf")
        pages = []

        for i, page in enumerate(doc):
            text = page.get_text(sort=True)  # sort=True: reading-order, handles rotation
            if text and text.strip():
                pages.append(text)
            else:
                logger.debug("Page %d returned no text from PyMuPDF", i + 1)

        doc.close()
        return "\n\n".join(pages), len(pages)

    def _extract_ocr_pdf(self, file_bytes: bytes) -> str:
        """
        Extract text from a scanned PDF using pdf2image + pytesseract.

        Steps:
          1. pdf2image renders each PDF page at 200 DPI as a PIL Image.
          2. pytesseract runs Tesseract OCR on each image.
          3. Results are joined with double newlines.
        """
        if not _HAS_OCR:
            raise ExtractionError(
                "This PDF appears to be scanned but OCR is not available. "
                "Install tesseract-ocr and poppler-utils, then reinstall "
                "pdf2image and pytesseract."
            )

        images: list[Image.Image] = pdf2image.convert_from_bytes(
            file_bytes, dpi=200
        )
        pages = []

        for i, image in enumerate(images):
            text = pytesseract.image_to_string(image, lang="eng")
            if text.strip():
                pages.append(text)
            else:
                logger.debug("OCR page %d returned no text", i + 1)

        return "\n\n".join(pages)

    def _extract_image(self, file_bytes: bytes) -> str:
        """Extract text from a PNG/JPEG via OCR (same Tesseract engine as _extract_ocr_pdf)."""
        if not _HAS_OCR:
            raise ExtractionError(
                "OCR is not available. Install tesseract-ocr, then reinstall pytesseract."
            )

        image = Image.open(io.BytesIO(file_bytes))
        return pytesseract.image_to_string(image, lang="eng")

    def _extract_txt(self, file_bytes: bytes) -> str:
        """
        Decode a plain text file as UTF-8.

        errors="replace" substitutes U+FFFD for undecodable bytes rather than raising
        UnicodeDecodeError — handles files with rogue Windows-1252 bytes.
        """
        return file_bytes.decode("utf-8", errors="replace")

    def _clean_text(self, text: str) -> str:
        """
        Normalize extracted text before passing to the chunker.

        - Replace \\f (PDF form feed) with newline.
        - Collapse 3+ consecutive newlines to 2 (preserve paragraphs, remove excess).
        - Strip leading/trailing whitespace.
        """
        import re
        text = text.replace("\f", "\n")
        text = re.sub(r"\n{3,}", "\n\n", text)
        return text.strip()
