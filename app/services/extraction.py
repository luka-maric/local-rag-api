import io
import logging
from pathlib import Path

import fitz
from PIL import Image

logger = logging.getLogger(__name__)

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

_MIN_CHARS_PER_PAGE = 50
_SUPPORTED_EXTENSIONS = {".pdf", ".txt", ".png", ".jpg", ".jpeg"}


class ExtractionError(Exception):
    pass


class ExtractionService:
    def extract(self, file_bytes: bytes, filename: str) -> str:
        if not file_bytes:
            raise ExtractionError(f"File '{filename}' is empty.")

        ext = Path(filename).suffix.lower()

        if ext not in _SUPPORTED_EXTENSIONS:
            raise ExtractionError(
                f"Unsupported file type '{ext}'. Supported: {', '.join(sorted(_SUPPORTED_EXTENSIONS))}"
            )

        logger.info("Extracting text from '%s' (%d bytes)", filename, len(file_bytes))

        if ext == ".pdf":
            text = self._extract_pdf(file_bytes)
        elif ext == ".txt":
            text = self._extract_txt(file_bytes)
        else:
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
        raw_text, page_count = self._extract_text_pdf(file_bytes)
        avg_chars = len(raw_text.strip()) / max(page_count, 1)

        if avg_chars < _MIN_CHARS_PER_PAGE:
            logger.info("PDF avg %.1f chars/page below threshold — falling back to OCR.", avg_chars)
            return self._extract_ocr_pdf(file_bytes)

        return raw_text

    def _extract_text_pdf(self, file_bytes: bytes) -> tuple[str, int]:
        doc = fitz.open(stream=file_bytes, filetype="pdf")
        pages = []

        for i, page in enumerate(doc):
            text = page.get_text(sort=True)
            if text and text.strip():
                pages.append(text)
            else:
                logger.debug("Page %d returned no text from PyMuPDF", i + 1)

        doc.close()
        return "\n\n".join(pages), len(pages)

    def _extract_ocr_pdf(self, file_bytes: bytes) -> str:
        if not _HAS_OCR:
            raise ExtractionError(
                "This PDF appears to be scanned but OCR is not available. "
                "Install tesseract-ocr and poppler-utils, then reinstall pdf2image and pytesseract."
            )

        images: list[Image.Image] = pdf2image.convert_from_bytes(file_bytes, dpi=200)
        pages = []

        for i, image in enumerate(images):
            text = pytesseract.image_to_string(image, lang="eng")
            if text.strip():
                pages.append(text)
            else:
                logger.debug("OCR page %d returned no text", i + 1)

        return "\n\n".join(pages)

    def _extract_image(self, file_bytes: bytes) -> str:
        if not _HAS_OCR:
            raise ExtractionError("OCR is not available. Install tesseract-ocr, then reinstall pytesseract.")
        return pytesseract.image_to_string(Image.open(io.BytesIO(file_bytes)), lang="eng")

    def _extract_txt(self, file_bytes: bytes) -> str:
        # errors="replace" handles stray non-UTF-8 bytes without raising
        return file_bytes.decode("utf-8", errors="replace")

    def _clean_text(self, text: str) -> str:
        import re
        text = text.replace("\f", "\n")
        text = re.sub(r"\n{3,}", "\n\n", text)
        return text.strip()
