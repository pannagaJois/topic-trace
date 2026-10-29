"""
PDF Extraction Module for TopicTrace.
Handles reading and extracting text from uploaded Syllabus and Textbook PDF files.
"""

from typing import Tuple, Dict, Any
import io
from pypdf import PdfReader


def extract_text_from_pdf(pdf_source) -> Tuple[str, Dict[str, Any]]:
    """
    Extracts text from a PDF file stream, buffer, or path.

    Args:
        pdf_source: A file-like object (e.g. Streamlit UploadedFile, BytesIO) or file path.

    Returns:
        A tuple of (extracted_text, metadata_dict).
        metadata_dict contains: page_count, char_count, word_count, is_empty_or_scanned.
    """
    if isinstance(pdf_source, bytes):
        pdf_source = io.BytesIO(pdf_source)
    elif hasattr(pdf_source, "read"):
        # Reset position if seekable
        if hasattr(pdf_source, "seek"):
            pdf_source.seek(0)

    reader = PdfReader(pdf_source)
    page_count = len(reader.pages)
    
    pages_text = []
    for idx, page in enumerate(reader.pages):
        page_text = page.extract_text() or ""
        # Strip excessive null characters or weird binary artifacts if any
        clean_page_text = page_text.replace("\x00", "").strip()
        if clean_page_text:
            pages_text.append(f"--- Page {idx + 1} ---\n{clean_page_text}")

    full_text = "\n\n".join(pages_text).strip()
    char_count = len(full_text)
    word_count = len(full_text.split()) if full_text else 0
    is_empty_or_scanned = (word_count < 15 and page_count > 0)

    metadata = {
        "page_count": page_count,
        "char_count": char_count,
        "word_count": word_count,
        "is_empty_or_scanned": is_empty_or_scanned
    }

    return full_text, metadata
