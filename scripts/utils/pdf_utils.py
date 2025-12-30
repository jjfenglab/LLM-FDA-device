import io
import re
import pymupdf
from PIL import Image
import pytesseract

def extract_text_from_pdf(pdf_filename: str) -> str:
    """Extracts text from a PDF file, including both embedded text and text found within images using OCR.
    
    Args:
        pdf_filename: local pdf file path
    
    Returns:
        str: Combined extracted text from all pages
    """
    doc = pymupdf.open(pdf_filename)
    pages = list(doc)
    all_text = []

    # Extract text page by page, applying OCR to problematic pages
    for i, page in enumerate(pages):
        page_text = _extract_text_from_page(page)
        
        if not page_text:
            # No text found, use OCR
            print(f"No direct text found on page {i + 1}, applying OCR...")
            image = _convert_page_to_image(page)
            ocr_text = _extract_text_with_ocr(image)
            all_text.append(ocr_text)
        elif not _is_clean_text_per_page(page_text):
            # Text found but garbled, use OCR
            print(f"Garbled text detected on page {i + 1}, applying OCR...")
            image = _convert_page_to_image(page)
            ocr_text = _extract_text_with_ocr(image)
            all_text.append(ocr_text)
        else:
            # Clean text, use as-is
            all_text.append(page_text)

    extracted_text = "\n".join(all_text)
    
    # Final check: If overall text still isn't clean, apply OCR to all pages
    if not _is_clean_text_overall(extracted_text):
        print("Overall text quality poor, applying OCR to all pages...")
        all_text = []
        for i, page in enumerate(pages):
            # print(f"OCR processing page {i + 1}...")
            image = _convert_page_to_image(page)
            ocr_text = _extract_text_with_ocr(image)
            all_text.append(ocr_text)
        extracted_text = "\n".join(all_text)
    
    # Fallback: If text contains weird encodings, force OCR on all pages
    elif _has_weird_encodings(extracted_text):
        print("Weird encodings detected, forcing OCR on all pages...")
        all_text = []
        for i, page in enumerate(pages):
            # print(f"OCR processing page {i + 1}...")
            image = _convert_page_to_image(page)
            ocr_text = _extract_text_with_ocr(image)
            all_text.append(ocr_text)
        extracted_text = "\n".join(all_text)
    
    return extracted_text

def _extract_text_from_page(page) -> str:
    """Extract text directly from a PDF page."""
    return page.get_text("text").strip()

def _convert_page_to_image(page) -> Image:
    """Convert a PDF page to a PIL Image."""
    pix = page.get_pixmap(matrix=pymupdf.Matrix(2, 2))
    image_bytes = pix.tobytes("png")
    return Image.open(io.BytesIO(image_bytes))

def _extract_text_with_ocr(image: Image) -> str:
    """Extract text from an image using OCR."""
    return pytesseract.image_to_string(image)

def _is_clean_text_per_page(text: str, min_word_count: int = 10, max_non_ascii_ratio: float = 0.4, max_garbled_ratio: float = 0.3) -> bool:
    """Check if extracted text from a single page is clean and usable."""
    if not text or len(text.strip()) < 20:
        return False
        
    # Check if text contains a reasonable number of words
    words = re.findall(r'\w+', text)
    if len(words) < min_word_count:
        return False

    # Check for excessive non-ASCII characters
    non_ascii_ratio = sum(ord(c) > 127 for c in text) / max(len(text), 1)
    if non_ascii_ratio > max_non_ascii_ratio:
        return False
    
    # Check for garbled text patterns (common in encoding issues)
    garbled_patterns = [
        r'[^\w\s\.,!?;:()\-"\']+',  # Too many special characters
        r'\b[a-zA-Z]{1,2}\b(?:\s+[a-zA-Z]{1,2}\b){5,}',  # Many short words in sequence
        r'[A-Z]{10,}',  # Long sequences of uppercase letters
        r'[^\x20-\x7E\n\r\t]{3,}',  # Sequences of non-printable ASCII
    ]
    
    garbled_chars = 0
    for pattern in garbled_patterns:
        matches = re.findall(pattern, text)
        garbled_chars += sum(len(match) for match in matches)
    
    garbled_ratio = garbled_chars / max(len(text), 1)
    if garbled_ratio > max_garbled_ratio:
        return False

    return True

def _is_clean_text_overall(text: str, min_word_count: int = 50, max_non_ascii_ratio: float = 0.3) -> bool:
    """Check if the overall extracted text from entire document is clean and usable."""
    if not text:
        return False
        
    # Check if text contains a reasonable number of words
    words = re.findall(r'\w+', text)
    if len(words) < min_word_count:
        return False

    # Check for excessive non-ASCII characters
    non_ascii_ratio = sum(ord(c) > 127 for c in text) / max(len(text), 1)
    if non_ascii_ratio > max_non_ascii_ratio:
        return False

    return True

def count_pdf_pages(pdf_path: str) -> int:
    """Count the number of pages in a PDF file.
    
    Args:
        pdf_path: Path to the PDF file
        
    Returns:
        int: Number of pages in the PDF, or -1 if error/corrupted file
    """
    try:
        doc = pymupdf.open(pdf_path)
        page_count = len(doc)
        doc.close()
        return page_count
    except Exception:
        return -1

def _has_weird_encodings(text: str, weird_encoding_threshold: float = 0.02) -> bool:
    """
    Check if text contains weird encodings that suggest corruption.
    
    Args:
        text: The text to check
        weird_encoding_threshold: Maximum ratio of weird patterns allowed
    
    Returns:
        bool: True if text contains significant weird encodings
    """
    if not text or len(text) < 100:
        return False
    
    # Look for specific encoding corruption patterns
    corruption_patterns = [
        # Isolated weird symbol sequences (common in encoding corruption)
        r'[*#$%&@^`|~]{2,}',
        # Parentheses with symbols/numbers (like "(*/0")
        r'\([*#$%&@^`|~\d/\\]+\)',
        # Symbol-number-symbol patterns
        r'[*#$%&@^`|~]\d+[*#$%&@^`|~]',
        # Multiple consecutive non-word characters on separate lines
        r'^\s*[^\w\s]{1,5}\s*$',
        # Sequences like @%A, *('$#
        r'[*#$%&@^`|~\'\"]{3,}',
    ]
    
    total_corruption_chars = 0
    corruption_instances = 0
    
    for pattern in corruption_patterns:
        matches = re.findall(pattern, text, re.MULTILINE)
        total_corruption_chars += sum(len(match) for match in matches)
        corruption_instances += len(matches)
    
    # Calculate ratio of corruption characters to total text
    corruption_ratio = total_corruption_chars / len(text)
    
    # Also check for very low alphanumeric ratio (another sign of corruption)
    alphanumeric_chars = sum(c.isalnum() for c in text)
    alphanumeric_ratio = alphanumeric_chars / len(text)
    
    # Check for isolated lines with only weird characters
    lines = text.split('\n')
    weird_lines = 0
    for line in lines:
        line_stripped = line.strip()
        if line_stripped and len(line_stripped) < 10:
            # Count non-alphanumeric characters in short lines
            non_alnum = sum(1 for c in line_stripped if not c.isalnum() and c not in ' \t.,!?;:()\-"\'')
            if non_alnum >= len(line_stripped) * 0.5:  # 50% or more weird chars
                weird_lines += 1
    
    # Text is considered to have weird encodings if:
    # 1. High ratio of corruption patterns, OR
    # 2. Very low alphanumeric ratio (< 50%), OR
    # 3. Multiple lines with predominantly weird characters
    return (corruption_ratio > weird_encoding_threshold or 
            alphanumeric_ratio < 0.5 or 
            weird_lines >= 5) 