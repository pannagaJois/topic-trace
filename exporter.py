"""
Export Module for TopicTrace.
Handles exporting study notes to Markdown (.md) and formatted PDF (.pdf).
"""

import os
import re
from typing import Optional
from fpdf import FPDF
from fpdf.enums import XPos, YPos


class StudyNotesPDF(FPDF):
    """
    Custom PDF class styled for academic study notes with header and footer.
    """
    def __init__(self, doc_title: str = "TopicTrace Study Notes"):
        super().__init__()
        self.doc_title = doc_title
        self._font_name = "helvetica"
        self._init_fonts()

    def _init_fonts(self):
        # Prefer DejaVu if available for full Unicode support
        dejavu_regular = "/usr/share/fonts/TTF/DejaVuSans.ttf"
        dejavu_bold = "/usr/share/fonts/TTF/DejaVuSans-Bold.ttf"
        if os.path.exists(dejavu_regular):
            try:
                self.add_font("DejaVu", "", dejavu_regular)
                if os.path.exists(dejavu_bold):
                    self.add_font("DejaVu", "B", dejavu_bold)
                self._font_name = "DejaVu"
            except Exception:
                self._font_name = "helvetica"

    def header(self):
        self.set_font(self._font_name, "B", 9)
        self.set_text_color(100, 116, 139)  # Slate gray
        self.cell(0, 7, self.doc_title, border=0, align="L")
        self.cell(0, 7, "Generated via TopicTrace", border=0, align="R", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
        self.set_draw_color(226, 232, 240)
        self.set_line_width(0.3)
        self.line(self.l_margin, self.get_y(), self.w - self.r_margin, self.get_y())
        self.ln(6)

    def footer(self):
        self.set_y(-15)
        self.set_font(self._font_name, "", 8)
        self.set_text_color(148, 163, 184)
        self.cell(0, 10, f"Page {self.page_no()}", border=0, align="C")


def sanitize_text_for_pdf(text: str, is_unicode_font: bool) -> str:
    """
    Sanitizes text by replacing problematic characters for non-unicode fonts,
    or normalizing spaces and unicode symbols.
    """
    if is_unicode_font:
        # Simple cleanup
        return text.replace("\r\n", "\n").replace("\r", "\n")

    # Fallback replacements for standard Latin-1 helvetica
    replacements = {
        "“": '"', "”": '"', "‘": "'", "’": "'",
        "—": "-", "–": "-", "…": "...", "•": "*",
        "→": "->", "←": "<-", "⚠️": "[WARNING]",
        "✔": "[x]", "✅": "[x]", "❌": "[!]"
    }
    for orig, rep in replacements.items():
        text = text.replace(orig, rep)
    # Strip non-latin1 characters
    return text.encode("latin-1", "replace").decode("latin-1")


def export_markdown(notes_text: str) -> bytes:
    """
    Returns study notes formatted as UTF-8 encoded markdown bytes.
    """
    return notes_text.strip().encode("utf-8")


def export_pdf(notes_text: str, title: str = "TopicTrace Study Notes") -> bytes:
    """
    Converts study notes markdown into a clean, styled PDF document.
    """
    pdf = StudyNotesPDF(doc_title=title)
    pdf.set_auto_page_break(auto=True, margin=18)
    pdf.set_margins(left=18, top=18, right=18)
    pdf.add_page()

    font_name = pdf._font_name
    is_unicode = (font_name == "DejaVu")

    # Document Main Title
    pdf.set_font(font_name, "B", 20)
    pdf.set_text_color(15, 23, 42)  # Dark slate
    pdf.cell(0, 12, sanitize_text_for_pdf(title, is_unicode), new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    pdf.ln(2)

    # Mandatory AI Warning Callout Banner
    warning_text = "AI-generated content. Please verify with your textbook."
    pdf.set_fill_color(254, 243, 199)  # Soft amber background
    pdf.set_draw_color(245, 158, 11)   # Amber border
    pdf.set_text_color(146, 64, 14)    # Dark amber text
    pdf.set_font(font_name, "B", 10)
    pdf.rect(pdf.l_margin, pdf.get_y(), pdf.w - pdf.l_margin - pdf.r_margin, 9, style="FD")
    pdf.set_xy(pdf.l_margin + 4, pdf.get_y() + 1.5)
    pdf.cell(0, 6, sanitize_text_for_pdf(f"[!] {warning_text}", is_unicode), new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    pdf.ln(8)

    # Parse notes markdown line by line
    lines = notes_text.splitlines()
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        if not line:
            pdf.ln(3)
            i += 1
            continue

        # Check line types
        if line.startswith("# "):
            header_text = line[2:].strip()
            pdf.ln(5)
            pdf.set_font(font_name, "B", 16)
            pdf.set_text_color(30, 41, 59)
            pdf.cell(0, 9, sanitize_text_for_pdf(header_text, is_unicode), new_x=XPos.LMARGIN, new_y=YPos.NEXT)
            pdf.ln(2)

        elif line.startswith("## "):
            header_text = line[3:].strip()
            pdf.ln(6)
            # Underline bar / colored topic header
            pdf.set_fill_color(241, 245, 249)
            pdf.rect(pdf.l_margin, pdf.get_y(), pdf.w - pdf.l_margin - pdf.r_margin, 8, style="F")
            pdf.set_font(font_name, "B", 13)
            pdf.set_text_color(15, 23, 42)
            pdf.set_x(pdf.l_margin + 2)
            pdf.cell(0, 8, sanitize_text_for_pdf(header_text, is_unicode), new_x=XPos.LMARGIN, new_y=YPos.NEXT)
            pdf.ln(3)

        elif line.startswith("### "):
            header_text = line[4:].strip()
            pdf.ln(3)
            pdf.set_font(font_name, "B", 11)
            pdf.set_text_color(51, 65, 85)
            pdf.cell(0, 7, sanitize_text_for_pdf(header_text, is_unicode), new_x=XPos.LMARGIN, new_y=YPos.NEXT)
            pdf.ln(1)

        elif line.startswith("> "):
            # Blockquote / Warning box
            quote_text = line[2:].strip()
            pdf.set_fill_color(254, 242, 242) if "Missing from Textbook" in quote_text else (248, 250, 252)
            pdf.set_draw_color(239, 68, 68) if "Missing from Textbook" in quote_text else (203, 213, 225)
            pdf.set_font(font_name, "B" if "Missing" in quote_text else "", 10)
            pdf.set_text_color(185, 28, 28) if "Missing" in quote_text else (71, 85, 105)
            
            clean_q = sanitize_text_for_pdf(quote_text, is_unicode)
            pdf.multi_cell(0, 6, clean_q, border=1, fill=True)
            pdf.ln(3)

        elif line.startswith("- ") or line.startswith("* ") or line.startswith("• "):
            bullet_text = line[2:].strip()
            pdf.set_font(font_name, "", 10)
            pdf.set_text_color(30, 41, 59)
            bullet_char = "• " if is_unicode else "* "
            
            # Remove markdown bold markers ** for clean PDF reading
            clean_bullet = re.sub(r"\*\*(.*?)\*\*", r"\1", bullet_text)
            pdf.set_x(pdf.l_margin + 4)
            pdf.multi_cell(pdf.w - pdf.l_margin - pdf.r_margin - 4, 6, sanitize_text_for_pdf(f"{bullet_char}{clean_bullet}", is_unicode))
            pdf.ln(1)

        else:
            # Regular paragraph
            clean_p = re.sub(r"\*\*(.*?)\*\*", r"\1", line)
            pdf.set_font(font_name, "", 10)
            pdf.set_text_color(51, 65, 85)
            pdf.multi_cell(0, 6, sanitize_text_for_pdf(clean_p, is_unicode))
            pdf.ln(2)

        i += 1

    return bytes(pdf.output())
