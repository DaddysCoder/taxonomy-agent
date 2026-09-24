"""
Extracts a text representation from any file for embedding.

Strategy:
- Text files: read directly (truncated)
- PDFs: extract first N pages of text
- Word docs: extract paragraph text
- Images: use filename + EXIF metadata
- Code: read as text (it IS text)
- Unknown: filename + extension only

We only need enough text to classify — not the full document.
Target: ~500 tokens (roughly 400 words). More than that
gives diminishing returns for classification and slows embedding.
"""

from __future__ import annotations
import re
from pathlib import Path
from typing import Optional

# Max chars to extract — ~500 tokens worth
MAX_CHARS = 2000
# Max chars for "snippet" used in auto-rule matching (much less)
SNIPPET_CHARS = 500


class FileReader:
    """Extracts text from files for embedding-based classification."""

    TEXT_EXTENSIONS = {
        ".txt", ".md", ".markdown", ".rst", ".csv",
        ".json", ".yaml", ".yml", ".toml", ".xml",
        ".html", ".htm", ".tex",
    }
    CODE_EXTENSIONS = {
        ".py", ".js", ".ts", ".jsx", ".tsx", ".java",
        ".c", ".cpp", ".h", ".cs", ".go", ".rs",
        ".rb", ".php", ".swift", ".kt", ".sh", ".bash",
        ".r", ".m", ".sql", ".ipynb",
    }

    def extract(self, file_path: Path) -> dict:
        """
        Returns:
            {
                "text": str,          # for embedding
                "snippet": str,       # for auto-rules (short)
                "filename": str,
                "extension": str,
                "size_bytes": int,
                "extraction_method": str,
            }
        """
        result = {
            "filename": file_path.name,
            "extension": file_path.suffix.lower(),
            "size_bytes": file_path.stat().st_size if file_path.exists() else 0,
            "text": "",
            "snippet": "",
            "extraction_method": "unknown",
        }

        ext = result["extension"]
        text = ""

        try:
            if ext in self.TEXT_EXTENSIONS or ext in self.CODE_EXTENSIONS:
                text = self._read_text_file(file_path)
                result["extraction_method"] = "text"

            elif ext == ".pdf":
                text = self._read_pdf(file_path)
                result["extraction_method"] = "pdf"

            elif ext in (".docx", ".doc"):
                text = self._read_docx(file_path)
                result["extraction_method"] = "docx"

            elif ext in (".jpg", ".jpeg", ".png", ".gif", ".webp", ".heic"):
                text = self._read_image_meta(file_path)
                result["extraction_method"] = "image_meta"

            else:
                # Fallback: just use filename — still useful for classification
                text = f"Filename: {file_path.name}\nPath: {file_path.parent}"
                result["extraction_method"] = "filename_only"

        except Exception as e:
            text = f"Filename: {file_path.name}\nExtraction error: {type(e).__name__}"
            result["extraction_method"] = "error_fallback"

        # Always prepend filename — it's often the strongest signal
        full_text = f"Filename: {file_path.name}\n\n{text}"
        result["text"] = self._clean(full_text)[:MAX_CHARS]
        result["snippet"] = result["text"][:SNIPPET_CHARS]

        return result

    def _read_text_file(self, path: Path) -> str:
        try:
            with open(path, encoding="utf-8", errors="ignore") as f:
                return f.read(MAX_CHARS * 2)  # read more, then truncate after clean
        except Exception:
            return ""

    def _read_pdf(self, path: Path) -> str:
        try:
            import PyPDF2
            text_parts = []
            with open(path, "rb") as f:
                reader = PyPDF2.PdfReader(f)
                for page in reader.pages[:5]:  # first 5 pages is enough
                    text_parts.append(page.extract_text() or "")
                    if sum(len(t) for t in text_parts) > MAX_CHARS * 2:
                        break
            return "\n".join(text_parts)
        except ImportError:
            return f"[PDF - install PyPDF2 for content extraction]"
        except Exception as e:
            return f"[PDF extraction failed: {e}]"

    def _read_docx(self, path: Path) -> str:
        try:
            from docx import Document
            doc = Document(path)
            paragraphs = [p.text for p in doc.paragraphs if p.text.strip()]
            return "\n".join(paragraphs)
        except ImportError:
            return "[DOCX - install python-docx for content extraction]"
        except Exception as e:
            return f"[DOCX extraction failed: {e}]"

    def _read_image_meta(self, path: Path) -> str:
        """For images, use filename + EXIF data (date, camera, GPS hints)."""
        parts = [f"Image file: {path.name}"]
        try:
            from PIL import Image
            from PIL.ExifTags import TAGS
            img = Image.open(path)
            exif_data = img._getexif() or {}
            for tag_id, value in exif_data.items():
                tag = TAGS.get(tag_id, str(tag_id))
                if tag in ("DateTime", "DateTimeOriginal", "Make", "Model", "Software"):
                    parts.append(f"{tag}: {value}")
        except Exception:
            pass
        return "\n".join(parts)

    def _clean(self, text: str) -> str:
        """Remove excessive whitespace and non-printable characters."""
        text = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x9f]", " ", text)
        text = re.sub(r"\s{3,}", "\n\n", text)
        return text.strip()
