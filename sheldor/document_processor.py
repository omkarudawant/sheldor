import fitz
import logging
from typing import List, Dict, Any
from pathlib import Path
from datetime import datetime

logger = logging.getLogger(__name__)


class PDFProcessor:
    """Handles PDF document processing with Sheldon's attention to detail."""

    @staticmethod
    async def process_pdf(file_path: Path) -> List[Dict[str, Any]]:
        """
        Process a PDF file with meticulous attention to detail, as Sheldon would insist upon.
        Each document represents a page with its content and metadata.
        """
        try:
            documents = []
            pdf_document = fitz.open(file_path)

            for page_num in range(len(pdf_document)):
                page = pdf_document[page_num]
                text = page.get_text()

                if text.strip():  # Only include non-empty pages
                    # Sheldon would insist on precise metadata
                    documents.append(
                        {
                            "content": text,
                            "metadata": {
                                "source": str(file_path),
                                "page": page_num + 1,
                                "type": "pdf",
                                "word_count": len(text.split()),
                                "processing_timestamp": datetime.now().isoformat(),
                                "content_type": (
                                    "scientific"
                                    if any(
                                        word in text.lower()
                                        for word in [
                                            "physics",
                                            "quantum",
                                            "theory",
                                            "scientific",
                                        ]
                                    )
                                    else "general"
                                ),
                            },
                        }
                    )

            pdf_document.close()
            logger.info(
                f"Successfully processed PDF with Sheldon's level of precision: {file_path}"
            )
            return documents

        except Exception as e:
            logger.error(f"Error processing PDF {file_path}: {str(e)}")
            raise ValueError(f"Failed to process PDF with required precision: {str(e)}")
