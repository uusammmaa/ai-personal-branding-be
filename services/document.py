# backend/services/document.py
from pypdf import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from typing import List
import io


def extract_text_from_pdf(file_bytes: bytes) -> str:
    """Extract all text from a PDF file."""
    reader = PdfReader(io.BytesIO(file_bytes))
    text = ""
    for page in reader.pages:
        text += page.extract_text() + "\n"
    return text


def chunk_text(text: str, chunk_size: int = 400, overlap: int = 50) -> List[str]:
    """Split text into overlapping chunks for embedding."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=overlap,
        separators=["\n\n", "\n", ".", " ", ""]
    )
    return splitter.split_text(text)
