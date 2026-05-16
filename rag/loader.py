from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter


def load_and_chunk_pdf(filepath: str) -> list:
    """
    Loads a PDF and splits it into overlapping chunks.

    Why overlap? If an answer spans two chunks, the overlap ensures
    context isn't lost at the boundary.

    Args:
        filepath: Path to the PDF file on disk.

    Returns:
        List of LangChain Document objects (chunks).
    """
    # Load the PDF — each page becomes a Document
    loader = PyPDFLoader(filepath)
    pages = loader.load()

    # Split pages into smaller chunks for embedding
    # chunk_size: max characters per chunk (~300-500 tokens)
    # chunk_overlap: characters shared between adjacent chunks
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=150,
        separators=["\n\n", "\n", " ", ""]  # tries to split at paragraphs first
    )

    chunks = splitter.split_documents(pages)
    print(f"[loader] Split {len(pages)} pages into {len(chunks)} chunks")
    return chunks