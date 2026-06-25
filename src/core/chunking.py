from langchain_text_splitters import RecursiveCharacterTextSplitter


def chunk(text: str, size: int, overlap: int) -> list[str]:
    """Split text into overlapping chunks using recursive character splitting."""
    splitter = RecursiveCharacterTextSplitter(chunk_size=size, chunk_overlap=overlap)
    return splitter.split_text(text)
