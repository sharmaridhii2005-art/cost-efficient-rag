from app.retrieval.chunker import chunk_text


def test_chunk_text_returns_chunks():
    text = "This is a test document. " * 100

    chunks = chunk_text(text)

    assert len(chunks) > 0
    assert all(isinstance(chunk, str) for chunk in chunks)


def test_chunk_text_preserves_content():
    text = "Hello world. This is a RAG test document."

    chunks = chunk_text(text)

    combined = " ".join(chunks)

    assert "Hello world" in combined
    assert "RAG test document" in combined


def test_short_text_creates_one_chunk():
    text = "Short document."

    chunks = chunk_text(text)

    assert len(chunks) == 1
    assert chunks[0] == text


def test_empty_text():
    chunks = chunk_text("")

    assert isinstance(chunks, list)
