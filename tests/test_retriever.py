from app.retrieval.retriever import Retriever


def test_retriever_returns_results():
    retriever = Retriever(top_k=1)

    results = retriever.retrieve("What is RAG?")

    assert "documents" in results
    assert len(results["documents"]) > 0
    assert len(results["documents"][0]) > 0


def test_retriever_returns_expected_source():
    retriever = Retriever(top_k=1)

    results = retriever.retrieve("What is RAG?")

    metadata = results["metadatas"][0][0]

    assert metadata["source"] == "sample.txt"


def test_retriever_respects_top_k():
    retriever = Retriever(top_k=1)

    results = retriever.retrieve("What is RAG?")

    assert len(results["documents"][0]) <= 1


def test_retriever_returns_distances():
    retriever = Retriever(top_k=1)

    results = retriever.retrieve("What is RAG?")

    assert "distances" in results
    assert len(results["distances"][0]) > 0