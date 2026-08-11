from app.generation.context_builder import build_context


def test_build_context_returns_text():
    results = {
        "documents": [["RAG combines retrieval and generation."]],
        "metadatas": [[
            {
                "source": "sample.txt",
                "chunk_index": 0,
            }
        ]],
    }

    context = build_context(results)

    assert isinstance(context, str)
    assert "RAG combines retrieval and generation." in context


def test_build_context_handles_empty_results():
    results = {
        "documents": [[]],
        "metadatas": [[]],
    }

    context = build_context(results)

    assert context == ""


def test_build_context_handles_missing_metadata():
    results = {
        "documents": [["Some document text."]],
        "metadatas": [[{}]],
    }

    context = build_context(results)

    assert "Some document text." in context