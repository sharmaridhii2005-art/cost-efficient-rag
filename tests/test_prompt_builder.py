from app.generation.prompt_builder import build_prompt


def test_build_prompt_contains_question():
    contexts = [
        {
            "text": "RAG combines retrieval and generation.",
            "source": "sample.txt",
            "chunk_index": 0,
        }
    ]

    question = "What is RAG?"

    prompt = build_prompt(question, contexts)

    assert question in prompt


def test_build_prompt_contains_context():
    contexts = [
        {
            "text": "RAG combines retrieval and generation.",
            "source": "sample.txt",
            "chunk_index": 0,
        }
    ]

    prompt = build_prompt("What is RAG?", contexts)

    assert "RAG combines retrieval and generation." in prompt
    assert "sample.txt" in prompt


def test_build_prompt_has_grounding_instruction():
    contexts = [
        {
            "text": "RAG combines retrieval and generation.",
            "source": "sample.txt",
            "chunk_index": 0,
        }
    ]

    prompt = build_prompt("What is RAG?", contexts)

    assert "only the context" in prompt.lower()