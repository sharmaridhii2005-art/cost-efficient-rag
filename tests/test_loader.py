from app.config import DOCUMENTS_DIR
from app.ingestion.loader import load_documents


def test_load_documents():
    documents = load_documents(DOCUMENTS_DIR)

    assert len(documents) > 0
    assert documents[0]["text"]
    assert documents[0]["source"]