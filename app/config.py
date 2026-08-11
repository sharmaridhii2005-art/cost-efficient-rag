from pathlib import Path
import os
from dotenv import load_dotenv

# Project root
BASE_DIR = Path(__file__).resolve().parent.parent

# Load environment variables
load_dotenv(BASE_DIR / ".env")

# Main directories
DOCUMENTS_DIR = BASE_DIR / "documents"
EVALUATION_DIR = BASE_DIR / "evaluation"
REPORTS_DIR = BASE_DIR / "reports"

# ChromaDB
CHROMA_DIR = BASE_DIR / "chroma_db"

# OpenAI
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# RAG settings
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

CHUNK_SIZE = 500
CHUNK_OVERLAP = 50
TOP_K = 1

# Maximum ChromaDB distance accepted for retrieval.
# Larger distances indicate weaker semantic matches.
RETRIEVAL_DISTANCE_THRESHOLD = 1.2
# Number of document chunks retrieved for each query.
# Selected through empirical TOP-K benchmarking.
TOP_K = 2
