from pathlib import Path
from typing import List


def load_documents(documents_dir: Path) -> List[dict]:
    """
    Load text documents from the documents directory.
    """

    documents = []

    for file_path in documents_dir.glob("*"):
        if file_path.is_file():
            try:
                text = file_path.read_text(encoding="utf-8")

                if text.strip():
                    documents.append(
                        {
                            "source": file_path.name,
                            "text": text,
                        }
                    )

            except UnicodeDecodeError:
                print(f"Skipping unsupported file: {file_path.name}")

    return documents