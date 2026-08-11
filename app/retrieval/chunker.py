import re


def chunk_text(text, chunk_size=500, overlap=50):
    """
    Split text into chunks while trying to preserve sentence boundaries.
    """

    if not text:
        return []

    if overlap >= chunk_size:
        raise ValueError("overlap must be smaller than chunk_size")

    # Normalize whitespace
    text = re.sub(r"\s+", " ", text).strip()

    if not text:
        return []

    # Split into sentences
    sentences = re.split(r"(?<=[.!?])\s+", text)

    chunks = []
    current = ""

    for sentence in sentences:
        sentence = sentence.strip()

        if not sentence:
            continue

        # If adding the sentence stays within the limit
        candidate = f"{current} {sentence}".strip()

        if len(candidate) <= chunk_size:
            current = candidate
        else:
            if current:
                chunks.append(current)

            # Handle a single sentence longer than chunk_size
            if len(sentence) > chunk_size:
                start = 0

                while start < len(sentence):
                    end = start + chunk_size
                    chunks.append(sentence[start:end])

                    if end >= len(sentence):
                        break

                    start = end - overlap

                current = ""
            else:
                current = sentence

    if current:
        chunks.append(current)

    return chunks
