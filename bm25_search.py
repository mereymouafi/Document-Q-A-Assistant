import re

from rank_bm25 import BM25Okapi


def tokenize(text):
    """Turn text into lowercase tokens."""
    return re.findall(r"\w+", text.lower())


def build_bm25_index(chunks):
    """Prepare BM25 statistics for the uploaded chunks."""
    tokenized_chunks = [tokenize(chunk) for chunk in chunks]

    if not any(tokenized_chunks):
        return None

    return BM25Okapi(tokenized_chunks)


def rank_bm25(question, bm25_index, limit=10):
    """Return matching chunk positions ordered by BM25 score."""
    query_tokens = tokenize(question)

    if bm25_index is None or not query_tokens:
        return []

    scores = bm25_index.get_scores(query_tokens)

    matching_indices = []
    for index, term_counts in enumerate(bm25_index.doc_freqs):
        # Keep the chunk if it contains at least one query token.
        if any(token in term_counts for token in query_tokens):
            matching_indices.append(index)

    # Look up each matching chunk's score to decide its position.
    return sorted(
        matching_indices,
        key=lambda index: scores[index],
        reverse=True,
    )[:limit]