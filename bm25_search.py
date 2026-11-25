# bm25_search.py

import re

from rank_bm25 import BM25Okapi


# 1. Turn text into lowercase tokens
def tokenize(text):
    # Keep repeated words so BM25 can count how often they appear.
    # Example: "S3 stores photos, photos!" becomes
    # ["s3", "stores", "photos", "photos"].
    return re.findall(r"\w+", text.lower())


# 2. Build a reusable BM25 index from the uploaded document chunks
def build_bm25_index(chunks):
    # Tokenize each chunk and preserve the original chunk order.
    tokenized_chunks = [tokenize(chunk) for chunk in chunks]

    # Create a BM25 object that prepares word counts, word rarity,
    # and chunk lengths. Reuse this object for each new question.
    return BM25Okapi(tokenized_chunks)


# 3. Search for matching chunks and return their positions
def rank_bm25(question, bm25_index, limit=10):
    # Apply the same tokenization to the user's question.
    query_tokens = tokenize(question)

    # Get one BM25 score per chunk. scores[3] belongs to chunks[3].
    scores = bm25_index.get_scores(query_tokens)

    # Save the positions of chunks that contain at least one query token.
    matching_indices = []
    for index, term_counts in enumerate(bm25_index.doc_freqs):
        # any() returns True as soon as it finds a query token
        # in this chunk's word-count dictionary.
        if any(token in term_counts for token in query_tokens):
            matching_indices.append(index)

    # For each matching position, key looks up its score.
    # If matching_indices is [1, 3, 4], it uses only
    # scores[1], scores[3], and scores[4] to decide the order.
    # reverse=True puts the highest score first.
    # [:limit] keeps up to `limit` chunk positions.
    return sorted(
        matching_indices,
        key=lambda index: scores[index],
        reverse=True,
    )[:limit]

