from bm25_search import rank_bm25
from embed_notes import rank_semantic


def fuse_rankings(keyword_ranking, semantic_ranking, k=60):
    # Store one combined score per chunk position.
    fused_scores = {}

    for ranking in (keyword_ranking, semantic_ranking):
        for rank, index in enumerate(ranking, start=1):
            contribution = 1 / (k + rank)
            fused_scores[index] = fused_scores.get(index, 0.0) + contribution

    return sorted(
        fused_scores,
        key=lambda index: fused_scores[index],
        reverse=True,
    )


def rank_hybrid(question, bm25_index, chunk_vectors, limit=4):
    keyword_ranking = rank_bm25(question, bm25_index, limit=10)
    semantic_ranking = rank_semantic(question, chunk_vectors, limit=10)

    return fuse_rankings(keyword_ranking, semantic_ranking)[:limit]