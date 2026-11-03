"""Hybrid retrieval by composing existing lexical and vector retrievers."""

from app.fusion import reciprocal_rank_fusion, weighted_score_fusion


class HybridRetriever:
    def __init__(
        self,
        bm25_retriever,
        vector_retriever,
        *,
        method: str = "rrf",
        k: int = 60,
        alpha: float = 0.5,
        candidate_depth: int = 100,
    ) -> None:
        if candidate_depth <= 0:
            raise ValueError("candidate_depth must be greater than zero")
        if method not in {"rrf", "weighted"}:
            raise ValueError("method must be rrf or weighted")
        self.bm25_retriever = bm25_retriever
        self.vector_retriever = vector_retriever
        self.method = method
        self.k = k
        self.alpha = alpha
        self.candidate_depth = candidate_depth

    def search(self, query: str, k: int) -> list[tuple[str, float]]:
        if k <= 0:
            raise ValueError("k must be greater than zero")
        bm25 = self.bm25_retriever.search(query, self.candidate_depth)
        vector = self.vector_retriever.search(query, self.candidate_depth)
        if self.method == "rrf":
            fused = reciprocal_rank_fusion(bm25, vector, k=self.k)
        else:
            fused = weighted_score_fusion(bm25, vector, alpha=self.alpha)
        return fused[:k]
