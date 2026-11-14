"""Query-scoped dense vector retrieval isolating research sessions and preventing contamination."""

import logging
from typing import List, Dict, Any, Optional
import numpy as np
import faiss

from core.answer_generator import load_retrieval_pipeline

logger = logging.getLogger(__name__)


class QueryScopedRetriever:
    """Manages an isolated, in-memory FAISS index scoped strictly to the current research query."""

    def __init__(self, chunks: List[Dict[str, Any]]):
        self.chunks = chunks
        self.model, _, _ = load_retrieval_pipeline()
        self.index: Optional[faiss.IndexFlatIP] = None
        self._build_scoped_index()

    def _build_scoped_index(self):
        """Encode all query-specific chunks and index in a clean, isolated FAISS index."""
        if not self.chunks:
            return

        texts = [c["text"] for c in self.chunks]
        embeddings = self.model.encode(
            texts,
            batch_size=32,
            normalize_embeddings=True,
            show_progress_bar=False,
            convert_to_numpy=True,
        ).astype(np.float32)

        dimension = embeddings.shape[1]
        self.index = faiss.IndexFlatIP(dimension)
        self.index.add(embeddings)

    def search(
        self,
        query: str,
        top_k: int = 8,
        score_threshold: float = 0.15,
    ) -> List[Dict[str, Any]]:
        """Retrieve top relevant evidence chunks for the user's research query."""
        clean_q = query.strip()
        if not clean_q or self.index is None or self.index.ntotal == 0:
            return []

        query_vec = self.model.encode(
            [clean_q],
            normalize_embeddings=True,
            convert_to_numpy=True,
        ).astype(np.float32)

        k = min(top_k, self.index.ntotal)
        scores, indices = self.index.search(query_vec, k)

        results: List[Dict[str, Any]] = []
        for score, idx in zip(scores[0], indices[0]):
            if idx == -1:
                continue
            val_score = round(float(score), 4)
            if val_score < score_threshold:
                continue

            chunk_copy = dict(self.chunks[idx])
            chunk_copy["score"] = val_score
            results.append(chunk_copy)

        return results
