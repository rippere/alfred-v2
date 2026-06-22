"""BM25/TF-IDF sparse vector store for Milvus hybrid search.

Also supports offline BM25-only search (no Milvus/Ollama) for lightweight
deployments. Use fit_and_store() during rebuild to enable search().
"""
from __future__ import annotations

import pickle
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from sklearn.feature_extraction.text import TfidfVectorizer


class BM25Store:
    """Fits a TF-IDF model over the vault corpus and produces sparse vectors.

    Vectors are returned as {token_id: weight} dicts, which Milvus accepts
    as SPARSE_FLOAT_VECTOR values for hybrid search.

    When fit_and_store() is used instead of fit(), also stores the corpus
    matrix and chunk_ids to enable offline search() without Milvus.
    """

    def __init__(self, path: Path) -> None:
        self.path = path
        self._vec: TfidfVectorizer | None = None
        self._corpus_matrix = None   # scipy sparse (n_chunks × vocab) — only when stored
        self._chunk_ids: list[str] = []

    @property
    def is_fitted(self) -> bool:
        return self._vec is not None

    @property
    def has_corpus(self) -> bool:
        """True if corpus vectors are available for offline search()."""
        return self._corpus_matrix is not None and len(self._chunk_ids) > 0

    def fit(self, texts: list[str]) -> None:
        from sklearn.feature_extraction.text import TfidfVectorizer
        self._vec = TfidfVectorizer(
            max_features=30_000,
            sublinear_tf=True,   # log(1+tf) — matches BM25 behaviour more closely
            strip_accents="unicode",
            analyzer="word",
            token_pattern=r"(?u)\b\w+\b",
            min_df=1,
        )
        self._vec.fit(texts)

    def fit_and_store(self, texts: list[str], chunk_ids: list[str]) -> None:
        """Fit vectorizer AND store corpus matrix for offline search().

        Stores the transformed document matrix alongside the vectorizer.
        The saved pickle will be larger (~5–15 MB) but enables search()
        without Milvus — used for the lightweight second-machine deployment.
        """
        self.fit(texts)
        self._corpus_matrix = self._vec.transform(texts)
        self._chunk_ids = list(chunk_ids)

    def search(self, text: str, top_k: int = 8) -> list[tuple[str, float]]:
        """BM25 search against stored corpus. Returns [(chunk_id, score)] descending.

        Only works after fit_and_store() — raises RuntimeError otherwise.
        """
        if not self.has_corpus:
            raise RuntimeError(
                "Corpus not available — rebuild with scripts/_archive/phase4_rebuild_milvus.py to enable "
                "offline search, or use hybrid Milvus search instead."
            )
        query_vec = self._vec.transform([text])
        scores = (self._corpus_matrix @ query_vec.T).toarray().flatten()
        top_idx = scores.argsort()[::-1][:top_k]
        return [
            (self._chunk_ids[i], float(scores[i]))
            for i in top_idx
            if scores[i] > 0
        ]

    def encode(self, text: str) -> dict[int, float]:
        """Return sparse vector for a single text as {token_id: weight}."""
        if self._vec is None:
            raise RuntimeError("BM25Store not fitted — call fit() or load() first")
        mat = self._vec.transform([text])
        cx = mat[0].tocoo()
        return {int(j): float(v) for j, v in zip(cx.col, cx.data)}

    def encode_batch(self, texts: list[str]) -> list[dict[int, float]]:
        if self._vec is None:
            raise RuntimeError("BM25Store not fitted — call fit() or load() first")
        mat = self._vec.transform(texts)
        result = []
        for i in range(mat.shape[0]):
            cx = mat[i].tocoo()
            result.append({int(j): float(v) for j, v in zip(cx.col, cx.data)})
        return result

    def save(self) -> None:
        if self._vec is None:
            raise RuntimeError("Nothing to save — BM25Store not fitted")
        payload = {
            "vec": self._vec,
            "corpus": self._corpus_matrix,
            "chunk_ids": self._chunk_ids,
        }
        self.path.write_bytes(pickle.dumps(payload))

    def load(self) -> bool:
        if not self.path.exists():
            return False
        data = pickle.loads(self.path.read_bytes())
        # Handle old format (bare vectorizer) and new format (dict)
        if isinstance(data, dict):
            self._vec = data["vec"]
            self._corpus_matrix = data.get("corpus")
            self._chunk_ids = data.get("chunk_ids", [])
        else:
            self._vec = data   # legacy: bare TfidfVectorizer
        return True

    def vocab_size(self) -> int:
        if self._vec is None:
            return 0
        return len(self._vec.vocabulary_)
