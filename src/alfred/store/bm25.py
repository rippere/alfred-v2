"""BM25/TF-IDF sparse vector store for Milvus hybrid search."""
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
    """

    def __init__(self, path: Path) -> None:
        self.path = path
        self._vec: TfidfVectorizer | None = None

    @property
    def is_fitted(self) -> bool:
        return self._vec is not None

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
        self.path.write_bytes(pickle.dumps(self._vec))

    def load(self) -> bool:
        if not self.path.exists():
            return False
        self._vec = pickle.loads(self.path.read_bytes())
        return True

    def vocab_size(self) -> int:
        if self._vec is None:
            return 0
        return len(self._vec.vocabulary_)
