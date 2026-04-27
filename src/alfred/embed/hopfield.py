"""Modern Hopfield Network query refinement (Ramsauer et al. 2020)."""
from __future__ import annotations

import numpy as np
import scipy.special


class HopfieldRefiner:
    def __init__(self, iterations: int = 3, beta: float = 2.0) -> None:
        self.iterations = iterations
        self.beta = beta

    def refine(self, query_vec: np.ndarray, candidate_embs: np.ndarray) -> np.ndarray:
        """Iteratively refine query_vec toward the attractor of stored patterns.

        query_vec: shape (D,) — normalized
        candidate_embs: shape (N, D) — normalized row vectors
        Returns refined vector of shape (D,), unit normalized.
        """
        x = query_vec.copy()
        for _ in range(self.iterations):
            scores = candidate_embs @ x          # (N,)
            weights = scipy.special.softmax(self.beta * scores)   # (N,)
            x = weights @ candidate_embs          # (D,)
            norm = np.linalg.norm(x)
            if norm > 1e-9:
                x = x / norm
        return x
