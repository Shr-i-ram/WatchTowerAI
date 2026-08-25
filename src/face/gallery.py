"""
Identity Gallery Module

Manages the identity gallery: loading, saving, and comparing face embeddings
against enrolled identities using cosine similarity.
"""

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class GalleryMatch:
    """Result of matching a face embedding against the gallery."""

    identity: str  # Identity name or "UNKNOWN"
    similarity: float  # Cosine similarity score (0.0 to 1.0)
    is_known: bool  # True if similarity >= threshold


class IdentityGallery:
    """
    Manages face embeddings for known identities.

    Supports loading/saving gallery data and comparing new face embeddings
    against enrolled identities using cosine similarity.
    """

    def __init__(
        self,
        similarity_threshold: float = 0.45,
        gallery_dir: str = "models/face_gallery",
    ):
        """
        Initialize the identity gallery.

        Args:
            similarity_threshold: Minimum cosine similarity to consider a match.
                                  Below this, the person is classified as UNKNOWN.
            gallery_dir: Directory containing embeddings.npy and identities.json.
        """
        self.similarity_threshold = similarity_threshold
        self.gallery_dir = Path(gallery_dir)
        self.embeddings: Optional[np.ndarray] = None  # (N, 512) matrix
        self.identities: List[str] = []  # List of identity names
        self._num_identities = 0  # Unique number of identities

    def load(self) -> bool:
        """
        Load the gallery from disk.

        Returns:
            True if gallery loaded successfully, False otherwise.
        """
        emb_path = self.gallery_dir / "embeddings.npy"
        id_path = self.gallery_dir / "identities.json"

        if not emb_path.exists() or not id_path.exists():
            logger.warning(f"Gallery not found at {self.gallery_dir}. Run scripts/build_gallery.py first.")
            return False

        self.embeddings = np.load(str(emb_path))
        with open(id_path, "r") as f:
            data = json.load(f)
            self.identities = data["identities"]
            self._num_identities = data.get("num_identities", len(set(self.identities)))

        logger.info(
            f"Gallery loaded: {len(self.embeddings)} embeddings, "
            f"{self._num_identities} identities from {self.gallery_dir}"
        )
        return True

    def save(self) -> None:
        """Save the gallery to disk."""
        self.gallery_dir.mkdir(parents=True, exist_ok=True)

        emb_path = self.gallery_dir / "embeddings.npy"
        id_path = self.gallery_dir / "identities.json"

        np.save(str(emb_path), self.embeddings)
        with open(id_path, "w") as f:
            json.dump(
                {
                    "identities": self.identities,
                    "num_identities": self._num_identities,
                    "similarity_threshold": self.similarity_threshold,
                },
                f,
                indent=2,
            )

        logger.info(
            f"Gallery saved: {len(self.embeddings)} embeddings, "
            f"{self._num_identities} identities to {self.gallery_dir}"
        )

    def add_identity(self, name: str, embedding: np.ndarray) -> None:
        """
        Add a single embedding for an identity.

        Args:
            name: Identity name.
            embedding: (512,) L2-normalized ArcFace embedding.
        """
        if self.embeddings is None:
            self.embeddings = embedding.reshape(1, -1)
        else:
            self.embeddings = np.vstack([self.embeddings, embedding.reshape(1, -1)])
        self.identities.append(name)

    def build_from_embeddings(self, embeddings: np.ndarray, identities: List[str]) -> None:
        """
        Build the gallery from pre-computed embeddings.

        Args:
            embeddings: (N, 512) L2-normalized embeddings matrix.
            identities: List of N identity names (one per embedding).
        """
        self.embeddings = embeddings.astype(np.float32)
        self.identities = identities
        self._num_identities = len(set(identities))

    def match(self, query_embedding: np.ndarray) -> GalleryMatch:
        """
        Match a single query embedding against the gallery.

        Args:
            query_embedding: (512,) L2-normalized embedding.

        Returns:
            GalleryMatch with the best match.
        """
        if self.embeddings is None or len(self.embeddings) == 0:
            return GalleryMatch(identity="UNKNOWN", similarity=0.0, is_known=False)

        # Ensure query is normalized and 2D
        query = query_embedding.reshape(1, -1).astype(np.float32)
        query_norm = query / (np.linalg.norm(query, axis=1, keepdims=True) + 1e-8)

        # Normalize gallery embeddings (should already be, but ensure)
        gallery_norm = self.embeddings / (
            np.linalg.norm(self.embeddings, axis=1, keepdims=True) + 1e-8
        )

        # Cosine similarity = dot product of normalized vectors
        similarities = gallery_norm @ query_norm.T  # (N, 1)
        similarities = similarities.flatten()  # (N,)

        best_idx = int(np.argmax(similarities))
        best_sim = float(similarities[best_idx])
        best_identity = self.identities[best_idx]

        is_known = best_sim >= self.similarity_threshold

        return GalleryMatch(
            identity=best_identity if is_known else "UNKNOWN",
            similarity=best_sim,
            is_known=is_known,
        )

    def match_batch(self, query_embeddings: np.ndarray) -> List[GalleryMatch]:
        """
        Match multiple query embeddings against the gallery.

        Args:
            query_embeddings: (M, 512) L2-normalized embeddings.

        Returns:
            List of GalleryMatch objects.
        """
        if self.embeddings is None or len(self.embeddings) == 0:
            return [
                GalleryMatch(identity="UNKNOWN", similarity=0.0, is_known=False)
                for _ in range(len(query_embeddings))
            ]

        # Normalize
        queries = query_embeddings.astype(np.float32)
        q_norm = queries / (np.linalg.norm(queries, axis=1, keepdims=True) + 1e-8)
        g_norm = self.embeddings / (
            np.linalg.norm(self.embeddings, axis=1, keepdims=True) + 1e-8
        )

        # Similarity matrix (M, N)
        sim_matrix = q_norm @ g_norm.T

        results = []
        for i in range(len(queries)):
            best_idx = int(np.argmax(sim_matrix[i]))
            best_sim = float(sim_matrix[i, best_idx])
            is_known = best_sim >= self.similarity_threshold
            results.append(
                GalleryMatch(
                    identity=self.identities[best_idx] if is_known else "UNKNOWN",
                    similarity=best_sim,
                    is_known=is_known,
                )
            )

        return results

    @property
    def num_identities(self) -> int:
        """Number of unique identities in the gallery."""
        return self._num_identities

    @property
    def num_embeddings(self) -> int:
        """Total number of embeddings in the gallery."""
        return len(self.embeddings) if self.embeddings is not None else 0
