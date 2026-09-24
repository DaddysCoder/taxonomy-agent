"""
Local embedding + HNSW index.

Why HNSW over brute-force cosine search:
- For 10k files vs ~50 taxonomy nodes, brute force is fine
- But HNSW means adding 50k files later costs ~O(log n) per query
  instead of O(n), keeping classification instant at any scale.

Model choice: nomic-embed-text-v1.5 (recommended) or all-MiniLM-L6-v2
- nomic: 8192 token context, great for long documents, Apache 2.0 license
- MiniLM: faster, smaller, good for short text like filenames
"""

from __future__ import annotations
import json
import pickle
import numpy as np
from pathlib import Path
from typing import Optional

# These imports happen at runtime — installed locally by user
try:
    from sentence_transformers import SentenceTransformer
    import hnswlib
    HAS_ML = True
except ImportError:
    HAS_ML = False
    print("[WARNING] sentence-transformers or hnswlib not installed.")
    print("Run: pip install sentence-transformers hnswlib")


RECOMMENDED_MODEL = "nomic-ai/nomic-embed-text-v1.5"
FALLBACK_MODEL = "all-MiniLM-L6-v2"

# HNSW parameters
# ef_construction: higher = better index quality, slower build (200 is good)
# M: connections per node — 16 is standard, 32 for higher recall
HNSW_EF_CONSTRUCTION = 200
HNSW_M = 16
HNSW_EF_QUERY = 50   # higher = better recall at query time, still fast


class TaxonomyEmbedder:
    """
    Builds and queries an HNSW index over taxonomy node embeddings.
    
    Flow:
    1. embed_taxonomy()  — called once, embeds all taxonomy nodes
    2. classify_file()   — called per file, returns (node_id, confidence)
    """

    def __init__(
        self,
        model_name: str = RECOMMENDED_MODEL,
        index_path: Optional[Path] = None,
        dimension: int = 768,
    ):
        self.model_name = model_name
        self.index_path = index_path or Path.home() / ".taxonomy_agent" / "hnsw_index"
        self.index_path.mkdir(parents=True, exist_ok=True)

        self.dimension = dimension
        self._model: Optional[SentenceTransformer] = None
        self._index: Optional[hnswlib.Index] = None
        self._id_to_node: dict[int, str] = {}   # HNSW int id → taxonomy node_id
        self._node_to_id: dict[str, int] = {}   # taxonomy node_id → HNSW int id

    @property
    def model(self) -> SentenceTransformer:
        if self._model is None:
            if not HAS_ML:
                raise RuntimeError("sentence-transformers not installed")
            print(f"[Embedder] Loading model: {self.model_name}")
            try:
                self._model = SentenceTransformer(self.model_name, trust_remote_code=True)
            except Exception:
                print(f"[Embedder] Falling back to {FALLBACK_MODEL}")
                self._model = SentenceTransformer(FALLBACK_MODEL)
            # Infer dimension from model
            test = self._model.encode(["test"], convert_to_numpy=True)
            self.dimension = test.shape[1]
        return self._model

    def embed(self, texts: list[str]) -> np.ndarray:
        """Embed a list of strings. Returns (N, D) float32 array."""
        return self.model.encode(
            texts,
            convert_to_numpy=True,
            normalize_embeddings=True,   # cosine similarity via dot product
            show_progress_bar=len(texts) > 20,
            batch_size=32,
        ).astype(np.float32)

    def build_taxonomy_index(self, nodes: list) -> None:
        """
        Embed all taxonomy nodes and build HNSW index.
        Called once when taxonomy changes. Fast: ~50 nodes takes <1 second.
        """
        if not HAS_ML:
            raise RuntimeError("ML dependencies not installed")

        texts = [node.embed_text() for node in nodes]
        embeddings = self.embed(texts)

        # Update dimension now we know it
        self.dimension = embeddings.shape[1]

        # Build HNSW index
        # Space 'ip' = inner product (equivalent to cosine when vectors are normalised)
        self._index = hnswlib.Index(space="ip", dim=self.dimension)
        self._index.init_index(
            max_elements=max(len(nodes) * 2, 1000),  # room to grow
            ef_construction=HNSW_EF_CONSTRUCTION,
            M=HNSW_M,
        )
        self._index.set_ef(HNSW_EF_QUERY)

        # Map integer IDs (HNSW requirement) to node IDs
        self._id_to_node = {i: node.id for i, node in enumerate(nodes)}
        self._node_to_id = {node.id: i for i, node in enumerate(nodes)}

        self._index.add_items(embeddings, list(range(len(nodes))))
        self._save_index()
        print(f"[Embedder] Built HNSW index over {len(nodes)} taxonomy nodes")

    def find_nearest(
        self,
        text: str,
        k: int = 3,
        candidate_ids: Optional[list[str]] = None,
    ) -> list[tuple[str, float]]:
        """
        Find k nearest taxonomy nodes for a piece of text.
        Returns list of (node_id, similarity_score) sorted best-first.
        
        candidate_ids: if provided, restricts search to these nodes
        (used during hierarchical top-down classification — only search
        among children of the already-selected parent).
        """
        if self._index is None:
            self._load_index()

        query_embedding = self.embed([text])

        if candidate_ids:
            # Filter search to candidate set — H3Prompt top-down narrowing
            candidate_int_ids = [
                self._node_to_id[nid]
                for nid in candidate_ids
                if nid in self._node_to_id
            ]
            if not candidate_int_ids:
                return []
            # HNSW doesn't support filtered search natively in hnswlib,
            # so we embed candidates and do exact search over the small set.
            # This is fine: candidate set is tiny (5-10 nodes).
            candidate_embeddings = np.array([
                self._index.get_items([i])[0] for i in candidate_int_ids
            ], dtype=np.float32)
            sims = (candidate_embeddings @ query_embedding.T).squeeze()
            if sims.ndim == 0:
                sims = np.array([float(sims)])
            top_k_local = np.argsort(sims)[::-1][:k]
            return [
                (candidate_ids[i], float(sims[i]))
                for i in top_k_local
            ]
        else:
            # Full HNSW search — O(log n), very fast
            labels, distances = self._index.knn_query(query_embedding, k=min(k, self._index.element_count))
            results = []
            for label, dist in zip(labels[0], distances[0]):
                node_id = self._id_to_node.get(int(label))
                if node_id:
                    # dist is inner product (higher = more similar for normalised vectors)
                    results.append((node_id, float(dist)))
            return sorted(results, key=lambda x: x[1], reverse=True)

    def _save_index(self):
        self._index.save_index(str(self.index_path / "hnsw.bin"))
        with open(self.index_path / "id_map.pkl", "wb") as f:
            pickle.dump((self._id_to_node, self._node_to_id, self.dimension), f)

    def _load_index(self):
        map_path = self.index_path / "id_map.pkl"
        index_path = self.index_path / "hnsw.bin"
        if not map_path.exists() or not index_path.exists():
            raise RuntimeError(
                "HNSW index not built yet. Run: taxonomy-agent build-index"
            )
        with open(map_path, "rb") as f:
            self._id_to_node, self._node_to_id, self.dimension = pickle.load(f)
        self._index = hnswlib.Index(space="ip", dim=self.dimension)
        self._index.load_index(str(index_path))
        self._index.set_ef(HNSW_EF_QUERY)
