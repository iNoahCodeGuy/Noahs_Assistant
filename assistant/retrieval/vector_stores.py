from typing import List, Any
try:
    import faiss  # type: ignore
except Exception:  # fallback stub implementation
    class _StubIndex:
        def __init__(self, d):
            self.d = d
            self._vecs = []
        def add(self, arr):
            self._vecs.extend(arr.tolist())
        def search(self, arr, k):
            # naive linear dist
            import numpy as np
            q = arr[0]
            dists = []
            for i, v in enumerate(self._vecs):
                v = np.array(v)
                d = float(((q - v) ** 2).sum())
                dists.append((d, i))
            dists.sort(key=lambda x: x[0])
            sel = dists[:k]
            if not sel:
                import numpy as np
                return np.array([[0.0]*k]), np.array([[-1]*k])
            distances = [x[0] for x in sel]
            indices = [x[1] for x in sel]
            # pad
            while len(distances) < k:
                distances.append(0.0)
                indices.append(-1)
            import numpy as np
            return np.array([distances]), np.array([indices])
    class faiss:  # type: ignore
        IndexFlatL2 = _StubIndex
import numpy as np

class VectorStore:
    def __init__(self, dimension: int):
        self.dimension = dimension
        self.index = faiss.IndexFlatL2(dimension)
        self.vectors = []

    def add_vectors(self, vectors: List[np.ndarray], metadata: List[Any] = None):
        """Add vectors to the FAISS index."""
        self.index.add(np.array(vectors).astype('float32'))
        self.vectors.extend(metadata if metadata else [None] * len(vectors))

    def search(self, query_vector: np.ndarray, k: int = 5) -> List[Any]:
        """Search for the top k nearest vectors to the query vector."""
        distances, indices = self.index.search(np.array([query_vector]).astype('float32'), k)
        return [(self.vectors[idx], distances[0][i]) for i, idx in enumerate(indices[0]) if idx >=0 and idx < len(self.vectors)]

    def reset(self):
        """Reset the vector store."""
        self.index.reset()
        self.vectors = []
