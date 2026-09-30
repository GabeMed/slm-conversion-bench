"""C4's `assign`: the cluster of a call, computed from its prompt only (SPEC S3; design §1.3).

The same `embed(prompt_text(...))` produces the vectors J5 clusters and the vector `assign`
compares, so both live in one space. It runs in the agent environment (sentence-transformers as
pinned by env/agent/requirements.lock), on CPU, with normalized vectors; the model and revision
come from the `centroids` fact.
"""
import threading
from typing import Dict, List, Sequence

_models: Dict[tuple, object] = {}
_lock = threading.Lock()


def prompt_text(messages: Sequence[Dict[str, str]]) -> str:
    """The text a call is embedded by: its prompt messages, never its response."""
    return "\n\n".join(f"{m['role']}: {m['content']}" for m in messages)


def embed(texts: Sequence[str], model: str, revision: str) -> List[List[float]]:
    with _lock:
        key = (model, revision)
        if key not in _models:
            from sentence_transformers import SentenceTransformer
            _models[key] = SentenceTransformer(model, revision=revision, device="cpu")
        vectors = _models[key].encode(list(texts), normalize_embeddings=True, convert_to_numpy=True)
    return vectors.tolist()


def nearest(vector: Sequence[float], clusters: Dict[str, Sequence[float]]) -> str:
    """The cluster with the highest dot product (cosine, for unit vectors); ties go to the smallest id."""
    return max(sorted(clusters), key=lambda c: sum(a * b for a, b in zip(vector, clusters[c])))


def assign(messages: Sequence[Dict[str, str]], centroids: Dict) -> str:
    vector = embed([prompt_text(messages)], **centroids["embedding"])[0]
    return nearest(vector, centroids["clusters"])
