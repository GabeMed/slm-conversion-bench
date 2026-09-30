"""C4's `assign`: the cluster of a call, computed from its prompt only (SPEC S3; design §1.3).

`embed(texts, embedding)` is the one embedding function: J5 uses it (through the `embed` execution)
to put every call in the prompt-only space where it computes centroids, and the router uses it to
assign a new call. The `embedding` block of the `centroids` fact fully specifies it (model, pinned
revision, max_seq_length, which end is kept when a prompt is longer), so both sides compute the
same vector. It runs in the agent environment (sentence-transformers as pinned by
env/agent/requirements.lock), on CPU, and returns unit vectors.
"""
import math
import threading
from typing import Dict, List, Sequence

_models: Dict[tuple, object] = {}
_lock = threading.Lock()


def prompt_text(messages: Sequence[Dict[str, str]]) -> str:
    """The text a call is embedded by: its prompt messages, never its response."""
    return "\n\n".join(f"{m['role']}: {m['content']}" for m in messages)


def _model(embedding: Dict):
    trust = embedding.get("trust_remote_code", False)
    key = (embedding["model"], embedding["revision"], embedding["max_seq_length"], embedding["truncation"], trust)
    if key not in _models:
        from sentence_transformers import SentenceTransformer
        model = SentenceTransformer(embedding["model"], revision=embedding["revision"], device="cpu",
                                    trust_remote_code=trust)
        model.max_seq_length = embedding["max_seq_length"]
        model.tokenizer.truncation_side = "right" if embedding["truncation"] == "head" else "left"
        _models[key] = model
    return _models[key]


def embed(texts: Sequence[str], embedding: Dict) -> List[List[float]]:
    with _lock:
        vectors = _model(embedding).encode(list(texts), normalize_embeddings=True, convert_to_numpy=True)
    return vectors.tolist()


def nearest(vector: Sequence[float], clusters: Dict[str, Sequence[float]]) -> str:
    """The cluster of highest cosine similarity; ties go to the smallest id."""
    norm = math.sqrt(sum(x * x for x in vector))
    for cluster, centroid in clusters.items():
        if len(centroid) != len(vector):
            raise ValueError(f"centroid {cluster} has dimension {len(centroid)}, the vector {len(vector)}")

    def cosine(c: str) -> float:
        centroid = clusters[c]
        return sum(a * b for a, b in zip(vector, centroid)) / (norm * math.sqrt(sum(x * x for x in centroid)))
    return max(sorted(clusters), key=cosine)


def assign(messages: Sequence[Dict[str, str]], centroids: Dict) -> str:
    vector = embed([prompt_text(messages)], centroids["embedding"])[0]
    return nearest(vector, centroids["clusters"])
