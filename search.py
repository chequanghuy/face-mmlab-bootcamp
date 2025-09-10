import cv2
import numpy as np
import torch
import os


def l2_normalize(x: np.ndarray, axis=-1, eps: float = 1e-10) -> np.ndarray:
    norm = np.linalg.norm(x, axis=axis, keepdims=True) + eps
    return x / norm

def cosine_distance_batch(query: np.ndarray, db: np.ndarray) -> np.ndarray:
    """
    query: (d,) hoặc (m,d)
    db: (N,d)
    return: (N,) hoặc (m,N) khoảng cách cosine
    """
    query = np.asarray(query, dtype=np.float32)
    db = np.asarray(db, dtype=np.float32)

    query = l2_normalize(query, axis=-1)
    db = l2_normalize(db, axis=-1)

    if query.ndim == 1:
        sims = db @ query        # (N,)
        dists = 1 - sims
    else:
        sims = query @ db.T      # (m,N)
        dists = 1 - sims
    return dists

# 0.5477386934673367
def predict_id(query_emb: np.ndarray, db: dict, threshold: float = 0.6 ): 
    """
    query_emb: (d,) numpy array
    db: dict {id: np.ndarray (n,d)}
    threshold: ngưỡng cosine distance (càng nhỏ càng giống)
    return: (predicted_id, best_distance)
    """
    best_id, best_dist = None, 1e9

    for pid, arr in db.items():
        if arr.ndim == 1:
            arr = arr[None, :]  # đảm bảo (n,d)
        dists = cosine_distance_batch(query_emb, arr)  # (n,)
        min_dist = float(np.min(dists))
        if min_dist < best_dist:
            best_dist = min_dist
            best_id = pid

    if best_dist <= threshold:
        return best_id, best_dist
    else:
        return "STRANGER", best_dist

def cosine_distance(a: torch.Tensor, b: torch.Tensor) -> float:
    """
    Tính cosine distance giữa 2 embedding vector
    a, b: torch.Tensor 1-D (shape [d])
    """
    a = a / a.norm(p=2)   # normalize L2
    b = b / b.norm(p=2)
    sim = torch.dot(a, b)  # cosine similarity
    return 1 - sim.item()  # distance = 1 - similarity