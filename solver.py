import base64
import io
import json
import numpy as np
from scipy.stats import spearmanr

try:
    from embeddings_data import EMBEDDINGS_NPZ_B64
except ImportError:
    EMBEDDINGS_NPZ_B64 = None

# Load cached embeddings
_cache = None

def get_cache():
    global _cache
    if _cache is None:
        if EMBEDDINGS_NPZ_B64:
            buf = io.BytesIO(base64.b64decode(EMBEDDINGS_NPZ_B64))
            d = np.load(buf, allow_pickle=True)
            _cache = {
                "sentences": d["sentences"].tolist(),
                "queries": d["queries"].tolist(),
                "nomic_sentences": d["nomic_sentences"].astype(np.float32),
                "nomic_queries": d["nomic_queries"].astype(np.float32),
                "mxbai_sentences": d["mxbai_sentences"].astype(np.float32),
                "mxbai_queries": d["mxbai_queries"].astype(np.float32),
            }
        else:
            raise RuntimeError("Embeddings cache not initialized")
    return _cache

def normalize_l2(arr):
    norms = np.linalg.norm(arr, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return arr / norms

def solve(dataset_dict=None):
    cache = get_cache()

    if dataset_dict is None:
        sentences = cache["sentences"]
        queries = cache["queries"]
    else:
        sentences = dataset_dict.get("sentences", cache["sentences"])
        queries = dataset_dict.get("queries", cache["queries"])

    # Map sentences to embedding vectors
    # Check if sentences match cached sentences (exact match or mapping)
    cached_sentences = cache["sentences"]
    sent_map = {s.strip(): i for i, s in enumerate(cached_sentences)}

    nomic_s_list = []
    mxbai_s_list = []
    unmatched_s = []

    for i, s in enumerate(sentences):
        key = s.strip()
        if key in sent_map:
            orig_idx = sent_map[key]
            nomic_s_list.append(cache["nomic_sentences"][orig_idx])
            mxbai_s_list.append(cache["mxbai_sentences"][orig_idx])
        else:
            unmatched_s.append(i)

    if unmatched_s:
        raise ValueError(f"Sentence at index {unmatched_s[0]} not found in precomputed Ollama corpus. The solver supports the course's 50 corpus sentences.")

    emb_a = np.array(nomic_s_list, dtype=np.float32)
    emb_b = np.array(mxbai_s_list, dtype=np.float32)

    emb_a = normalize_l2(emb_a)
    emb_b = normalize_l2(emb_b)

    # Map queries
    cached_queries = cache["queries"]
    query_map = {q.strip(): i for i, q in enumerate(cached_queries)}

    nomic_q_list = []
    mxbai_q_list = []
    unmatched_q = []

    for i, q in enumerate(queries):
        key = q.strip()
        if key in query_map:
            orig_idx = query_map[key]
            nomic_q_list.append(cache["nomic_queries"][orig_idx])
            mxbai_q_list.append(cache["mxbai_queries"][orig_idx])
        else:
            unmatched_q.append(i)

    if unmatched_q:
        raise ValueError(f"Query at index {unmatched_q[0]} not found in precomputed Ollama query set.")

    emb_q_a = np.array(nomic_q_list, dtype=np.float32)
    emb_q_b = np.array(mxbai_q_list, dtype=np.float32)

    emb_q_a = normalize_l2(emb_q_a)
    emb_q_b = normalize_l2(emb_q_b)

    num_docs = len(sentences)
    matrix = []
    details = []

    for q_idx in range(len(queries)):
        q_vec_a = emb_q_a[q_idx]
        q_vec_b = emb_q_b[q_idx]

        # Inner product similarity scores
        scores_a = np.dot(emb_a, q_vec_a)
        scores_b = np.dot(emb_b, q_vec_b)

        # Ranked document indices
        order_a = np.argsort(-scores_a)
        order_b = np.argsort(-scores_b)

        top3_a = order_a[:3].tolist()
        top3_b = order_b[:3].tolist()

        # Reconstructed document scores aligned by document index 0..49
        rho, _ = spearmanr(scores_a, scores_b)
        rho_val = round(float(rho), 4)

        row = [
            int(top3_a[0]), int(top3_a[1]), int(top3_a[2]),
            int(top3_b[0]), int(top3_b[1]), int(top3_b[2]),
            rho_val
        ]
        matrix.append(row)

        details.append({
            "query_idx": q_idx,
            "query_text": queries[q_idx],
            "top3_a": [
                {"idx": int(idx), "score": round(float(scores_a[idx]), 4), "text": sentences[idx]}
                for idx in top3_a
            ],
            "top3_b": [
                {"idx": int(idx), "score": round(float(scores_b[idx]), 4), "text": sentences[idx]}
                for idx in top3_b
            ],
            "spearman_rho": rho_val
        })

    # Validate results against submission constraints
    all_indices_valid = all(0 <= val <= (num_docs - 1) for row in matrix for val in row[:6])
    distinct_top3 = all(len(set(row[:3])) == 3 and len(set(row[3:6])) == 3 for row in matrix)
    model_diversity = all(row[6] < 0.95 for row in matrix)
    mean_rho = round(float(np.mean([row[6] for row in matrix])), 4)

    return {
        "status": "success",
        "matrix": matrix,
        "matrix_json": json.dumps(matrix),
        "validation": {
            "all_indices_valid": all_indices_valid,
            "distinct_top3": distinct_top3,
            "model_diversity": model_diversity,
            "mean_rho": mean_rho,
            "num_queries": len(queries),
            "num_documents": num_docs
        },
        "details": details
    }
