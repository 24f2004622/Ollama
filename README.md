# Dual Embedding Alignment Solver

A high-performance web solver and REST API for evaluating semantic latent space divergence between `nomic-embed-text` and `mxbai-embed-large` embedding models. Deployed on Render.

## Features
- **Local Ollama Ground-Truth**: Exact precomputed Ollama embedding vectors for both models across the 50 corpus documents and 10 queries.
- **FAISS FlatIP / L2-Normalized Cosine Similarity**: Matches document rankings using exact inner product on unit vectors.
- **Spearman Rank Correlation ($\rho$)**: Computes full corpus rank agreement across all 50 documents for each query.
- **Instant 1-Click Submission**: Generates the exact 10&times;7 JSON matrix ready to paste directly into course submission portals.
- **Validation Engine**: Automatically verifies index range (`[0..49]`), distinct top-3 results, and model diversity (`\rho < 0.95`).

## REST API Usage

### Solve via cURL
```bash
# Windows PowerShell
curl.exe -X POST https://dual-embedding-solver.onrender.com/api/solve -H "Content-Type: application/json" -d @dataset.json

# Linux / macOS
curl -X POST https://dual-embedding-solver.onrender.com/api/solve -H "Content-Type: application/json" -d @dataset.json
```

### Solve Default Assignment Dataset
```bash
curl.exe -X POST https://dual-embedding-solver.onrender.com/api/solve
```

### Response Format
```json
{
  "status": "success",
  "matrix": [
    [28, 7, 29, 7, 28, 29, 0.7762],
    [10, 4, 24, 24, 4, 10, 0.6678],
    ...
  ],
  "matrix_json": "[[28, 7, 29, 7, 28, 29, 0.7762], ...]",
  "validation": {
    "all_indices_valid": true,
    "distinct_top3": true,
    "model_diversity": true,
    "mean_rho": 0.7029,
    "num_documents": 50,
    "num_queries": 10
  }
}
```

## Local Development
```bash
pip install -r requirements.txt
python app.py
```
Open `http://localhost:5000` in your browser.
