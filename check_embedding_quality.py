import json
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

def check_embedding_quality(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    embeddings = np.array([item['embedding'] for item in data])
    print(f"Loaded {len(embeddings)} vectors.")
    n_samples = min(len(embeddings), 1000)
    indices = np.random.choice(len(embeddings), n_samples, replace=False)
    sample_embs = embeddings[indices]
    sim_matrix = cosine_similarity(sample_embs)
    sim_values = sim_matrix[np.triu_indices_from(sim_matrix, k=1)]
    avg_sim = np.mean(sim_values)
    min_sim = np.min(sim_values)
    max_sim = np.max(sim_values)
    print(f"Average similarity: {avg_sim:.4f}")
    print(f"Minimum similarity: {min_sim:.4f}")
    print(f"Similarity measure: {max_sim:.4f}")

check_embedding_quality("transformer_vector_danmu.json")
