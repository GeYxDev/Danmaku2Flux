import json
import numpy as np
import matplotlib.pyplot as plt

from sklearn.metrics.pairwise import cosine_similarity, pairwise_distances
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

def load_embeddings(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    embs = np.array([item["embedding"] for item in data], dtype=np.float32)
    titles = [item.get("title", "") for item in data]
    return embs, titles

def basic_similarity_check(embs):
    print("===== Basic Similarity Check =====")
    n_samples = min(len(embs), 1000)
    idx = np.random.choice(len(embs), n_samples, replace=False)
    sample = embs[idx]
    sim_matrix = cosine_similarity(sample)
    sims = sim_matrix[np.triu_indices_from(sim_matrix, k=1)]
    print(f"Average similarity: {sims.mean():.4f}")
    print(f"Minimum similarity: {sims.min():.4f}")
    print(f"Maximum similarity: {sims.max():.4f}")
    print()

def full_diagnostics(embs):
    print("===== Embedding Diagnostics =====")
    N, D = embs.shape
    print(f"N = {N}, D = {D}")
    norms = np.linalg.norm(embs, axis=1)
    print("Norm mean/std/min/max:",
          norms.mean(), norms.std(), norms.min(), norms.max())
    print("\n===== Pairwise Cosine Similarity =====")
    if N > 2000:
        idx = np.random.choice(N, 2000, replace=False)
        sample = embs[idx]
    else:
        sample = embs
    cos = cosine_similarity(sample)
    mask = ~np.eye(cos.shape[0], dtype=bool)
    pair_sims = cos[mask].reshape(cos.shape[0], -1).mean(axis=1)
    print("Mean similarity to others:", pair_sims.mean())
    print("Median similarity:", np.median(pair_sims))
    print("Global avg similarity:", cos[mask].mean())
    print("Min similarity:", cos[mask].min())
    print("Max similarity:", cos[mask].max())
    print("\n===== Nearest Neighbor Similarity =====")
    dists = pairwise_distances(embs, metric="cosine")
    np.fill_diagonal(dists, np.inf)
    nn_idx = np.argmin(dists, axis=1)
    nn_sim = 1 - dists[np.arange(N), nn_idx]
    print("NN similarity mean:", nn_sim.mean())
    print("NN similarity min :", nn_sim.min())
    print("NN similarity max :", nn_sim.max())
    print("\n===== PCA Variance Explained =====")
    pca = PCA(n_components=min(D, 50))
    pca.fit(embs)
    explained = pca.explained_variance_ratio_
    print("First 10 PCA components:", explained[:10].round(4))
    print("Cumulative (first 10):", explained[:10].sum().round(4))
    print("\n===== Running t-SNE (this may take a while) =====")
    tsne_count = min(2000, N)
    idx = np.random.choice(N, tsne_count, replace=False)
    tsne_embs = embs[idx]
    tsne = TSNE(n_components=2, perplexity=30, init="pca")
    vis = tsne.fit_transform(tsne_embs)
    plt.figure(figsize=(6, 5))
    plt.scatter(vis[:, 0], vis[:, 1], s=3)
    plt.title("t-SNE of Embeddings")
    plt.savefig("emb_tsne.png", dpi=200)
    print("Saved t-SNE visualization: emb_tsne.png")

def main():
    file_path = "transformer_vector_danmu.json"
    print(f"Loading: {file_path}")
    embs, titles = load_embeddings(file_path)
    print("Loaded", len(embs), "vectors.\n")
    basic_similarity_check(embs)
    full_diagnostics(embs)

if __name__ == "__main__":
    main()
