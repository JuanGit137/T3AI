import json
import matplotlib.pyplot as plt

with open("data/clustering_results.json", "r") as f:
    results = json.load(f)

# Organizar los datos
names = sorted(set(r["name"] for r in results if r["name"] != "DINOv2"))
dims = sorted(set(
    int(name[3:]) if name.startswith("PCA") else int(name[4:])
    for name in names if name.startswith("PCA") or name.startswith("UMAP")
))
ks = [20, 50]

# DINOv2
ri_dinov2 = [r["RI"] for r in results if r["name"] == "DINOv2"]
ari_dinov2 = [r["ARI"] for r in results if r["name"] == "DINOv2"]

def get_metric(method, metric, k):
    return [
        r[metric]
        for r in results
        if r["name"].startswith(method) and r["k"] == k
    ]

# Plot RI
plt.figure(figsize=(8, 5))
plt.plot(dims, get_metric("PCA", "RI", 20), marker='o', label='PCA k=20')
plt.plot(dims, get_metric("PCA", "RI", 50), marker='o', label='PCA k=50')
plt.plot(dims, get_metric("UMAP", "RI", 20), marker='s', label='UMAP k=20')
plt.plot(dims, get_metric("UMAP", "RI", 50), marker='s', label='UMAP k=50')
if len(ri_dinov2) > 0:
    plt.axhline(ri_dinov2[0], color='gray', linestyle='--', label='DINOv2 k=20')
if len(ri_dinov2) > 1:
    plt.axhline(ri_dinov2[1], color='black', linestyle='--', label='DINOv2 k=50')
plt.xlabel('Dimensión')
plt.ylabel('Rand Index (RI)')
plt.title('Rand Index (RI) para PCA, UMAP y DINOv2')
plt.legend()
plt.tight_layout()
plt.show()

# Plot ARI
plt.figure(figsize=(8, 5))
plt.plot(dims, get_metric("PCA", "ARI", 20), marker='o', label='PCA k=20')
plt.plot(dims, get_metric("PCA", "ARI", 50), marker='o', label='PCA k=50')
plt.plot(dims, get_metric("UMAP", "ARI", 20), marker='s', label='UMAP k=20')
plt.plot(dims, get_metric("UMAP", "ARI", 50), marker='s', label='UMAP k=50')
if len(ari_dinov2) > 0:
    plt.axhline(ari_dinov2[0], color='gray', linestyle='--', label='DINOv2 k=20')
if len(ari_dinov2) > 1:
    plt.axhline(ari_dinov2[1], color='black', linestyle='--', label='DINOv2 k=50')
plt.xlabel('Dimensión')
plt.ylabel('Adjusted Rand Index (ARI)')
plt.title('Adjusted Rand Index (ARI) para PCA, UMAP y DINOv2')
plt.legend()
plt.tight_layout()
plt.show()