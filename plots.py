import json
import matplotlib.pyplot as plt
from config import JSON_DIR

def load_results(json_path="data/clustering_results.json"):
    with open(json_path, "r") as f:
        return json.load(f)

def get_metric(results, method, metric, k):
    return [
        r[metric]
        for r in results
        if r["name"].startswith(method) and r["k"] == k
    ]

def extract_dims(results, method_prefix):
    return sorted(set(
        int(r["name"].replace(method_prefix, ""))
        for r in results
        if r["name"].startswith(method_prefix)
    ))

def plot_metric(results, metric_name, ylabel, title):
    dims = extract_dims(results, "PCA") 
    ks = [20, 50]
    
    plt.figure(figsize=(8, 5))

    for k in ks:
        plt.plot(dims, get_metric(results, "PCA", metric_name, k),
                 marker='o', label=f'PCA k={k}')
        plt.plot(dims, get_metric(results, "UMAP", metric_name, k),
                 marker='s', label=f'UMAP k={k}')

    dinov2_vals = [r[metric_name] for r in results if r["name"] == "DINOv2"]
    for idx, val in enumerate(dinov2_vals):
        if idx < len(ks):
            plt.axhline(val, linestyle='--', color='gray' if idx == 0 else 'black',
                        label=f'DINOv2 k={ks[idx]}')

    plt.xlabel("Dimensión")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.show()

def generate_plots():
    results = load_results(JSON_DIR)
    plot_metric(results, "RI", "Rand Index (RI)", "Rand Index (RI) para PCA, UMAP y DINOv2")
    plot_metric(results, "ARI", "Adjusted Rand Index (ARI)", "Adjusted Rand Index (ARI) para PCA, UMAP y DINOv2")
