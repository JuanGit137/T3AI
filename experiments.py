from clustering import run_clustering_on_file
import json

def run_all_experiments():
    results = []

    run_clustering_on_file(
        "data/feat_dinov2_vits14_VocPascal.npy",
        name="DINOv2",
        cluster_dir="data/clusterdinov2",
        results=results
    )

    for dim in [16, 32, 64, 128, 256]:
        run_clustering_on_file(
            f"data/pca/features_pca_{dim}.npy",
            name=f"PCA{dim}",
            cluster_dir="data/clusterpca",
            dim=dim,
            results=results
        )

    for dim in [16, 32, 64, 128, 256]:
        run_clustering_on_file(
            f"data/umap/features_umap_{dim}.npy",
            name=f"UMAP{dim}",
            cluster_dir="data/clusterumap",
            dim=dim,
            results=results
        )

    with open("data/clustering_results.json", "w") as f:
        json.dump(results, f, indent=4)
