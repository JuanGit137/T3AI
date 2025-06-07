import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import rand_score, adjusted_rand_score
from sklearn.preprocessing import LabelEncoder
import os
from config import VAL_LIST

def load_labels(val_list):
    with open(val_list, 'r') as f:
        lines = f.readlines()
        labels = [line.strip().split()[1] for line in lines]
    le = LabelEncoder()
    return le.fit_transform(labels)

def evaluate_kmeans(features, true_labels, k):
    kmeans = KMeans(n_clusters=k, random_state=42)
    pred_labels = kmeans.fit_predict(features)
    ri = rand_score(true_labels, pred_labels)
    ari = adjusted_rand_score(true_labels, pred_labels)
    return pred_labels, ri, ari

def run_clustering_on_file(
    feature_file,
    name="Espacio",
    cluster_dir=None,
    dim=None,
    k_values=[20, 50],
    results=None
):
    print(f"\nCargando vectores: {feature_file}")
    features = np.load(feature_file)
    true_labels = load_labels(VAL_LIST)
    print(f"Aplicando K-means en {name}")
    for k in k_values:
        pred_labels, ri, ari = evaluate_kmeans(features, true_labels, k)
        print(f"{name} | K={k}: RI = {ri:.4f}, ARI = {ari:.4f}")
        if results is not None:
            results.append({
                "name": name,
                "k": k,
                "RI": float(ri),
                "ARI": float(ari)
            })
        if cluster_dir is not None:
            os.makedirs(cluster_dir, exist_ok=True)
            filename = f"clusters_k{k}.npy" if dim is None else f"clusters_{dim}_k{k}.npy"
            cluster_path = os.path.join(cluster_dir, filename)
            np.save(cluster_path, pred_labels)
