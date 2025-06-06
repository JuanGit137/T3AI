import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import rand_score, adjusted_rand_score
from sklearn.preprocessing import LabelEncoder
from config import FEATURE_FILE, VAL_LIST

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
    return ri, ari

def clustering():
    print("Cargando vectores")
    features = np.load(FEATURE_FILE)
    true_labels = load_labels(VAL_LIST)
    print("Aplicando K-means")
    for k in [20, 50]:
        ri, ari = evaluate_kmeans(features, true_labels, k)
        print(f"K={k}: RI = {ri:.4f}, ARI = {ari:.4f}")
