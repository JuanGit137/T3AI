import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import rand_score, adjusted_rand_score

if __name__ == '__main__':
    print("Iniciando clustering...")

    # Cargar vectores de características
    features = np.load("data/feat_dinov2_vits14_VocPascal.npy")

    # Cargar etiquetas verdaderas
    label_file = "VocPascal/val_voc.txt"
    with open(label_file, 'r') as f:
        lines = f.readlines()
        labels = [line.strip().split()[1] for line in lines]

    # Convertir etiquetas a números con diccionario
    from sklearn.preprocessing import LabelEncoder
    le = LabelEncoder()
    true_labels = le.fit_transform(labels)

    # k = 20 (corresponde al número de clases del dataset)
    kmeans_20 = KMeans(n_clusters=20, random_state=42)
    pred_labels_20 = kmeans_20.fit_predict(features)

    # k = 50 (para ver cómo se comporta con más clusters)
    kmeans_50 = KMeans(n_clusters=50, random_state=42)
    pred_labels_50 = kmeans_50.fit_predict(features)

    # Evaluar k=20
    ri_20 = rand_score(true_labels, pred_labels_20)
    ari_20 = adjusted_rand_score(true_labels, pred_labels_20)

    # Evaluar k=50
    ri_50 = rand_score(true_labels, pred_labels_50)
    ari_50 = adjusted_rand_score(true_labels, pred_labels_50)

    print(f"K=20: RI = {ri_20:.4f}, ARI = {ari_20:.4f}")
    print(f"K=50: RI = {ri_50:.4f}, ARI = {ari_50:.4f}")
