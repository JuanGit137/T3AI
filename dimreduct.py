import numpy as np
from sklearn.decomposition import PCA
import umap
import os
from config import PCA_DIR, UMAP_DIR, FEATURE_FILE

os.makedirs(PCA_DIR, exist_ok=True)
os.makedirs(UMAP_DIR, exist_ok=True)

features = np.load(FEATURE_FILE)
dimensions = [16, 32, 64, 128, 256]

def reducesave_pca(features, n_components, save_path):
    pca = PCA(n_components=n_components)
    features_pca = pca.fit_transform(features)
    np.save(save_path, features_pca)
    return features_pca

def reducesave_umap(features, n_components, save_path, random_state=42):
    reducer = umap.UMAP(n_components=n_components, random_state=random_state)
    features_umap = reducer.fit_transform(features)
    np.save(save_path, features_umap)
    return features_umap

def get_pca_path(dim):
    return os.path.join(PCA_DIR, f"features_pca_{dim}.npy")

def get_umap_path(dim):
    return os.path.join(UMAP_DIR, f"features_umap_{dim}.npy")

def reduce_dimensions():
    print("Reduciendo dimensiones...")
    for dim in dimensions:
        reducesave_pca(features, dim, get_pca_path(dim))
        reducesave_umap(features, dim, get_umap_path(dim))
    print("Reducciones guardadas correctamente")