import numpy as np
from sklearn.decomposition import PCA
import umap
import os

current_dir = os.path.dirname(os.path.abspath(__file__))
data_dir = os.path.join(current_dir, "data")
pca_dir = os.path.join(data_dir, "pca")
umap_dir = os.path.join(data_dir, "umap")
os.makedirs(pca_dir, exist_ok=True)
os.makedirs(umap_dir, exist_ok=True)

features_path = os.path.join(data_dir, "feat_dinov2_vits14_VocPascal.npy")
features = np.load(features_path)
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
    return os.path.join(pca_dir, f"features_pca_{dim}.npy")

def get_umap_path(dim):
    return os.path.join(umap_dir, f"features_umap_{dim}.npy")

for dim in dimensions:
    reducesave_pca(features, dim, get_pca_path(dim))
    reducesave_umap(features, dim, get_umap_path(dim))