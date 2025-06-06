import numpy as np
from sklearn.decomposition import PCA
import umap
import os

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
