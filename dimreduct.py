import numpy as np
from sklearn.decomposition import PCA
import umap
import os
#from config import NPY_DIR, CURRENT_DIR

current_dir = os.path.dirname(os.path.abspath(__file__))
npydir = os.path.join(current_dir, "data", "feat_dinov2_vits14_VocPascal.npy")
features = np.load(npydir)

def reducesave_pca(features, n_components=2):
    pca = PCA(n_components=n_components)
    features_pca = pca.fit_transform(features)
    np.save(os.path.join(current_dir, "data", "features_pca.npy"), features_pca)
    return pca.fit_transform(features)

def reducesave_umap(features, n_components=2, random_state=42):
    reducer = umap.UMAP(n_components=n_components, random_state=random_state)
    features_umap = reducer.fit_transform(features)
    np.save(os.path.join(current_dir, "data", "features_umap.npy"), features_umap)
    return reducer.fit_transform(features)

reducesave_pca(features, n_components=2)
reducesave_umap(features, n_components=2)
