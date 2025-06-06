import os
import numpy as np

current_dir = os.path.dirname(os.path.abspath(__file__))
data_dir = os.path.join(current_dir, "data")
features_path = os.path.join(data_dir, "feat_dinov2_vits14_VocPascal.npy")
features = np.load(features_path)
dimensions = [16, 32, 64, 128, 256]

def get_pca_path(dim):
    return os.path.join(data_dir, f"features_pca_{dim}.npy")

def get_umap_path(dim):
    return os.path.join(data_dir, f"features_umap_{dim}.npy")