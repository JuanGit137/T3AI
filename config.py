import os 

DATASET = "VocPascal"
MODEL = "dinov2_vits14"

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
NPY_DIR = os.path.join(CURRENT_DIR, "data", "feat_dinov2_vits14_VocPascal.npy")
PCA_DIR = os.path.join(CURRENT_DIR, "data", "pca")
UMAP_DIR = os.path.join(CURRENT_DIR, "data", "umap")
IMAGE_DIR = os.path.join(CURRENT_DIR, DATASET, "JPEGImages")
JSON_DIR = os.path.join(CURRENT_DIR, "data", "clustering_results.json")

FEATURE_FILE = os.path.join(CURRENT_DIR, "data", f"feat_dinov2_vits14_VocPascal.npy")
VAL_LIST = os.path.join(CURRENT_DIR, DATASET, "val_voc.txt")