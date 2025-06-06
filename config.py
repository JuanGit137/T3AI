import os 

DATASET = "VocPascal"
MODEL = "dinov2_vits14"

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
NPY_DIR = os.path.join(CURRENT_DIR, "data", "feat_dinov2_vits14_VocPascal.npy")
IMAGE_DIR = os.path.join(CURRENT_DIR, DATASET, "JPEGImages")

FEATURE_FILE = os.path.join("data", f"feat_{MODEL}_{DATASET}.npy")
VAL_LIST = os.path.join(CURRENT_DIR, DATASET, "val_voc.txt")