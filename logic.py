import os
import numpy as np
from torchvision import transforms, models
import numpy as np
from PIL import Image
import os
import torch

current_dir = os.path.dirname(os.path.abspath(__file__))
data_dir = os.path.join(current_dir, "data")
features_path = os.path.join(data_dir, "feat_dinov2_vits14_VocPascal.npy")
features = np.load(features_path)
dimensions = [16, 32, 64, 128, 256]



DATASET = "VocPascal"
MODEL = "dinov2_vits14"
preprocess = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406], 
        std=[0.229, 0.224, 0.225]),
])
model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14').to(device)
image_dir = os.path.join(current_dir, "VocPascal", "JPEGImages")
sample_image_path = os.path.join(image_dir, "2007_000027.jpg")
sample_image = Image.open(sample_image_path).convert('RGB')
sample_image = preprocess(sample_image).unsqueeze(0).to(device)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def get_pca_path(dim):
    return os.path.join(data_dir, f"features_pca_{dim}.npy")

def get_umap_path(dim):
    return os.path.join(data_dir, f"features_umap_{dim}.npy")