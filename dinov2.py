import torch
from torchvision import transforms, models
import numpy as np
from PIL import Image
import os

DATASET = "VocPascal"
MODEL = "dinov2_vits14"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

preprocess = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406], 
        std=[0.229, 0.224, 0.225]),
])

model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14').to(device)
current_dir = os.path.dirname(os.path.abspath(__file__))
image_dir = os.path.join(current_dir, "VocPascal", "JPEGImages")
sample_image_path = os.path.join(image_dir, "2007_000027.jpg")
sample_image = Image.open(sample_image_path).convert('RGB')
sample_image = preprocess(sample_image).unsqueeze(0).to(device)

with torch.no_grad():
    sample_features = model(sample_image)
    dim = sample_features.shape[1]
print(f'Feature dimension: {dim}')

file_list_path = os.path.join(current_dir, "VocPascal", "val_voc.txt")
with open(file_list_path, 'r') as f:
    files = [line.strip().split() for line in f]

with torch.no_grad():
    n_images = len(files)
    features = np.zeros((n_images, dim), dtype=np.float32)
    for i, file in enumerate(files):
        filename = os.path.join(image_dir, file[0] + '.jpg')
        try:
            image = Image.open(filename).convert('RGB')
            image = preprocess(image).unsqueeze(0).to(device)
            features[i,:] = model(image).cpu()[0,:]
            if i % 100 == 0:
                print(f'{i}/{n_images}')
        except Exception as e:
            print(f"Error processing {filename}: {e}")

    os.makedirs('data', exist_ok=True)
    feat_file = os.path.join('data', f'feat_{MODEL}_{DATASET}.npy')
    np.save(feat_file, features)
    print(f'Features saved to {feat_file}')