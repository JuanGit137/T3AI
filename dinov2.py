import torch
from torchvision import transforms, models
import numpy as np
from PIL import Image
import os
from config import IMAGE_DIR, VAL_LIST, FEATURE_FILE, MODEL

def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    #print(f"Using device: {device}")

def get_preprocess():
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406], 
            std=[0.229, 0.224, 0.225]),
])

def load_model(device=None):
    if device is None:
        device = get_device()
    model = torch.hub.load('facebookresearch/dinov2', MODEL).to(device)
    model.eval()
    return model

def load_file_list():
    with open(VAL_LIST, 'r') as f:
        files = [line.strip().split()[0] for line in f]
    return [os.path.join(IMAGE_DIR, f"{filename}.jpg") for filename in files]

def extract_features():
    device = get_device()
    preprocess = get_preprocess()
    model = load_model(device)

    image_paths = load_file_list()
    n_images = len(image_paths)

    with torch.no_grad():
        sample_image = Image.open(image_paths).convert('RGB')
        sample_image = preprocess(sample_image).unsqueeze(0).to(device)
        sample_features = model(sample_image)
        dim = sample_features.shape[1]
    print(f'Feature dimension: {dim}')

    features = np.zeros((n_images, dim), dtype=np.float32)

    with torch.no_grad():
        for i, img_path in enumerate(image_paths):
            try:
                image = Image.open(img_path).convert('RGB')
                image = preprocess(image).unsqueeze(0).to(device)
                feat = model(image).cpu().numpy()[0]
                features[i, :] = feat
                if i % 100 == 0:
                    print(f"Processed {i}/{n_images}")
            except Exception as e:
                print(f"Error processing {img_path}: {e}")

    os.makedirs(os.path.dirname(FEATURE_FILE), exist_ok=True)
    np.save(FEATURE_FILE, features)
    print(f"Features saved to {FEATURE_FILE}")

    return features
