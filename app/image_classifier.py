import io

import torch
import torchvision.transforms as transforms
from PIL import Image

from app.cnn_model import SimpleCNN

CLASSES = ["airplane", "automobile", "bird", "cat", "deer",
           "dog", "frog", "horse", "ship", "truck"]

# Must be the same preprocessing used in training
transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor(),
])


class ImageClassifier:
    def __init__(self, weights_path="app/models/cnn_cifar10.pth"):
        self.model = SimpleCNN()
        self.model.load_state_dict(torch.load(weights_path, map_location="cpu"))
        self.model.eval()

    def predict(self, image_bytes):
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")  # handles PNG/JPG, drops alpha
        x = transform(image).unsqueeze(0)                           # add batch dim -> 1x3x64x64

        with torch.no_grad():
            probs = torch.softmax(self.model(x), dim=1)[0]

        idx = int(torch.argmax(probs))
        return {
            "predicted_class": CLASSES[idx],
            "confidence": round(float(probs[idx]), 4),
        }