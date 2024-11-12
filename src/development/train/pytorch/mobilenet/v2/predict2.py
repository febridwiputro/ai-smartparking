import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
import os
import glob
import argparse
import numpy as np

# Load label encoder classes (optional, based on your earlier training process)
def load_classes(classes_path):
    with open(classes_path, 'rb') as f:
        classes = np.load(f)
    return classes

# Function to predict from a single image
def predict_image(image, model, transform, device):
    image = Image.open(image).convert('RGB')
    image = transform(image).unsqueeze(0)  # Add batch dimension
    image = image.to(device)

    model.eval()
    with torch.no_grad():
        output = model(image)
        _, pred = torch.max(output, 1)
    return pred.item()

# Function to predict from a folder of images
def predict_folder(folder, model, transform, device, classes):
    image_paths = glob.glob(os.path.join(folder, '*.*'))
    predictions = {}

    for image_path in image_paths:
        pred = predict_image(image_path, model, transform, device)
        class_name = classes[pred] if classes is not None else pred
        predictions[image_path] = class_name

    return predictions

# Function to predict from a NumPy array (image data)
def predict_from_array(array, model, transform, device):
    image = Image.fromarray(array).convert('RGB')
    image = transform(image).unsqueeze(0)  # Add batch dimension
    image = image.to(device)

    model.eval()
    with torch.no_grad():
        output = model(image)
        _, pred = torch.max(output, 1)

    return pred.item()

def load_model(model_path, num_classes, device):
    model = models.mobilenet_v2(pretrained=False)
    model.classifier[1] = nn.Linear(model.last_channel, num_classes)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model = model.to(device)
    return model

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Predict using a trained MobileNetV2 model')
    parser.add_argument('--input', type=str, required=True, help='Path to input folder, image, or array')
    parser.add_argument('--model_path', type=str, required=True, help='Path to the trained model .pth file')
    parser.add_argument('--classes_path', type=str, help='Path to the npy file containing class names')
    parser.add_argument('--mode', type=str, choices=['folder', 'image', 'array'], default='image', help='Prediction mode: folder, image, or array')
    
    args = parser.parse_args()

    # Load the model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if args.classes_path:
        classes = load_classes(args.classes_path)
    else:
        classes = None

    # Assume 36 classes as in your previous example, adjust as necessary
    model = load_model(args.model_path, num_classes=36, device=device)

    # Define the image transformation (same as in training)
    transform = transforms.Compose([
        transforms.Resize((80, 80)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    # Run prediction based on the mode
    if args.mode == 'folder':
        predictions = predict_folder(args.input, model, transform, device, classes)
        for image_path, class_name in predictions.items():
            print(f"Image: {image_path}, Predicted class: {class_name}")
    elif args.mode == 'image':
        pred = predict_image(args.input, model, transform, device)
        class_name = classes[pred] if classes is not None else pred
        print(f"Predicted class: {class_name}")
    elif args.mode == 'array':
        # For array input, let's assume a NumPy file is passed as input
        array = np.load(args.input)
        pred = predict_from_array(array, model, transform, device)
        class_name = classes[pred] if classes is not None else pred
        print(f"Predicted class: {class_name}")
