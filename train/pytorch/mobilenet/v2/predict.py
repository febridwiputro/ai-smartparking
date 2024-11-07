import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
import numpy as np
import os
import glob
import argparse
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def get_transforms(img_size=(80, 80)):
    return transforms.Compose([
        transforms.Resize(img_size),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

def load_model(model_path, num_classes, device):
    model = models.mobilenet_v2(pretrained=False)
    model.classifier[1] = nn.Linear(model.last_channel, num_classes)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model = model.to(device)
    model.eval()
    return model

def predict_single_image(image_path, model, device, img_size, class_names):
    transform = get_transforms(img_size)
    image = Image.open(image_path).convert('RGB')
    image = transform(image).unsqueeze(0).to(device)
    with torch.no_grad():
        outputs = model(image)
        _, preds = torch.max(outputs, 1)

    predicted_class = class_names[preds.item()]
    return predicted_class

def predict_from_folder(folder_path, model, device, img_size, class_names, output_folder):
    image_paths = glob.glob(os.path.join(folder_path, '*.*'))
    predictions = {}
    os.makedirs(output_folder, exist_ok=True)

    for image_path in image_paths:
        pred_class = predict_single_image(image_path, model, device, img_size, class_names)
        predictions[os.path.basename(image_path)] = pred_class
        logging.info(f"Image: {os.path.basename(image_path)}, Prediction: {pred_class}")

        class_output_folder = os.path.join(output_folder, pred_class)
        os.makedirs(class_output_folder, exist_ok=True)

        image = Image.open(image_path)
        output_image_path = os.path.join(class_output_folder, os.path.basename(image_path))
        image.save(output_image_path)
        
    return predictions

def predict_from_array(image_array, model, device, img_size, class_names):
    if isinstance(image_array, np.ndarray):
        transform = get_transforms(img_size)
        image = Image.fromarray(image_array).convert('RGB')
        image = transform(image).unsqueeze(0).to(device)
        with torch.no_grad():
            outputs = model(image)
            _, preds = torch.max(outputs, 1)
        
        predicted_class = class_names[preds.item()]
        return predicted_class
    else:
        raise ValueError("Input must be a valid NumPy array.")

if __name__ == "__main__":
    input_path = r"C:\Users\DOT\Documents\febri\github\ai-smartparking\output_chars_3"
    base_model_path = r"C:\Users\DOT\Documents\febri\github\ai-smartparking\train\pytorch\mobilenet\v2\models\new-model\20240922-19-05-37"
    model_path = os.path.join(base_model_path, "checkpoint.pth")
    class_file_path = os.path.join(base_model_path, "character_classes.npy")
    parser = argparse.ArgumentParser(description="Make predictions using a trained MobileNetV2 model")
    parser.add_argument('--model', type=str, default=model_path, help="Path to the trained model (e.g., 'model.pth')")
    parser.add_argument('--class_file', type=str, default=class_file_path, help="Path to the class names file (e.g., 'character_classes.npy')")
    parser.add_argument('--input_type', type=str, default="folder", choices=['folder', 'image', 'array'], help="Type of input: 'folder', 'image', or 'array'")
    parser.add_argument('--input_path', type=str, default=input_path, help="Path to the input image or folder")
    parser.add_argument('--output_folder', type=str, default="./output_folder", help="Path to the base output folder where results will be saved")
    parser.add_argument('--img_size', type=int, nargs=2, default=[80, 80], help="Input image size, default is [80, 80]")
    parser.add_argument('--device', type=str, default='cuda', help="Device to run the model on ('cuda' or 'cpu')")

    args = parser.parse_args()
    class_names = np.load(args.class_file)
    num_classes = len(class_names)
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    model = load_model(args.model, num_classes, device)

    os.makedirs(args.output_folder, exist_ok=True)

    if args.input_type == 'image':
        if not args.input_path:
            raise ValueError("For 'image' input type, you must provide the --input_path argument.")
        prediction = predict_single_image(args.input_path, model, device, args.img_size, class_names)
        logging.info(f"Prediction for {args.input_path}: {prediction}")
    elif args.input_type == 'folder':
        if not args.input_path:
            raise ValueError("For 'folder' input type, you must provide the --input_path argument.")

        predictions = predict_from_folder(args.input_path, model, device, args.img_size, class_names, args.output_folder)

        with open(os.path.join(args.output_folder, "predictions.txt"), "w") as f:
            for image_name, pred_class in predictions.items():
                f.write(f"{image_name}: {pred_class}\n")

    elif args.input_type == 'array':
        logging.info("Please provide the NumPy array input programmatically for 'array' input type.")