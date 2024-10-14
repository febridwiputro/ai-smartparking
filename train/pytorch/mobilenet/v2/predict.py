import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
import numpy as np
import os
import glob
import argparse
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Image transformations (must match training)
def get_transforms(img_size=(80, 80)):
    return transforms.Compose([
        transforms.Resize(img_size),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

# Load model architecture and weights
def load_model(model_path, num_classes, device):
    model = models.mobilenet_v2(pretrained=False)
    model.classifier[1] = nn.Linear(model.last_channel, num_classes)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model = model.to(device)
    model.eval()  # Set model to evaluation mode
    return model

# Predict on a single image
def predict_single_image(image_path, model, device, img_size, class_names):
    transform = get_transforms(img_size)
    image = Image.open(image_path).convert('RGB')
    image = transform(image).unsqueeze(0).to(device)  # Add batch dimension
    with torch.no_grad():
        outputs = model(image)
        _, preds = torch.max(outputs, 1)
    
    # Convert prediction index to class name
    predicted_class = class_names[preds.item()]
    return predicted_class

# Predict on all images in a folder
def predict_from_folder(folder_path, model, device, img_size, class_names):
    image_paths = glob.glob(os.path.join(folder_path, '*.*'))
    predictions = {}
    for image_path in image_paths:
        pred_class = predict_single_image(image_path, model, device, img_size, class_names)
        predictions[os.path.basename(image_path)] = pred_class
        logging.info(f"Image: {os.path.basename(image_path)}, Prediction: {pred_class}")
    return predictions

# Predict on a NumPy array
def predict_from_array(image_array, model, device, img_size, class_names):
    if isinstance(image_array, np.ndarray):
        transform = get_transforms(img_size)
        image = Image.fromarray(image_array).convert('RGB')
        image = transform(image).unsqueeze(0).to(device)
        with torch.no_grad():
            outputs = model(image)
            _, preds = torch.max(outputs, 1)
        
        # Convert prediction index to class name
        predicted_class = class_names[preds.item()]
        return predicted_class
    else:
        raise ValueError("Input must be a valid NumPy array.")

if __name__ == "__main__":
    input_path = r"D:\engine\smart_parking\train_model\dataset\bg-black-reduksi-20240930\Z"
    parser = argparse.ArgumentParser(description="Make predictions using a trained MobileNetV2 model")
    parser.add_argument('--model', type=str, default=r"D:\engine\smart_parking\train_model\model-training\pytorch\mobilenet\v2\models\new-model\20241013-14-53-06\checkpoint.pth", help="Path to the trained model (e.g., 'model.pth')")
    parser.add_argument('--class_file', type=str, default=r"D:\engine\smart_parking\train_model\model-training\pytorch\mobilenet\v2\models\new-model\20241013-14-53-06\character_classes.npy", help="Path to the class names file (e.g., 'character_classes.npy')")
    parser.add_argument('--input_type', type=str, required=True, choices=['folder', 'image', 'array'], help="Type of input: 'folder', 'image', or 'array'")
    parser.add_argument('--input_path', type=str, default=input_path, help="Path to the input image or folder")
    parser.add_argument('--img_size', type=int, nargs=2, default=[80, 80], help="Input image size, default is [80, 80]")
    parser.add_argument('--device', type=str, default='cuda', help="Device to run the model on ('cuda' or 'cpu')")

    args = parser.parse_args()

    # Load classes
    class_names = np.load(args.class_file)
    num_classes = len(class_names)

    # Set device
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')

    # Load model
    model = load_model(args.model, num_classes, device)

    # Perform prediction based on input type
    if args.input_type == 'image':
        if not args.input_path:
            raise ValueError("For 'image' input type, you must provide the --input_path argument.")
        prediction = predict_single_image(args.input_path, model, device, args.img_size, class_names)
        logging.info(f"Prediction for {args.input_path}: {prediction}")
    elif args.input_type == 'folder':
        if not args.input_path:
            raise ValueError("For 'folder' input type, you must provide the --input_path argument.")
        predictions = predict_from_folder(args.input_path, model, device, args.img_size, class_names)
        # You can process `predictions` here if needed
    elif args.input_type == 'array':
        logging.info("Please provide the NumPy array input programmatically for 'array' input type.")



# import torch
# import torch.nn as nn
# from torchvision import transforms, models
# from PIL import Image
# import numpy as np
# import os
# import glob
# import argparse
# import logging

# # Setup logging
# logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# # Image transformations (must match training)
# def get_transforms(img_size=(80, 80)):
#     return transforms.Compose([
#         transforms.Resize(img_size),
#         transforms.ToTensor(),
#         transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
#     ])

# # Load model architecture and weights
# def load_model(model_path, num_classes, device):
#     model = models.mobilenet_v2(pretrained=False)
#     model.classifier[1] = nn.Linear(model.last_channel, num_classes)
#     model.load_state_dict(torch.load(model_path, map_location=device))
#     model = model.to(device)
#     model.eval()  # Set model to evaluation mode
#     return model

# # Predict on a single image
# def predict_single_image(image_path, model, device, img_size):
#     transform = get_transforms(img_size)
#     image = Image.open(image_path).convert('RGB')
#     image = transform(image).unsqueeze(0).to(device)  # Add batch dimension
#     with torch.no_grad():
#         outputs = model(image)
#         _, preds = torch.max(outputs, 1)
#     return preds.item()

# # Predict on all images in a folder
# def predict_from_folder(folder_path, model, device, img_size):
#     image_paths = glob.glob(os.path.join(folder_path, '*.*'))
#     predictions = {}
#     for image_path in image_paths:
#         pred = predict_single_image(image_path, model, device, img_size)
#         predictions[os.path.basename(image_path)] = pred
#         logging.info(f"Image: {os.path.basename(image_path)}, Prediction: {pred}")
#     return predictions

# # Predict on a NumPy array
# def predict_from_array(image_array, model, device, img_size):
#     if isinstance(image_array, np.ndarray):
#         transform = get_transforms(img_size)
#         image = Image.fromarray(image_array).convert('RGB')
#         image = transform(image).unsqueeze(0).to(device)
#         with torch.no_grad():
#             outputs = model(image)
#             _, preds = torch.max(outputs, 1)
#         return preds.item()
#     else:
#         raise ValueError("Input must be a valid NumPy array.")

# if __name__ == "__main__":
#     parser = argparse.ArgumentParser(description="Make predictions using a trained MobileNetV2 model")
#     parser.add_argument('--model', type=str, required=True, help="Path to the trained model (e.g., 'model.pth')")
#     parser.add_argument('--class_file', type=str, required=True, help="Path to the class names file (e.g., 'character_classes.npy')")
#     parser.add_argument('--input_type', type=str, required=True, choices=['folder', 'image', 'array'], help="Type of input: 'folder', 'image', or 'array'")
#     parser.add_argument('--input_path', type=str, help="Path to the input image or folder")
#     parser.add_argument('--img_size', type=int, nargs=2, default=[80, 80], help="Input image size, default is [80, 80]")
#     parser.add_argument('--device', type=str, default='cuda', help="Device to run the model on ('cuda' or 'cpu')")

#     args = parser.parse_args()

#     # Load classes
#     class_names = np.load(args.class_file)
#     num_classes = len(class_names)

#     # Set device
#     device = torch.device(args.device if torch.cuda.is_available() else 'cpu')

#     # Load model
#     model = load_model(args.model, num_classes, device)

#     # Perform prediction based on input type
#     if args.input_type == 'image':
#         if not args.input_path:
#             raise ValueError("For 'image' input type, you must provide the --input_path argument.")
#         prediction = predict_single_image(args.input_path, model, device, args.img_size)
#         logging.info(f"Prediction for {args.input_path}: {class_names[prediction]}")
#     elif args.input_type == 'folder':
#         if not args.input_path:
#             raise ValueError("For 'folder' input type, you must provide the --input_path argument.")
#         predictions = predict_from_folder(args.input_path, model, device, args.img_size)
#         logging.info(f"Predictions for folder '{args.input_path}': {predictions}")
#     elif args.input_type == 'array':
#         logging.info("Please provide the NumPy array input programmatically for 'array' input type.")
