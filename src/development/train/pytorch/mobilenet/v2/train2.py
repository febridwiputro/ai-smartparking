import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import glob
from PIL import Image
from datetime import datetime

# PyTorch GPU configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class CustomDataset(Dataset):
    def __init__(self, image_paths, labels, transform=None):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        image = Image.open(image_path).convert("RGB")
        label = self.labels[idx]
        if self.transform:
            image = self.transform(image)
        return image, label

class Train:
    def __init__(self, dataset_path, img_size=(80, 80), batch_size=64, epochs=30, lr=1e-4, output_dir=None):
        self.dataset_path = dataset_path
        self.img_size = img_size
        self.batch_size = batch_size
        self.epochs = epochs
        self.lr = lr
        self.model = None
        self.train_loader = None
        self.val_loader = None
        self.lb = None

        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        self.output_dir = output_dir

    def load_dataset(self):
        dataset_paths = glob.glob(os.path.join(self.dataset_path, "**/*.*"), recursive=True)
        valid_extensions = [".jpg", ".jpeg", ".png"]

        X, labels = [], []

        for image_path in dataset_paths:
            if any(image_path.endswith(ext) for ext in valid_extensions):
                label = image_path.split(os.path.sep)[-2]
                X.append(image_path)
                labels.append(label)

        if len(X) == 0:
            print(f"[ERROR] No images found in the dataset path: {self.dataset_path}")
            return

        print(f"[INFO] Found {len(X)} images with {len(set(labels))} classes.")
        self.lb = LabelEncoder()
        self.lb.fit(labels)
        labels = self.lb.transform(labels)

        np.save(os.path.join(self.output_dir, 'character_classes.npy'), self.lb.classes_)

        # Split dataset
        trainX, testX, trainY, testY = train_test_split(X, labels, test_size=0.15, stratify=labels, random_state=42)

        # Data augmentation and normalization for training
        data_transforms = {
            'train': transforms.Compose([
                transforms.RandomResizedCrop(self.img_size),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ]),
            'val': transforms.Compose([
                transforms.Resize(self.img_size),
                transforms.CenterCrop(self.img_size),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ]),
        }

        train_dataset = CustomDataset(trainX, trainY, transform=data_transforms['train'])
        val_dataset = CustomDataset(testX, testY, transform=data_transforms['val'])

        self.train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=4)
        self.val_loader = DataLoader(val_dataset, batch_size=self.batch_size, shuffle=False, num_workers=4)

    def create_model(self):
        # Load pretrained MobileNetV2 model
        self.model = models.mobilenet_v2(pretrained=True)
        self.model.classifier[1] = nn.Linear(self.model.last_channel, len(self.lb.classes_))  # Replace the final layer

        self.model = self.model.to(device)

    def train_model(self):
        print("[INFO] Starting training process...")

        # Loss function and optimizer
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(self.model.parameters(), lr=self.lr)

        best_acc = 0.0
        for epoch in range(self.epochs):
            print(f"Epoch {epoch+1}/{self.epochs}")
            print('-' * 10)

            # Each epoch has a training and validation phase
            for phase in ['train', 'val']:
                if phase == 'train':
                    self.model.train()  # Set model to training mode
                    loader = self.train_loader
                else:
                    self.model.eval()  # Set model to evaluate mode
                    loader = self.val_loader

                running_loss = 0.0
                running_corrects = 0

                # Iterate over data.
                for inputs, labels in loader:
                    inputs, labels = inputs.to(device), labels.to(device)

                    # Zero the parameter gradients
                    optimizer.zero_grad()

                    # Forward pass
                    with torch.set_grad_enabled(phase == 'train'):
                        outputs = self.model(inputs)
                        _, preds = torch.max(outputs, 1)
                        loss = criterion(outputs, labels)

                        # Backward + optimize only if in training phase
                        if phase == 'train':
                            loss.backward()
                            optimizer.step()

                    # Statistics
                    running_loss += loss.item() * inputs.size(0)
                    running_corrects += torch.sum(preds == labels.data)

                epoch_loss = running_loss / len(loader.dataset)
                epoch_acc = running_corrects.double() / len(loader.dataset)

                print(f"{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}")

                # Deep copy the model
                if phase == 'val' and epoch_acc > best_acc:
                    best_acc = epoch_acc
                    torch.save(self.model.state_dict(), os.path.join(self.output_dir, 'best_model.pth'))

        print(f"[INFO] Best val Acc: {best_acc:.4f}")

    def save_model(self):
        torch.save(self.model.state_dict(), os.path.join(self.output_dir, 'final_model.pth'))
        print("[INFO] Model and weights saved successfully.")

# Dataset and output path configuration
dataset = r"D:\engine\smart_parking\train_model\dataset\bg-black-reduksi-20240925"
timestamp = datetime.now().strftime('%Y%m%d-%H-%M-%S')
output_dir = os.path.join(r"D:\engine\smart_parking\train_model\model-training\pytorch\mobilenet\v2\models", timestamp)

# Initialize and run training
train = Train(dataset_path=dataset, batch_size=128, epochs=100, lr=1e-4, output_dir=output_dir)
train.load_dataset()
train.create_model()
train.train_model()
train.save_model()
