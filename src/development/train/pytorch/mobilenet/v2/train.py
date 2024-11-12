import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms, models
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from PIL import Image
import numpy as np
import os
import glob
import ssl
import random
from datetime import datetime
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from tqdm import tqdm  # For progress tracking
import logging
import argparse  # For argument parsing

ssl._create_default_https_context = ssl._create_unverified_context

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Dataset class for PyTorch
class ImageDataset(Dataset):
    def __init__(self, image_paths, labels, transform=None):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image = Image.open(self.image_paths[idx]).convert('RGB')
        if self.transform:
            image = self.transform(image)
        label = self.labels[idx]
        return image, label

# Early Stopping class to stop training when validation loss doesn't improve
class EarlyStopping:
    def __init__(self, patience=10, verbose=False, delta=0, path='checkpoint.pth'):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_loss = None
        self.early_stop = False
        self.delta = delta
        self.path = path
        self.best_score = None

    def __call__(self, val_loss, model):
        score = -val_loss
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.verbose:
                print(f"EarlyStopping counter: {self.counter} out of {self.patience}")
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        """Saves the model when validation loss decreases."""
        torch.save(model.state_dict(), self.path)
        if self.verbose:
            print(f"Validation loss decreased. Saving model ...")

class Train:
    def __init__(self, dataset_path, img_size=(80, 80), batch_size=64, epochs=30, lr=1e-4, output_dir=None, patience=10, device='cuda'):
        self.dataset_path = dataset_path
        self.img_size = img_size
        self.batch_size = batch_size
        self.epochs = epochs
        self.lr = lr
        self.model = None
        self.train_loader = None
        self.test_loader = None
        self.lb = None
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}
        self.early_stopping = EarlyStopping(patience=patience, verbose=True, path=os.path.join(output_dir, 'checkpoint.pth'))

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

        logging.info(f"Found {len(X)} images with {len(set(labels))} classes.")

        self.lb = LabelEncoder()
        self.lb.fit(labels)
        labels = self.lb.transform(labels)

        np.save(os.path.join(self.output_dir, 'character_classes.npy'), self.lb.classes_)
        X_train, X_test, y_train, y_test = train_test_split(X, labels, test_size=0.15, stratify=labels, random_state=42)

        # Image transformations (data augmentation)
        transform = transforms.Compose([
            transforms.Resize(self.img_size),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

        self.train_loader = DataLoader(ImageDataset(X_train, y_train, transform), batch_size=self.batch_size, shuffle=True)
        self.test_loader = DataLoader(ImageDataset(X_test, y_test, transform), batch_size=self.batch_size, shuffle=False)

    def create_model(self):
        self.model = models.mobilenet_v2(pretrained=True)
        self.model.classifier[1] = nn.Linear(self.model.last_channel, len(self.lb.classes_))  # Adjust final layer
        self.model = self.model.to(self.device)
        
        # Save the model architecture to a text file
        architecture_path = os.path.join(self.output_dir, 'model_architecture.txt')
        with open(architecture_path, 'w') as f:
            f.write(str(self.model))  # Save the model architecture in the file

        logging.info(f"Model architecture saved to {architecture_path}")

    def train_model(self):
        logging.info("Starting training process...")
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(self.model.parameters(), lr=self.lr)

        for epoch in range(self.epochs):
            self.model.train()
            running_loss, running_corrects = 0.0, 0

            # Use tqdm to track the progress per batch
            progress_bar = tqdm(self.train_loader, desc=f"Epoch {epoch+1}/{self.epochs}")

            for inputs, labels in progress_bar:
                inputs, labels = inputs.to(self.device), labels.to(self.device)

                # Ensure labels are of type LongTensor
                labels = labels.long()

                optimizer.zero_grad()
                outputs = self.model(inputs)

                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                running_loss += loss.item() * inputs.size(0)
                preds = torch.argmax(outputs, dim=1)
                running_corrects += torch.sum(preds == labels.data)

                # Update progress bar with loss and accuracy
                progress_bar.set_postfix(loss=loss.item(), accuracy=(running_corrects.double() / len(self.train_loader.dataset)).item())

            epoch_loss = running_loss / len(self.train_loader.dataset)
            epoch_acc = running_corrects.double() / len(self.train_loader.dataset)
            self.history['train_loss'].append(epoch_loss)
            self.history['train_acc'].append(epoch_acc.item())

            print(f"Epoch {epoch+1}/{self.epochs}, Loss: {epoch_loss:.4f}, Accuracy: {epoch_acc:.4f}")

            # Validate after each epoch
            val_loss = self.validate_model(epoch)

            # Check early stopping
            self.early_stopping(val_loss, self.model)

            if self.early_stopping.early_stop:
                print("Early stopping")
                break
            
        self.plot_training_results()

    def validate_model(self, epoch):
        self.model.eval()
        running_loss, running_corrects = 0.0, 0
        criterion = nn.CrossEntropyLoss()

        # Use tqdm for tracking validation progress
        progress_bar = tqdm(self.test_loader, desc=f"Validating Epoch {epoch+1}/{self.epochs}")

        with torch.no_grad():
            for inputs, labels in progress_bar:
                inputs, labels = inputs.to(self.device), labels.to(self.device)

                # Ensure labels are of type LongTensor
                labels = labels.long()

                outputs = self.model(inputs)
                loss = criterion(outputs, labels)

                running_loss += loss.item() * inputs.size(0)
                preds = torch.argmax(outputs, dim=1)
                running_corrects += torch.sum(preds == labels.data)

                # Update progress bar with validation loss and accuracy
                progress_bar.set_postfix(loss=loss.item(), accuracy=(running_corrects.double() / len(self.test_loader.dataset)).item())

        epoch_loss = running_loss / len(self.test_loader.dataset)
        epoch_acc = running_corrects.double() / len(self.test_loader.dataset)
        self.history['val_loss'].append(epoch_loss)
        self.history['val_acc'].append(epoch_acc.item())

        print(f"Validation Loss: {epoch_loss:.4f}, Validation Accuracy: {epoch_acc:.4f}")
        return epoch_loss

    def test_model(self, test_dataset_path):
        # Loading test dataset
        test_dataset_paths = glob.glob(os.path.join(test_dataset_path, "**/*.*"), recursive=True)
        labels = [image_path.split(os.path.sep)[-2] for image_path in test_dataset_paths]

        transform = transforms.Compose([
            transforms.Resize(self.img_size),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

        # Initialize the test dataset and DataLoader
        test_dataset = ImageDataset(test_dataset_paths, labels, transform)
        test_loader = DataLoader(test_dataset, batch_size=self.batch_size, shuffle=False)

        running_corrects = 0
        self.model.eval()

        # Use tqdm for testing progress
        progress_bar = tqdm(test_loader, desc="Testing Model")

        with torch.no_grad():
            for inputs, labels in progress_bar:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                labels = labels.long()

                outputs = self.model(inputs)
                preds = torch.argmax(outputs, dim=1)

                running_corrects += torch.sum(preds == labels.data)

        accuracy = running_corrects.double() / len(test_dataset)
        print(f"Test Accuracy: {accuracy:.4f}")

    def plot_training_results(self):
        fig, ax = plt.subplots(1, 2, figsize=(12, 5))
        epochs_range = range(1, len(self.history['train_loss']) + 1)

        # Plot training and validation loss
        ax[0].plot(epochs_range, self.history['train_loss'], label='Train Loss')
        ax[0].plot(epochs_range, self.history['val_loss'], label='Validation Loss')
        ax[0].set_title('Loss')
        ax[0].legend()

        # Plot training and validation accuracy
        ax[1].plot(epochs_range, self.history['train_acc'], label='Train Accuracy')
        ax[1].plot(epochs_range, self.history['val_acc'], label='Validation Accuracy')
        ax[1].set_title('Accuracy')
        ax[1].legend()

        plt.savefig(os.path.join(self.output_dir, 'training_results.png'))
        plt.show()


# Example usage with argparse for setting default parameters
if __name__ == "__main__":
    dataset = r"D:\engine\smart_parking\train_model\dataset\bg-black-reduksi-20240925"
    # dataset = r"D:\engine\smart_parking\train_model\dataset\bg-black-reduksi-20240921"
    parser = argparse.ArgumentParser(description='Train MobileNetV2 for character recognition')
    parser.add_argument('--dataset', type=str, default=dataset, help='Path to the dataset')
    parser.add_argument('--batch_size', '-b', type=int, default=128, help='Batch size for training')
    parser.add_argument('--epochs', '-e', type=int, default=100, help='Number of epochs to train')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--output_dir', '-o', type=str, default=r"D:\engine\smart_parking\train_model\model-training\pytorch\mobilenet\v2\models\new-model", help='Directory to save models and results')
    parser.add_argument('--test_dataset', type=str, help='Path to the test dataset for evaluation')
    parser.add_argument('--mode', '-m', type=str, default='train', choices=['train', 'eval', 'test'], help="Mode: 'train', 'eval' (evaluate on validation set), or 'test' (evaluate on test set)")

    args = parser.parse_args()

    timestamp = datetime.now().strftime('%Y%m%d-%H-%M-%S')
    output_dir = os.path.join(args.output_dir, timestamp)

    train = Train(dataset_path=args.dataset, batch_size=args.batch_size, epochs=args.epochs, lr=args.lr, output_dir=output_dir)

    if args.mode == 'train':
        train.load_dataset()
        train.create_model()
        train.train_model()

    elif args.mode == 'test' and args.test_dataset:
        train.load_dataset()  # Load the training dataset for label encoding
        train.create_model()
        train.test_model(test_dataset_path=args.test_dataset)

    else:
        print("Please provide a valid mode and dataset.")
