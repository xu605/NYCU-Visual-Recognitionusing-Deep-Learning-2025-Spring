import os
import csv
import glob
import json
import zipfile
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models
from torch.utils.data import DataLoader, Dataset
from torch.optim.lr_scheduler import LambdaLR
from PIL import Image
from torchvision.models import ResNeXt50_32X4D_Weights
import tqdm

# Configurations
BATCH_SIZE = 32  # Increased for better training stability
EPOCHS = 30  # More epochs for better convergence
LEARNING_RATE = 0.0005  # Lower learning rate for stable training
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DATA_DIR = "./data"
MODEL_PATH = "resnet50_model.pth"

# Data Transformations with Data Augmentation
transform = transforms.Compose([
    transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(15),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

val_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

test_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])


class TestDataset(Dataset):
    """Custom Dataset for Test Data"""
    def __init__(self, root, transform=None):
        self.image_paths = sorted(glob.glob(os.path.join(root, "*.jpg")))
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, os.path.basename(img_path)


# Load Dataset
train_dataset = datasets.ImageFolder(root=os.path.join(DATA_DIR, "train"), transform=transform)
val_dataset = datasets.ImageFolder(root=os.path.join(DATA_DIR, "val"), transform=val_transform)
test_dataset = TestDataset(root=os.path.join(DATA_DIR, "test"), transform=test_transform)

with open("class_to_idx.json", "w") as f:
    json.dump(train_dataset.class_to_idx, f)

with open("class_to_idx.json", "r") as f:
    class_to_idx = json.load(f)

idx_to_class = {v: k for k, v in class_to_idx.items()}

# Define Model
model = models.resnext50_32x4d(weights=ResNeXt50_32X4D_Weights.IMAGENET1K_V1)
model.fc = nn.Sequential(
    nn.Dropout(0.5),
    nn.Linear(model.fc.in_features, 100)
)
model = model.to(DEVICE)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)

# Loss and Optimizer
criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
optimizer = optim.SGD(model.parameters(), lr=LEARNING_RATE, momentum=0.9, weight_decay=5e-4)


def warmup_cosine_decay(epoch):
    """Warmup for first 5 epochs, then cosine decay"""
    if epoch < 5:
        return epoch / 5
    return 0.5 * (1 + np.cos(np.pi * (epoch - 5) / (EPOCHS - 5)))


scheduler = LambdaLR(optimizer, warmup_cosine_decay)


def mixup_data(x, y, alpha=0.2):
    """Applies MixUp data augmentation"""
    lam = np.random.beta(alpha, alpha)
    batch_size = x.size()[0]
    index = torch.randperm(batch_size)

    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam


def mixup_criterion(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)


if __name__ == "__main__":
    tolerance = 5
    tolerance_count = 0
    best_acc = 0

    for epoch in range(EPOCHS):
        model.train()
        print(f"Epoch {epoch + 1}/{EPOCHS}")
        total_loss, correct, total = 0, 0, 0

        for images, labels in tqdm.tqdm(train_loader):
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            mixed_images, y_a, y_b, lam = mixup_data(images, labels)
            optimizer.zero_grad()
            outputs = model(mixed_images)
            loss = mixup_criterion(criterion, outputs, y_a, y_b, lam)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

        scheduler.step()
        print(f"Epoch {epoch + 1}/{EPOCHS}, Loss: {total_loss / len(train_loader):.4f}, Accuracy: {100 * correct / total:.2f}%")

        model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for images, labels in tqdm.tqdm(val_loader):
                images, labels = images.to(DEVICE), labels.to(DEVICE)
                outputs = model(images)
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()
        print(f"Validation Accuracy (Epoch {epoch + 1}): {100 * correct / total:.2f}%")

        if best_acc < 100 * correct / total:
            best_acc = 100 * correct / total
            tolerance_count = 0
        else:
            tolerance_count += 1
        if tolerance_count >= tolerance:
            break

    torch.save(model.state_dict(), MODEL_PATH)
    print(f"Model saved as {MODEL_PATH}")
