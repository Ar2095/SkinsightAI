import os
from pathlib import Path
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms, models
from PIL import Image
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# === Custom Dataset ===
class SkinDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = Path(root_dir)
        self.transform = transform
        self.samples = []
        self.subtype_to_idx = {}

        # Adjusted folder traversal for your dataset structure:
        # dataset/
        #   benign/
        #     benign-xxx/
        #     benign-yyy/
        #   malignant/
        #     malignant-xxx/
        #     malignant-yyy/
        for binary_class_dir in self.root_dir.glob("*"):
            if not binary_class_dir.is_dir():
                continue
            for subtype_dir in binary_class_dir.glob("*"):
                if not subtype_dir.is_dir():
                    continue
                subtype_name = subtype_dir.name.lower()
                binary_label = 0 if "benign" in binary_class_dir.name.lower() else 1
                if subtype_name not in self.subtype_to_idx:
                    self.subtype_to_idx[subtype_name] = len(self.subtype_to_idx)
                subtype_label = self.subtype_to_idx[subtype_name]

                for img_path in subtype_dir.glob("*.*"):
                    if img_path.suffix.lower() not in [".jpg", ".jpeg", ".png"]:
                        continue
                    self.samples.append((img_path, binary_label, subtype_label))

        self.idx_to_subtype = {v: k for k, v in self.subtype_to_idx.items()}

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, binary_label, subtype_label = self.samples[idx]
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, torch.tensor(binary_label, dtype=torch.float32), torch.tensor(subtype_label, dtype=torch.long)

# === Transforms ===
train_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

# === Model with Two Heads (ResNet18) ===
class DualHeadResNet(nn.Module):
    def __init__(self, num_subtypes):
        super().__init__()
        from torchvision.models import ResNet18_Weights
        self.base = models.resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
        num_ftrs = self.base.fc.in_features
        self.base.fc = nn.Identity()  # remove default classifier

        self.binary_head = nn.Sequential(
            nn.Linear(num_ftrs, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 1)
        )

        self.subtype_head = nn.Sequential(
            nn.Linear(num_ftrs, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, num_subtypes)
        )

    def forward(self, x):
        features = self.base(x)
        binary_out = self.binary_head(features)
        subtype_out = self.subtype_head(features)
        return binary_out, subtype_out

# === Losses & Optimizer ===
bce_loss = nn.BCEWithLogitsLoss()
ce_loss = nn.CrossEntropyLoss()

# === Training ===
def train(model, train_loader, optimizer, epochs=10):
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        all_bin_pred, all_bin_true = [], []
        all_sub_pred, all_sub_true = [], []

        print(f"\n--- Epoch {epoch+1}/{epochs} ---")

        for batch_idx, (x, bin_y, sub_y) in enumerate(train_loader):
            x, bin_y, sub_y = x.to(device), bin_y.to(device), sub_y.to(device)
            optimizer.zero_grad()
            bin_out, sub_out = model(x)

            loss_bin = bce_loss(bin_out, bin_y.unsqueeze(1))
            loss_sub = ce_loss(sub_out, sub_y)
            loss = loss_bin + loss_sub

            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            all_bin_pred += (torch.sigmoid(bin_out).cpu().detach().numpy() > 0.5).astype(int).flatten().tolist()
            all_bin_true += bin_y.cpu().numpy().tolist()
            all_sub_pred += sub_out.argmax(dim=1).cpu().numpy().tolist()
            all_sub_true += sub_y.cpu().numpy().tolist()

            if batch_idx % 50 == 0:
                print(f"  Batch {batch_idx}/{len(train_loader)} | Batch Loss: {loss.item():.4f}")

        bin_acc = accuracy_score(all_bin_true, all_bin_pred)
        sub_acc = accuracy_score(all_sub_true, all_sub_pred)
        print(f"Epoch {epoch+1} Summary | Loss: {total_loss:.4f} | Binary Acc: {bin_acc:.4f} | Subtype Acc: {sub_acc:.4f}")

# === Evaluation ===
def evaluate(model, val_loader, dataset):
    model.eval()
    all_bin_pred, all_bin_true = [], []
    all_sub_pred, all_sub_true = [], []

    with torch.no_grad():
        for x, bin_y, sub_y in val_loader:
            x, bin_y, sub_y = x.to(device), bin_y.to(device), sub_y.to(device)
            bin_out, sub_out = model(x)
            all_bin_pred += (torch.sigmoid(bin_out).cpu().numpy() > 0.5).astype(int).flatten().tolist()
            all_bin_true += bin_y.cpu().numpy().tolist()
            all_sub_pred += sub_out.argmax(dim=1).cpu().numpy().tolist()
            all_sub_true += sub_y.cpu().numpy().tolist()

    bin_acc = accuracy_score(all_bin_true, all_bin_pred)
    sub_acc = accuracy_score(all_sub_true, all_sub_pred)
    print("\n=== Evaluation ===")
    print(f"Binary Accuracy: {bin_acc:.4f}")
    print(f"Subtype Accuracy: {sub_acc:.4f}\n")

    print("Subtype Classification Report:")
    print(classification_report(all_sub_true, all_sub_pred, target_names=list(dataset.subtype_to_idx.keys())))

    cm = confusion_matrix(all_sub_true, all_sub_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=list(dataset.subtype_to_idx.keys()),
                yticklabels=list(dataset.subtype_to_idx.keys()))
    plt.title("Subtype Confusion Matrix")
    plt.ylabel("True")
    plt.xlabel("Predicted")
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.show()

# === Main ===
if __name__ == "__main__":
    dataset = SkinDataset("dataset", transform=train_transforms)

    print("Using device:", device)
    print(f"Total samples: {len(dataset)}")
    print(f"Subtypes ({len(dataset.subtype_to_idx)}): {list(dataset.subtype_to_idx.keys())}")  # This will print your subtype classes

    val_size = int(0.2 * len(dataset))
    train_size = len(dataset) - val_size
    train_data, val_data = random_split(dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_data, batch_size=64, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_data, batch_size=64, shuffle=False, num_workers=4, pin_memory=True)

    model = DualHeadResNet(num_subtypes=len(dataset.subtype_to_idx)).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    train(model, train_loader, optimizer, epochs=10)
    evaluate(model, val_loader, dataset)

    torch.save(model.state_dict(), "skin_model.pth")
    print("Model saved to skin_model.pth")
