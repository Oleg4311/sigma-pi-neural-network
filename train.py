import os
from collections import Counter
from PIL import Image
from sklearn.model_selection import train_test_split

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

# --- Dataset ---
class UnityShapesDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.classes = sorted([
            d for d in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, d))
        ])
        self.class_to_idx = {cls_name: idx for idx, cls_name in enumerate(self.classes)}
        self.samples = []
        for cls_name in self.classes:
            cls_folder = os.path.join(root_dir, cls_name)
            for fname in os.listdir(cls_folder):
                if fname.lower().endswith(".png"):
                    self.samples.append((os.path.join(cls_folder, fname), self.class_to_idx[cls_name]))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, label

# --- Sigma-Pi Layer ---
class SigmaPiLayer(nn.Module):
    def __init__(self, input_dim, num_factors, output_dim):
        super().__init__()
        assert input_dim % num_factors == 0, "input_dim must be divisible by num_factors"
        self.group_size = input_dim // num_factors
        self.linears = nn.ModuleList([
            nn.Linear(self.group_size, output_dim) for _ in range(num_factors)
        ])

    def forward(self, x):
        groups = torch.split(x, self.group_size, dim=1)
        sums = [torch.relu(linear(g)) for linear, g in zip(self.linears, groups)]
        prod = sums[0]
        for s in sums[1:]:
            prod = prod * s
        return prod

# --- Improved Sigma-Pi-Sigma Network (with only one Sigma-Pi layer) ---
class SigmaPiSigmaNet(nn.Module):
    def __init__(self, num_classes=6, num_factors=4, hidden_dim=192, dropout_prob=0.2):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(128, 192, 3, padding=1), nn.BatchNorm2d(192), nn.ReLU(), nn.MaxPool2d(2),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Dropout(dropout_prob)
        )
        self.flatten_dim = 192  # 192 % 4 == 0
        self.sigma_pi1 = SigmaPiLayer(self.flatten_dim, num_factors, hidden_dim)
        self.norm1 = nn.BatchNorm1d(hidden_dim)
        self.sigma = nn.Linear(hidden_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout_prob)
        self.classifier = nn.Linear(hidden_dim, num_classes)

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.sigma_pi1(x)
        x = self.norm1(x)
        x = torch.relu(self.sigma(x))
        x = self.dropout(x)
        x = self.classifier(x)
        return x

# --- L1 Regularization ---
def l1_regularization(model, l1_lambda=5e-6):
    l1_norm = 0.0
    for name, param in model.named_parameters():
        if param.requires_grad and "bias" not in name and "bn" not in name.lower():
            l1_norm += param.abs().sum()
    return l1_lambda * l1_norm

# --- Training ---
def train_model(
    data_dir,
    epochs=180,
    batch_size=128,
    lr=1e-4,
    device=None,
    test_size=0.2,
    l1_lambda=5e-6,
    weight_decay=1e-4,
):
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Stronger augmentation for better accuracy
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(128, scale=(0.8, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(20),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    val_transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    # Load and check dataset
    full_dataset = UnityShapesDataset(data_dir, transform=train_transform)
    print(f"Classes: {full_dataset.classes}")
    labels_all = [label for _, label in full_dataset.samples]
    print("Full dataset label distribution:", Counter(labels_all))

    train_indices, val_indices = train_test_split(
        list(range(len(full_dataset))),
        test_size=test_size,
        random_state=42,
        stratify=labels_all,
    )
    train_dataset = torch.utils.data.Subset(full_dataset, train_indices)
    val_dataset = torch.utils.data.Subset(full_dataset, val_indices)
    val_dataset.dataset.transform = val_transform

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)

    model = SigmaPiSigmaNet(num_classes=len(full_dataset.classes), num_factors=4, hidden_dim=192, dropout_prob=0.2).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.3, patience=5)

    best_val_acc = 0.0
    patience_counter = 0
    early_stopping_patience = 20

    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels) + l1_regularization(model, l1_lambda)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * images.size(0)
            preds = outputs.argmax(dim=1)
            train_correct += (preds == labels).sum().item()
            train_total += labels.size(0)

        train_loss /= train_total
        train_acc = train_correct / train_total

        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item() * images.size(0)
                preds = outputs.argmax(dim=1)
                val_correct += (preds == labels).sum().item()
                val_total += labels.size(0)

        val_loss /= val_total
        val_acc = val_correct / val_total

        scheduler.step(val_loss)

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), "best_sigma_pi_sigma_model.pth")
            patience_counter = 0
        else:
            patience_counter += 1

        print(
            f"Epoch {epoch}/{epochs} | "
            f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} | "
            f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f} | Best Val Acc: {best_val_acc:.4f}"
        )

        if patience_counter > early_stopping_patience:
            print("Early stopping triggered.")
            break

    print("Training complete. Loading best model...")
    model.load_state_dict(torch.load("best_sigma_pi_sigma_model.pth"))
    return model

if __name__ == "__main__":
    dataset_path = "./Dataset"  # Set your dataset path here
    trained_model = train_model(dataset_path)
