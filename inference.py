import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, accuracy_score
import matplotlib
matplotlib.use('Agg')  # For headless environments
import matplotlib.pyplot as plt

# --- Import your classes from train.py ---
from train import UnityShapesDataset, SigmaPiSigmaNet

def main():
    # --- PARAMETERS (set them according to your model) ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_classes = 6
    num_factors = 4
    hidden_dim = 192
    dropout_prob = 0.2
    model_path = "best_sigma_pi_sigma_model.pth"  # Path to your trained model
    new_data_dir = "./TestDataset"  # Set path to your new Unity dataset

    # --- TRANSFORM (should match validation transform in train.py) ---
    val_transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    # --- LOAD DATASET ---
    dataset = UnityShapesDataset(new_data_dir, transform=val_transform)
    loader = DataLoader(dataset, batch_size=64, shuffle=False, num_workers=4, pin_memory=True)

    # --- LOAD MODEL ---
    model = SigmaPiSigmaNet(
        num_classes=num_classes,
        num_factors=num_factors,
        hidden_dim=hidden_dim,
        dropout_prob=dropout_prob
    ).to(device)

    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    # --- INFERENCE ---
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            outputs = model(images)
            preds = outputs.argmax(dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels.numpy())

    # --- ACCURACY ---
    acc = accuracy_score(all_labels, all_preds)
    print(f"Test accuracy on new dataset: {acc:.4f}")

    # --- CONFUSION MATRIX ---
    cm = confusion_matrix(all_labels, all_preds)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=dataset.classes)
    fig, ax = plt.subplots(figsize=(8, 8))
    disp.plot(ax=ax, cmap="Blues", xticks_rotation=45)
    plt.title("Confusion Matrix on New Unity Dataset")
    plt.savefig("confusion_matrix.png")
    print("Confusion matrix saved as confusion_matrix.png")

if __name__ == '__main__':
    main()




