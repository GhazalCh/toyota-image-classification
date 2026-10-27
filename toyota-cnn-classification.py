import argparse
import copy
import json
import random
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from PIL import Image, UnidentifiedImageError
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score,
)
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, WeightedRandomSampler, Subset
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm


DEFAULT_DATA_DIR = (
    "/kaggle/input/toyota-cars-over-20k-labeled-images/"
    "toyota_image_dataset_v2/toyota_cars"
)
IMAGE_SIZE = 224
BATCH_SIZE = 32
NUM_CLASSES_TO_USE = 10
VALIDATION_SIZE = 0.20
RANDOM_SEED = 42
NUM_EPOCHS = 15
LEARNING_RATE = 1e-3
PATIENCE = 5
NUM_WORKERS = 0  # Safe default on Windows; increase if appropriate.


def set_seed(seed: int = RANDOM_SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def parse_args():
    parser = argparse.ArgumentParser(description="Train a CNN on Toyota car images.")
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path(DEFAULT_DATA_DIR),
        help="Dataset root with one subfolder per class.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs"),
        help="Directory for model weights, metrics, and plots.",
    )
    parser.add_argument("--epochs", type=int, default=NUM_EPOCHS)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--learning-rate", type=float, default=LEARNING_RATE)
    parser.add_argument("--num-workers", type=int, default=NUM_WORKERS)
    parser.add_argument("--classes", type=int, default=NUM_CLASSES_TO_USE)
    return parser.parse_args()


def is_valid_image(path: str) -> bool:
    """Check an image without changing or deleting the source file."""
    try:
        with Image.open(path) as image:
            image.verify()
        return True
    except (UnidentifiedImageError, OSError, ValueError):
        return False


class ToyotaCNN(nn.Module):
    """Three convolutional blocks followed by adaptive pooling and a classifier."""

    def __init__(self, num_classes: int, dropout: float = 0.3):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Dropout2d(0.1),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Dropout2d(0.2),

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Dropout2d(dropout),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.classifier = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, start_dim=1)
        # Return logits. CrossEntropyLoss applies the appropriate log-softmax.
        return self.classifier(x)


def build_datasets(data_dir: Path, number_of_classes: int):
    if not data_dir.exists() or not data_dir.is_dir():
        raise FileNotFoundError(
            f"Dataset directory not found: {data_dir}\n"
            "Pass the correct location using --data-dir."
        )

    # ImageFolder assigns class indices alphabetically.
    initial_dataset = ImageFolder(root=str(data_dir))
    all_classes = initial_dataset.classes
    if len(all_classes) < 2:
        raise ValueError(
            f"Expected at least two class subfolders in {data_dir}; "
            f"found {len(all_classes)}."
        )

    selected_classes = all_classes[:number_of_classes]
    selected_indices = {initial_dataset.class_to_idx[name] for name in selected_classes}
    valid_samples = []
    invalid_count = 0

    print("Selected classes:")
    for class_name in selected_classes:
        print(f" - {class_name}")

    print("\nChecking image files (source files will not be deleted)...")
    for path, label in tqdm(initial_dataset.samples, desc="Validating images"):
        if label not in selected_indices:
            continue
        if is_valid_image(path):
            valid_samples.append((path, label))
        else:
            invalid_count += 1

    if invalid_count:
        print(f"Skipped {invalid_count} corrupted/unreadable image(s).")

    # Remove classes with no valid files and remap labels to 0..N-1.
    present_old_labels = sorted({label for _, label in valid_samples})
    if len(present_old_labels) < 2:
        raise ValueError("Fewer than two classes contain valid images.")

    old_to_new = {old: new for new, old in enumerate(present_old_labels)}
    class_names = [initial_dataset.classes[old] for old in present_old_labels]
    valid_samples = [(path, old_to_new[label]) for path, label in valid_samples]

    labels = np.array([label for _, label in valid_samples])
    class_counts = np.bincount(labels, minlength=len(class_names))
    if np.any(class_counts < 2):
        rare = [class_names[i] for i, count in enumerate(class_counts) if count < 2]
        raise ValueError(
            "At least two valid images per class are required for a stratified "
            f"train/validation split. Insufficient classes: {rare}"
        )

    # Plot class distribution based on valid files.
    output_dir = None
    return valid_samples, class_names, labels, class_counts


class SamplesDataset(torch.utils.data.Dataset):
    """Dataset view over a fixed list of validated image paths."""

    def __init__(self, samples, class_names, transform=None):
        self.samples = samples
        self.classes = class_names
        self.transform = transform
        self.targets = [label for _, label in samples]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        path, label = self.samples[index]
        with Image.open(path) as image:
            image = image.convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, label


def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss = 0.0
    total_correct = 0
    total_items = 0

    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad(set_to_none=True)
        logits = model(images)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        batch_size = labels.size(0)
        total_loss += loss.item() * batch_size
        total_correct += (logits.argmax(dim=1) == labels).sum().item()
        total_items += batch_size

    return total_loss / max(total_items, 1), total_correct / max(total_items, 1)


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    total_items = 0
    predictions, targets = [], []

    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        logits = model(images)
        loss = criterion(logits, labels)

        batch_size = labels.size(0)
        total_loss += loss.item() * batch_size
        total_items += batch_size
        predictions.extend(logits.argmax(dim=1).cpu().numpy().tolist())
        targets.extend(labels.cpu().numpy().tolist())

    return total_loss / max(total_items, 1), np.array(targets), np.array(predictions)


def main():
    args = parse_args()
    set_seed()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    samples, class_names, labels, class_counts = build_datasets(
        args.data_dir, max(2, args.classes)
    )

    print("\nValid image counts:")
    for name, count in zip(class_names, class_counts):
        print(f" - {name}: {count}")

    # Split indices first so train augmentation never leaks into validation.
    indices = np.arange(len(samples))
    train_indices, val_indices = train_test_split(
        indices,
        test_size=VALIDATION_SIZE,
        random_state=RANDOM_SEED,
        stratify=labels,
    )

    train_transform = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=15),
        transforms.ColorJitter(
            brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05
        ),
        transforms.RandomResizedCrop(IMAGE_SIZE, scale=(0.8, 1.0)),
        transforms.ToTensor(),
    ])
    validation_transform = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
    ])

    train_base = SamplesDataset(samples, class_names, transform=train_transform)
    val_base = SamplesDataset(samples, class_names, transform=validation_transform)
    train_dataset = Subset(train_base, train_indices.tolist())
    val_dataset = Subset(val_base, val_indices.tolist())

    train_labels = labels[train_indices]
    train_class_counts = np.bincount(train_labels, minlength=len(class_names))
    class_weights = 1.0 / np.maximum(train_class_counts, 1)
    sample_weights = class_weights[train_labels]
    sampler = WeightedRandomSampler(
        weights=torch.as_tensor(sample_weights, dtype=torch.double),
        num_samples=len(sample_weights),
        replacement=True,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        sampler=sampler,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nDevice: {device}")
    print(f"Training images: {len(train_dataset)} | Validation images: {len(val_dataset)}")

    model = ToyotaCNN(num_classes=len(class_names)).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
    scheduler = ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=2, min_lr=1e-6
    )

    best_val_loss = float("inf")
    best_state = copy.deepcopy(model.state_dict())
    epochs_without_improvement = 0
    history = []

    for epoch in range(args.epochs):
        train_loss, train_accuracy = train_one_epoch(
            model, train_loader, criterion, optimizer, device
        )
        val_loss, y_true, y_pred = evaluate(
            model, val_loader, criterion, device
        )
        scheduler.step(val_loss)

        row = {
            "epoch": epoch + 1,
            "train_loss": train_loss,
            "train_accuracy": train_accuracy,
            "validation_loss": val_loss,
            "learning_rate": optimizer.param_groups[0]["lr"],
        }
        history.append(row)
        print(
            f"Epoch {epoch + 1}/{args.epochs} | "
            f"train_loss={train_loss:.4f} | train_acc={train_accuracy:.4f} | "
            f"val_loss={val_loss:.4f} | lr={row['learning_rate']:.2e}"
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = copy.deepcopy(model.state_dict())
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= PATIENCE:
                print(f"Early stopping: validation loss did not improve for {PATIENCE} epochs.")
                break

    model.load_state_dict(best_state)
    val_loss, y_true, y_pred = evaluate(model, val_loader, criterion, device)

    metrics = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision_macro": float(precision_score(
            y_true, y_pred, average="macro", zero_division=0
        )),
        "recall_macro": float(recall_score(
            y_true, y_pred, average="macro", zero_division=0
        )),
        "f1_macro": float(f1_score(
            y_true, y_pred, average="macro", zero_division=0
        )),
        "validation_loss": float(val_loss),
        "class_names": class_names,
    }
    print("\nFinal validation metrics (best checkpoint):")
    print(json.dumps(metrics, indent=2))
    print("\nClassification report:")
    print(classification_report(
        y_true, y_pred, labels=list(range(len(class_names))),
        target_names=class_names, zero_division=0
    ))

    # Save checkpoint with label mapping for later inference.
    checkpoint_path = args.output_dir / "toyota_cnn_best.pth"
    torch.save({
        "model_state_dict": model.state_dict(),
        "class_names": class_names,
        "image_size": IMAGE_SIZE,
    }, checkpoint_path)

    with (args.output_dir / "metrics.json").open("w", encoding="utf-8") as file:
        json.dump(metrics, file, indent=2)
    with (args.output_dir / "training_history.json").open("w", encoding="utf-8") as file:
        json.dump(history, file, indent=2)

    # Training curves.
    epochs = [entry["epoch"] for entry in history]
    plt.figure(figsize=(8, 5))
    plt.plot(epochs, [entry["train_loss"] for entry in history], label="Train loss")
    plt.plot(epochs, [entry["validation_loss"] for entry in history], label="Validation loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training and Validation Loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig(args.output_dir / "loss_curve.png", dpi=150)
    plt.close()

    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(class_names))))
    fig, ax = plt.subplots(figsize=(max(8, len(class_names) * 0.8), 7))
    image = ax.imshow(cm, interpolation="nearest", cmap="Blues")
    ax.set_title("Confusion Matrix")
    fig.colorbar(image, ax=ax)
    ticks = np.arange(len(class_names))
    ax.set_xticks(ticks, labels=class_names, rotation=45, ha="right")
    ax.set_yticks(ticks, labels=class_names)
    threshold = cm.max() / 2 if cm.size and cm.max() else 0
    for i, j in np.ndindex(cm.shape):
        ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                color="white" if cm[i, j] > threshold else "black")
    ax.set_ylabel("True label")
    ax.set_xlabel("Predicted label")
    fig.tight_layout()
    fig.savefig(args.output_dir / "confusion_matrix.png", dpi=150)
    plt.close(fig)

    print(f"\nSaved best model to: {checkpoint_path}")
    print(f"Saved metrics and plots to: {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
