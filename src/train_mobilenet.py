"""Train a MobileNetV2 classifier using transfer learning."""

import argparse
import copy
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from sklearn.metrics import accuracy_score, f1_score
from sklearn.utils.class_weight import compute_class_weight
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms


class ImageClassificationDataset(Dataset):
    """Dataset backed by a CSV containing image paths and class labels."""

    def __init__(self, dataframe, class_to_idx, transform=None):
        self.dataframe = dataframe.reset_index(drop=True)
        self.class_to_idx = class_to_idx
        self.transform = transform

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, index):
        row = self.dataframe.iloc[index]

        with Image.open(row["path"]) as image:
            image = image.convert("RGB")

        if self.transform:
            image = self.transform(image)

        label = self.class_to_idx[row["class_name"]]
        return image, label


def get_transforms():
    """Return training and evaluation transformations."""
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]

    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(10),
        transforms.ColorJitter(
            brightness=0.12,
            contrast=0.12,
            saturation=0.10,
        ),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])

    eval_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])

    return train_transform, eval_transform


def run_epoch(model, loader, criterion, device, optimizer=None):
    """Run one training or evaluation epoch."""
    training = optimizer is not None
    model.train(training)

    total_loss = 0.0
    all_labels = []
    all_predictions = []

    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)

        if training:
            optimizer.zero_grad()

        with torch.set_grad_enabled(training):
            logits = model(images)
            loss = criterion(logits, labels)

            if training:
                loss.backward()
                optimizer.step()

        total_loss += loss.item() * images.size(0)

        predictions = logits.argmax(dim=1)
        all_labels.extend(labels.cpu().numpy())
        all_predictions.extend(predictions.cpu().numpy())

    average_loss = total_loss / len(loader.dataset)
    accuracy = accuracy_score(all_labels, all_predictions)
    macro_f1 = f1_score(
        all_labels,
        all_predictions,
        average="macro",
        zero_division=0,
    )

    return average_loss, accuracy, macro_f1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--train-csv",
        required=True,
        help="CSV containing path and class_name columns for training.",
    )
    parser.add_argument(
        "--val-csv",
        required=True,
        help="CSV containing path and class_name columns for validation.",
    )
    parser.add_argument(
        "--output",
        default="models/mobilenet_v2_best.pth",
    )
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=0.001)
    parser.add_argument("--num-workers", type=int, default=2)
    args = parser.parse_args()

    if args.epochs < 1 or args.batch_size < 1:
        parser.error("epochs and batch-size must be positive")

    train_df = pd.read_csv(args.train_csv)
    val_df = pd.read_csv(args.val_csv)

    required_columns = {"path", "class_name"}
    for name, dataframe in [
        ("training", train_df),
        ("validation", val_df),
    ]:
        if not required_columns.issubset(dataframe.columns):
            raise ValueError(
                f"{name} CSV must contain path and class_name columns"
            )
        if dataframe.empty:
            raise ValueError(f"{name} CSV is empty")

    class_names = sorted(train_df["class_name"].unique().tolist())
    class_to_idx = {
        class_name: index
        for index, class_name in enumerate(class_names)
    }

    unknown_labels = set(val_df["class_name"]) - set(class_names)
    if unknown_labels:
        raise ValueError(
            f"Validation labels not present in training data: {unknown_labels}"
        )

    train_transform, eval_transform = get_transforms()

    train_dataset = ImageClassificationDataset(
        train_df, class_to_idx, train_transform
    )
    val_dataset = ImageClassificationDataset(
        val_df, class_to_idx, eval_transform
    )

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )
    print(f"Using device: {device}")

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )

    model = models.mobilenet_v2(
        weights=models.MobileNet_V2_Weights.DEFAULT
    )

    for parameter in model.features.parameters():
        parameter.requires_grad = False

    model.classifier[1] = nn.Linear(
        model.classifier[1].in_features,
        len(class_names),
    )
    model = model.to(device)

    weights = compute_class_weight(
        class_weight="balanced",
        classes=np.asarray(class_names),
        y=train_df["class_name"].to_numpy(),
    )
    class_weights = torch.tensor(
        weights, dtype=torch.float32, device=device
    )

    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.Adam(
        model.classifier.parameters(),
        lr=args.learning_rate,
    )

    best_f1 = -1.0
    best_state = None
    best_epoch = 0
    start_time = time.time()

    for epoch in range(1, args.epochs + 1):
        train_loss, train_acc, train_f1 = run_epoch(
            model, train_loader, criterion, device, optimizer
        )
        val_loss, val_acc, val_f1 = run_epoch(
            model, val_loader, criterion, device
        )

        print(
            f"Epoch {epoch}/{args.epochs} | "
            f"train loss={train_loss:.4f}, "
            f"train accuracy={train_acc:.4f}, "
            f"train macro F1={train_f1:.4f} | "
            f"val loss={val_loss:.4f}, "
            f"val accuracy={val_acc:.4f}, "
            f"val macro F1={val_f1:.4f}"
        )

        if val_f1 > best_f1:
            best_f1 = val_f1
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    torch.save(
        {
            "state_dict": best_state,
            "class_names": class_names,
            "epoch": best_epoch,
            "val_macro_f1": best_f1,
        },
        output_path,
    )

    elapsed = time.time() - start_time
    print(f"Best epoch: {best_epoch}")
    print(f"Best validation macro F1: {best_f1:.4f}")
    print(f"Training time: {elapsed:.1f} seconds")
    print(f"Saved checkpoint: {output_path}")


if __name__ == "__main__":
    main()
