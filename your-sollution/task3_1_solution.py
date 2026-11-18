from copy import deepcopy
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import ConfusionMatrixDisplay, classification_report, f1_score
from sklearn.model_selection import train_test_split
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

matplotlib.use("Agg")
import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "dataset" / "task3_1"
OUTPUT = ROOT / "outputs"
FIGURES = OUTPUT / "task3_1_results"
SEED = 42


def to_tensor(images):
    pixels = torch.from_numpy(images).permute(0, 3, 1, 2).float()
    return (pixels / 255 - 0.5) / 0.5


def make_model():
    layers = []
    channels = 3
    for width in [24, 48, 72]:
        layers.extend([
            nn.Conv2d(channels, width, 3, padding=1, bias=False),
            nn.BatchNorm2d(width),
            nn.ReLU(),
            nn.Conv2d(width, width, 3, padding=1, bias=False),
            nn.BatchNorm2d(width),
            nn.ReLU(),
            nn.MaxPool2d(2),
        ])
        channels = width
    layers.extend([nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(72, 9)])
    return nn.Sequential(*layers)


def run_epoch(model, loader, loss_fn, device, optimizer=None):
    training = optimizer is not None
    model.train(training)
    total_loss = 0
    predictions = []
    correct = 0
    with torch.set_grad_enabled(training):
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)
            if training:
                optimizer.zero_grad()
            scores = model(images)
            loss = loss_fn(scores, labels)
            if training:
                loss.backward()
                optimizer.step()
            total_loss += loss.item() * len(labels)
            predicted = scores.argmax(1)
            correct += (predicted == labels).sum().item()
            predictions.extend(predicted.tolist())
    size = len(loader.dataset)
    return total_loss / size, correct / size, np.array(predictions)


def plot_history(history):
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.5))
    for ax, metric in zip(axes, ["loss", "accuracy"]):
        ax.plot(history["epoch"], history["train_" + metric], label="Training")
        ax.plot(history["epoch"], history["valid_" + metric], label="Validation")
        ax.set(xlabel="Epoch", ylabel=metric.capitalize())
        ax.legend()
    fig.tight_layout()
    fig.savefig(FIGURES / "learning_curves.png", dpi=140)
    plt.close(fig)


def plot_examples(images, titles, filename):
    fig, axes = plt.subplots(3, 3, figsize=(8, 7))
    for ax in axes.flat:
        ax.axis("off")
    for ax, image, title in zip(axes.flat, images, titles):
        ax.imshow(image)
        ax.set_title(title, fontsize=9, wrap=True)
    fig.tight_layout()
    fig.savefig(FIGURES / filename, dpi=130)
    plt.close(fig)


def main():
    torch.manual_seed(SEED)
    torch.set_num_threads(min(4, torch.get_num_threads()))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    FIGURES.mkdir(parents=True, exist_ok=True)
    with np.load(DATA / "train.npz") as data:
        images = data["images"]
        labels = data["prediction"]
        names = data["label_names"]

    train, valid = train_test_split(
        np.arange(len(labels)), test_size=0.2, stratify=labels, random_state=SEED
    )
    samples = []
    for class_id in range(len(names)):
        samples.append(train[labels[train] == class_id][0])
    plot_examples(images[samples], names, "training_examples.png")

    pixels = to_tensor(images)
    targets = torch.from_numpy(labels)
    train_data = TensorDataset(pixels[train], targets[train])
    valid_data = TensorDataset(pixels[valid], targets[valid])
    train_loader = DataLoader(train_data, batch_size=96, shuffle=True)
    valid_loader = DataLoader(valid_data, batch_size=128)
    model = make_model().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=8, gamma=0.4)
    loss_fn = nn.CrossEntropyLoss()
    best_f1 = -1
    history = []

    for epoch in range(1, 25):
        train_loss, train_accuracy, _ = run_epoch(model, train_loader, loss_fn, device, optimizer)
        valid_loss, valid_accuracy, valid_pred = run_epoch(model, valid_loader, loss_fn, device)
        valid_f1 = f1_score(labels[valid], valid_pred, average="macro")
        history.append([epoch, train_loss, valid_loss, train_accuracy, valid_accuracy, valid_f1])
        print(f"Epoch {epoch:2d}  accuracy {valid_accuracy:.3f}  macro F1 {valid_f1:.3f}", flush=True)
        if valid_f1 > best_f1:
            best_f1 = valid_f1
            best_epoch = epoch
            best_state = deepcopy(model.state_dict())
        if epoch - best_epoch >= 5:
            break
        scheduler.step()

    columns = ["epoch", "train_loss", "valid_loss", "train_accuracy", "valid_accuracy", "valid_macro_f1"]
    history = pd.DataFrame(history, columns=columns)
    history.to_csv(FIGURES / "training_history.csv", index=False)
    plot_history(history)

    model.load_state_dict(best_state)
    _, _, predictions = run_epoch(model, valid_loader, loss_fn, device)
    report = classification_report(labels[valid], predictions, target_names=names, output_dict=True)
    rows = pd.DataFrame(report).T.loc[names].copy()
    rows["support"] = rows["support"].astype(int)
    rows["training_count"] = np.bincount(labels[train])
    rows.to_csv(FIGURES / "class_metrics.csv", index_label="tissue")
    display = ConfusionMatrixDisplay.from_predictions(labels[valid], predictions, cmap="Blues")
    display.figure_.tight_layout()
    display.figure_.savefig(FIGURES / "confusion_matrix.png", dpi=140)
    plt.close(display.figure_)

    wrong = np.flatnonzero(labels[valid] != predictions)
    if len(wrong):
        rng = np.random.default_rng(SEED)
        selected = rng.choice(wrong, size=min(9, len(wrong)), replace=False)
        titles = []
        for i in selected:
            titles.append(f"{names[labels[valid[i]]]}\n→ {names[predictions[i]]}")
        plot_examples(images[valid[selected]], titles, "validation_errors.png")

    with np.load(DATA / "test_features.npz") as test:
        test_loader = DataLoader(to_tensor(test["images"]), batch_size=128)
        test_ids = test["image_id"]
    test_predictions = []
    with torch.no_grad():
        for batch in test_loader:
            scores = model(batch.to(device))
            test_predictions.extend(scores.argmax(1).tolist())
    result = pd.DataFrame({"image_id": test_ids, "prediction": test_predictions})
    result.to_csv(OUTPUT / "task3_1_predictions.csv", index=False)


if __name__ == "__main__":
    main()
