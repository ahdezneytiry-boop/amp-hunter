import json
import os
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from tqdm import tqdm

from model_cpu_light import TinyCNN

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT
TRAIN_DIR = DATA_DIR / "train"
VAL_DIR = DATA_DIR / "val"
CLASS_JSON = ROOT / "class_indices.json"
SAVE_PATH = ROOT / "TinyCNN_cpu.pth"

IMG_SIZE = 128
BATCH_SIZE = 8
EPOCHS = 3
LR = 0.001
NUM_WORKERS = 0


def main():
    torch.set_num_threads(max(1, min(4, os.cpu_count() or 1)))
    device = torch.device("cpu")
    print(f"using {device} device")

    if not TRAIN_DIR.exists() or not VAL_DIR.exists():
        raise FileNotFoundError("请确认 train/yes、train/no、val/yes、val/no 文件夹和本脚本在同一个 BRAIN_TUMOR 目录下。")

    data_transform = {
        "train": transforms.Compose([
            transforms.Resize((IMG_SIZE, IMG_SIZE)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ]),
        "val": transforms.Compose([
            transforms.Resize((IMG_SIZE, IMG_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
    }

    train_dataset = datasets.ImageFolder(root=str(TRAIN_DIR), transform=data_transform["train"])
    val_dataset = datasets.ImageFolder(root=str(VAL_DIR), transform=data_transform["val"])

    class_dict = {v: k for k, v in train_dataset.class_to_idx.items()}
    CLASS_JSON.write_text(json.dumps(class_dict, indent=1, ensure_ascii=False), encoding="utf-8")

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)

    print(f"training images: {len(train_dataset)}, validation images: {len(val_dataset)}")
    print(f"batch_size: {BATCH_SIZE}, image_size: {IMG_SIZE}, epochs: {EPOCHS}")

    net = TinyCNN(num_classes=2).to(device)
    loss_function = nn.CrossEntropyLoss()
    optimizer = optim.Adam(net.parameters(), lr=LR)

    best_acc = 0.0
    for epoch in range(EPOCHS):
        net.train()
        running_loss = 0.0
        start = time.perf_counter()
        train_bar = tqdm(train_loader, desc=f"train epoch {epoch + 1}/{EPOCHS}")

        for images, labels in train_bar:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = net(images)
            loss = loss_function(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            train_bar.set_postfix(loss=f"{loss.item():.3f}")

        net.eval()
        correct = 0
        with torch.no_grad():
            for images, labels in val_loader:
                outputs = net(images.to(device))
                pred = torch.max(outputs, dim=1)[1]
                correct += torch.eq(pred, labels.to(device)).sum().item()

        val_acc = correct / max(1, len(val_dataset))
        avg_loss = running_loss / max(1, len(train_loader))
        print(f"epoch {epoch + 1}: loss={avg_loss:.3f}, val_accuracy={val_acc:.3f}, time={time.perf_counter() - start:.1f}s")

        if val_acc >= best_acc:
            best_acc = val_acc
            torch.save(net.state_dict(), SAVE_PATH)
            print(f"saved best model: {SAVE_PATH.name}")

    print("Finished Training")


if __name__ == "__main__":
    main()
