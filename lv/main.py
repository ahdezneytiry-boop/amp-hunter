"""
脑肿瘤 MRI 二分类识别一体化运行程序
适用于普通 Windows 电脑 CPU 环境。

放置位置：
    请将本文件 main.py 放到 BRAIN_TUMOR 项目文件夹内，
    与 model.py、LeNet.pth、ResNet34.pth、train、val、test_yes.jpg、test_no.png 位于同一级。

运行方法：
    python main.py

功能：
    1. 使用已有 LeNet 权重预测图片
    2. 使用已有 ResNet34 权重预测图片
    3. 使用轻量 TinyCNN 在 CPU 上重新训练
    4. 使用训练后的 TinyCNN 预测图片
"""

import os
import json
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from PIL import Image
from torchvision import transforms, datasets
from tqdm import tqdm

try:
    from model import LeNet, resnet34
except Exception as e:
    LeNet = None
    resnet34 = None
    print("提示：未能正常导入 model.py，LeNet 或 ResNet34 功能可能不可用。")
    print("原因：", e)


ROOT = Path(__file__).resolve().parent
DEVICE = torch.device("cpu")


class TinyCNN(nn.Module):
    def __init__(self, num_classes=2):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 16 * 16, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x


def load_class_indices():
    json_path = ROOT / "class_indices.json"
    if json_path.exists():
        with open(json_path, "r", encoding="utf-8") as f:
            return json.load(f)
    return {"0": "no", "1": "yes"}


def save_class_indices(class_to_idx):
    idx_to_class = {str(v): k for k, v in class_to_idx.items()}
    with open(ROOT / "class_indices.json", "w", encoding="utf-8") as f:
        json.dump(idx_to_class, f, indent=2, ensure_ascii=False)


def get_predict_transform(model_name):
    if model_name == "resnet34":
        return transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])
    if model_name == "lenet":
        return transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])
    return transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])


def build_model(model_name):
    model_name = model_name.lower()
    if model_name == "lenet":
        if LeNet is None:
            raise RuntimeError("model.py 中未能导入 LeNet，无法使用 LeNet。")
        model = LeNet(num_classes=2)
        weight_path = ROOT / "LeNet.pth"
    elif model_name == "resnet34":
        if resnet34 is None:
            raise RuntimeError("model.py 中未能导入 resnet34，无法使用 ResNet34。")
        model = resnet34(num_classes=2)
        weight_path = ROOT / "ResNet34.pth"
    elif model_name == "tinycnn":
        model = TinyCNN(num_classes=2)
        weight_path = ROOT / "TinyCNN_cpu.pth"
    else:
        raise ValueError("模型名称只能是 lenet、resnet34 或 tinycnn。")

    if not weight_path.exists():
        raise FileNotFoundError(f"未找到权重文件：{weight_path}")

    state_dict = torch.load(weight_path, map_location=DEVICE)
    model.load_state_dict(state_dict)
    model.to(DEVICE)
    model.eval()
    return model


def predict_image(model_name, image_name):
    image_path = ROOT / image_name
    if not image_path.exists():
        raise FileNotFoundError(f"未找到图片：{image_path}")

    class_indices = load_class_indices()
    transform = get_predict_transform(model_name)
    model = build_model(model_name)

    image = Image.open(image_path).convert("RGB")
    image_tensor = transform(image).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        output = model(image_tensor).squeeze(0).cpu()
        probs = torch.softmax(output, dim=0)
        pred_idx = int(torch.argmax(probs).item())

    pred_name = class_indices.get(str(pred_idx), str(pred_idx))
    print("\n预测完成")
    print("图片：", image_path.name)
    print("模型：", model_name)
    print("预测类别：", pred_name)
    print("置信度：{:.2f}%".format(float(probs[pred_idx]) * 100))
    print("各类别概率：")
    for i, p in enumerate(probs):
        print("  {}: {:.2f}%".format(class_indices.get(str(i), str(i)), float(p) * 100))


def train_tinycnn():
    train_dir = ROOT / "train"
    val_dir = ROOT / "val"
    if not train_dir.exists() or not val_dir.exists():
        raise FileNotFoundError("未找到 train 或 val 文件夹，请确认数据集目录完整。")

    data_transform = {
        "train": transforms.Compose([
            transforms.Resize((128, 128)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]),
        "val": transforms.Compose([
            transforms.Resize((128, 128)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]),
    }

    train_dataset = datasets.ImageFolder(root=str(train_dir), transform=data_transform["train"])
    val_dataset = datasets.ImageFolder(root=str(val_dir), transform=data_transform["val"])
    save_class_indices(train_dataset.class_to_idx)

    batch_size = 8
    epochs = 3
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
    )

    model = TinyCNN(num_classes=2).to(DEVICE)
    loss_function = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.0005)

    best_acc = 0.0
    save_path = ROOT / "TinyCNN_cpu.pth"

    print("\n开始 CPU 轻量训练")
    print("训练图片数量：", len(train_dataset))
    print("验证图片数量：", len(val_dataset))
    print("batch_size：", batch_size)
    print("epochs：", epochs)

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        train_bar = tqdm(train_loader, desc=f"epoch {epoch + 1}/{epochs}")

        for images, labels in train_bar:
            images = images.to(DEVICE)
            labels = labels.to(DEVICE)

            optimizer.zero_grad()
            outputs = model(images)
            loss = loss_function(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            train_bar.set_postfix(loss="{:.3f}".format(loss.item()))

        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for images, labels in val_loader:
                images = images.to(DEVICE)
                labels = labels.to(DEVICE)
                outputs = model(images)
                preds = torch.argmax(outputs, dim=1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)

        val_acc = correct / total if total > 0 else 0
        avg_loss = running_loss / max(len(train_loader), 1)
        print("第 {} 轮：loss={:.3f}, val_acc={:.3f}".format(epoch + 1, avg_loss, val_acc))

        if val_acc >= best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), save_path)

    print("\n训练完成，模型已保存：", save_path.name)
    print("最佳验证准确率：{:.2f}%".format(best_acc * 100))


def show_menu():
    print("\n========== 脑肿瘤 MRI 二分类识别系统 ==========")
    print("当前目录：", ROOT)
    print("运行设备：CPU")
    print("1. 使用 LeNet 预测 test_yes.jpg")
    print("2. 使用 LeNet 预测 test_no.png")
    print("3. 使用 ResNet34 预测 test_yes.jpg")
    print("4. 使用 ResNet34 预测 test_no.png")
    print("5. CPU 轻量训练 TinyCNN")
    print("6. 使用 TinyCNN 预测 test_yes.jpg")
    print("7. 使用 TinyCNN 预测 test_no.png")
    print("8. 自定义图片预测")
    print("0. 退出")


def custom_predict():
    print("\n可选模型：lenet / resnet34 / tinycnn")
    model_name = input("请输入模型名称：").strip().lower()
    image_name = input("请输入图片文件名，例如 test_yes.jpg：").strip()
    predict_image(model_name, image_name)


def main():
    while True:
        show_menu()
        choice = input("请选择功能编号：").strip()
        try:
            if choice == "1":
                predict_image("lenet", "test_yes.jpg")
            elif choice == "2":
                predict_image("lenet", "test_no.png")
            elif choice == "3":
                predict_image("resnet34", "test_yes.jpg")
            elif choice == "4":
                predict_image("resnet34", "test_no.png")
            elif choice == "5":
                train_tinycnn()
            elif choice == "6":
                predict_image("tinycnn", "test_yes.jpg")
            elif choice == "7":
                predict_image("tinycnn", "test_no.png")
            elif choice == "8":
                custom_predict()
            elif choice == "0":
                print("已退出程序。")
                break
            else:
                print("输入无效，请重新选择。")
        except Exception as e:
            print("\n运行出错：", e)
            print("请检查文件是否放在 BRAIN_TUMOR 文件夹内，以及权重、图片、数据集是否存在。")


if __name__ == "__main__":
    main()
