import os
import json
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms, datasets
from torch.utils.data import DataLoader
from tqdm import tqdm

try:
    from model import LeNet, resnet34
except Exception as e:
    print("无法导入 model.py 中的 LeNet 或 resnet34，请确认 main_full_eval.py 和 model.py 在同一文件夹。")
    raise e


BASE_DIR = Path(__file__).resolve().parent
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
CLASS_JSON = BASE_DIR / "class_indices.json"
TRAIN_DIR = BASE_DIR / "train"
VAL_DIR = BASE_DIR / "val"


class TinyCNN(nn.Module):
    def __init__(self, num_classes=2):
        super(TinyCNN, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 16 * 16, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x


def read_class_indices():
    if CLASS_JSON.exists():
        with open(CLASS_JSON, "r", encoding="utf-8") as f:
            data = json.load(f)
        return {int(k): v for k, v in data.items()}
    return {0: "no", 1: "yes"}


def write_class_indices_from_dataset(dataset):
    class_to_idx = dataset.class_to_idx
    idx_to_class = {v: k for k, v in class_to_idx.items()}
    with open(CLASS_JSON, "w", encoding="utf-8") as f:
        json.dump({str(k): v for k, v in idx_to_class.items()}, f, ensure_ascii=False, indent=1)
    return idx_to_class


def get_transform(model_name, train=False):
    model_name = model_name.lower()
    if model_name == "tinycnn":
        size = 128
        normalize = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    elif model_name == "resnet34":
        size = 224
        normalize = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    else:
        size = 224
        normalize = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))

    if train:
        return transforms.Compose([
            transforms.Resize((size, size)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(8),
            transforms.ToTensor(),
            normalize,
        ])
    return transforms.Compose([
        transforms.Resize((size, size)),
        transforms.ToTensor(),
        normalize,
    ])


def load_model(model_name):
    model_name = model_name.lower()
    if model_name == "lenet":
        model = LeNet(num_classes=2)
        weight_path = BASE_DIR / "LeNet.pth"
    elif model_name == "resnet34":
        model = resnet34(num_classes=2)
        weight_path = BASE_DIR / "ResNet34.pth"
    elif model_name == "tinycnn":
        model = TinyCNN(num_classes=2)
        weight_path = BASE_DIR / "TinyCNN_cpu.pth"
    else:
        raise ValueError("模型名称只能是 lenet、resnet34 或 tinycnn")

    if not weight_path.exists():
        raise FileNotFoundError(f"没有找到权重文件：{weight_path.name}。如果是 tinycnn，请先选择功能 1 训练。")

    try:
        state_dict = torch.load(weight_path, map_location=DEVICE, weights_only=True)
    except TypeError:
        state_dict = torch.load(weight_path, map_location=DEVICE)
    model.load_state_dict(state_dict)
    model.to(DEVICE)
    model.eval()
    return model, weight_path.name


def train_tinycnn():
    if not TRAIN_DIR.exists() or not VAL_DIR.exists():
        print("没有找到 train 或 val 文件夹，请确认目录完整。")
        return

    train_dataset = datasets.ImageFolder(root=str(TRAIN_DIR), transform=get_transform("tinycnn", train=True))
    val_dataset = datasets.ImageFolder(root=str(VAL_DIR), transform=get_transform("tinycnn", train=False))
    class_map = write_class_indices_from_dataset(train_dataset)

    batch_size = 8
    epochs = 5
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    model = TinyCNN(num_classes=2).to(DEVICE)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0005)

    print("\n开始 CPU 轻量训练 TinyCNN")
    print("训练图片数量：", len(train_dataset))
    print("验证图片数量：", len(val_dataset))
    print("类别对应关系：", class_map)
    print("batch_size：", batch_size)
    print("epochs：", epochs)

    best_acc = 0.0
    save_path = BASE_DIR / "TinyCNN_cpu.pth"
    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        train_bar = tqdm(train_loader, desc=f"epoch {epoch + 1}/{epochs}")
        for images, labels in train_bar:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()
            outputs = model(images)
            loss = loss_fn(outputs, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            train_bar.set_postfix(loss=f"{loss.item():.3f}")

        val_acc, _, _ = evaluate_model_object(model, val_loader)
        avg_loss = total_loss / max(1, len(train_loader))
        print(f"第 {epoch + 1} 轮：loss={avg_loss:.3f}, val_acc={val_acc:.3f}")
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), save_path)

    print("\n训练完成，模型已保存：", save_path.name)
    print(f"最佳验证准确率：{best_acc * 100:.2f}%")


def evaluate_model_object(model, loader):
    correct = 0
    total = 0
    detail_rows = []
    class_correct = {}
    class_total = {}
    idx_to_class = read_class_indices()

    model.eval()
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            outputs = model(images)
            probs = torch.softmax(outputs, dim=1)
            preds = torch.argmax(probs, dim=1)
            for label, pred, prob in zip(labels.cpu(), preds.cpu(), probs.cpu()):
                label_i = int(label.item())
                pred_i = int(pred.item())
                conf = float(prob[pred_i].item())
                total += 1
                ok = label_i == pred_i
                correct += int(ok)
                class_total[label_i] = class_total.get(label_i, 0) + 1
                class_correct[label_i] = class_correct.get(label_i, 0) + int(ok)
                detail_rows.append((idx_to_class.get(label_i, str(label_i)), idx_to_class.get(pred_i, str(pred_i)), conf, ok))

    acc = correct / total if total else 0.0
    class_acc = {}
    for k in sorted(class_total.keys()):
        class_acc[idx_to_class.get(k, str(k))] = (class_correct.get(k, 0), class_total[k], class_correct.get(k, 0) / class_total[k])
    return acc, class_acc, detail_rows


def evaluate_val_set(model_name):
    if not VAL_DIR.exists():
        print("没有找到 val 文件夹。")
        return

    model, weight_name = load_model(model_name)
    val_dataset = datasets.ImageFolder(root=str(VAL_DIR), transform=get_transform(model_name, train=False))
    loader = DataLoader(val_dataset, batch_size=8, shuffle=False, num_workers=0)

    idx_to_class = {v: k for k, v in val_dataset.class_to_idx.items()}
    with open(CLASS_JSON, "w", encoding="utf-8") as f:
        json.dump({str(k): v for k, v in idx_to_class.items()}, f, ensure_ascii=False, indent=1)

    correct = 0
    total = 0
    class_correct = {name: 0 for name in idx_to_class.values()}
    class_total = {name: 0 for name in idx_to_class.values()}
    rows = []

    model.eval()
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            outputs = model(images)
            probs = torch.softmax(outputs, dim=1)
            preds = torch.argmax(probs, dim=1)
            for label, pred, prob in zip(labels.cpu(), preds.cpu(), probs.cpu()):
                true_name = idx_to_class[int(label.item())]
                pred_name = idx_to_class[int(pred.item())]
                conf = float(prob[int(pred.item())].item())
                ok = true_name == pred_name
                correct += int(ok)
                total += 1
                class_total[true_name] += 1
                class_correct[true_name] += int(ok)
                rows.append((true_name, pred_name, conf, ok))

    print("\n========== 完整验证集评估结果 ==========")
    print("模型：", model_name)
    print("权重：", weight_name)
    print("验证集路径：", VAL_DIR)
    print("验证图片总数：", total)
    print(f"整体准确率：{correct}/{total} = {correct / total * 100:.2f}%")
    for name in sorted(class_total.keys()):
        n = class_total[name]
        c = class_correct[name]
        print(f"类别 {name}：{c}/{n} = {c / n * 100:.2f}%")

    print("\n前 10 条预测明细：")
    for i, (true_name, pred_name, conf, ok) in enumerate(rows[:10], 1):
        flag = "正确" if ok else "错误"
        print(f"{i:02d}. 真实={true_name}, 预测={pred_name}, 置信度={conf * 100:.2f}%, {flag}")

    result_path = BASE_DIR / f"eval_val_{model_name}.txt"
    with open(result_path, "w", encoding="utf-8") as f:
        f.write("完整验证集评估结果\n")
        f.write(f"模型：{model_name}\n")
        f.write(f"权重：{weight_name}\n")
        f.write(f"验证图片总数：{total}\n")
        f.write(f"整体准确率：{correct}/{total} = {correct / total * 100:.2f}%\n")
        for name in sorted(class_total.keys()):
            n = class_total[name]
            c = class_correct[name]
            f.write(f"类别 {name}：{c}/{n} = {c / n * 100:.2f}%\n")
        f.write("\n逐张预测明细：\n")
        for i, (true_name, pred_name, conf, ok) in enumerate(rows, 1):
            flag = "正确" if ok else "错误"
            f.write(f"{i:03d}. 真实={true_name}, 预测={pred_name}, 置信度={conf * 100:.2f}%, {flag}\n")
    print("\n评估结果已保存：", result_path.name)


def predict_one_image(model_name, image_name):
    image_path = BASE_DIR / image_name
    if not image_path.exists():
        print("没有找到图片：", image_path)
        return

    model, weight_name = load_model(model_name)
    idx_to_class = read_class_indices()
    transform = get_transform(model_name, train=False)

    img = Image.open(image_path).convert("RGB")
    tensor = transform(img).unsqueeze(0).to(DEVICE)

    model.eval()
    with torch.no_grad():
        output = model(tensor)
        prob = torch.softmax(output.squeeze(0).cpu(), dim=0)
        pred_idx = int(torch.argmax(prob).item())
        pred_name = idx_to_class.get(pred_idx, str(pred_idx))

    print("\n========== 单张测试图片预测结果 ==========")
    print("图片：", image_name)
    print("模型：", model_name)
    print("权重：", weight_name)
    print("预测类别：", pred_name)
    print(f"置信度：{prob[pred_idx].item() * 100:.2f}%")
    print("各类别概率：")
    for i in range(len(prob)):
        print(f"  {idx_to_class.get(i, str(i))}: {prob[i].item() * 100:.2f}%")


def predict_two_test_images(model_name):
    predict_one_image(model_name, "test_yes.jpg")
    predict_one_image(model_name, "test_no.png")


def evaluate_all_models():
    for model_name in ["lenet", "resnet34", "tinycnn"]:
        try:
            evaluate_val_set(model_name)
        except Exception as e:
            print(f"\n{model_name} 验证集评估失败：{e}")


def predict_all_models_on_test_images():
    for model_name in ["lenet", "resnet34", "tinycnn"]:
        try:
            predict_two_test_images(model_name)
        except Exception as e:
            print(f"\n{model_name} 测试图片预测失败：{e}")


def print_menu():
    print("\n========== 脑肿瘤 MRI 二分类识别系统：完整评估版 ==========")
    print("当前目录：", BASE_DIR)
    print("运行设备：", DEVICE)
    print("1. CPU 轻量训练 TinyCNN")
    print("2. LeNet 评估完整 val 数据集")
    print("3. ResNet34 评估完整 val 数据集")
    print("4. TinyCNN 评估完整 val 数据集")
    print("5. 三个模型全部评估完整 val 数据集")
    print("6. LeNet 预测两张测试图片")
    print("7. ResNet34 预测两张测试图片")
    print("8. TinyCNN 预测两张测试图片")
    print("9. 三个模型全部预测两张测试图片")
    print("10. 自定义图片预测")
    print("0. 退出")


def main():
    while True:
        print_menu()
        choice = input("请选择功能编号：").strip()
        try:
            if choice == "1":
                train_tinycnn()
            elif choice == "2":
                evaluate_val_set("lenet")
            elif choice == "3":
                evaluate_val_set("resnet34")
            elif choice == "4":
                evaluate_val_set("tinycnn")
            elif choice == "5":
                evaluate_all_models()
            elif choice == "6":
                predict_two_test_images("lenet")
            elif choice == "7":
                predict_two_test_images("resnet34")
            elif choice == "8":
                predict_two_test_images("tinycnn")
            elif choice == "9":
                predict_all_models_on_test_images()
            elif choice == "10":
                model_name = input("请输入模型名称 lenet / resnet34 / tinycnn：").strip().lower()
                image_name = input("请输入图片文件名，例如 test_yes.jpg：").strip()
                predict_one_image(model_name, image_name)
            elif choice == "0":
                print("程序结束。")
                break
            else:
                print("输入无效，请重新输入。")
        except Exception as e:
            print("运行出错：", e)


if __name__ == "__main__":
    main()
