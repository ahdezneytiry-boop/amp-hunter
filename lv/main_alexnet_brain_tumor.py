"""
AlexNet 脑肿瘤 MRI 二分类完整运行程序
放置位置：P5_BRAIN_TUMOR 项目根目录
目录要求：
P5_BRAIN_TUMOR/
├─ train/no, train/yes
├─ val/no, val/yes
├─ test_yes.jpg
├─ test_no.png
├─ class_indices.json
├─ model/                 # 保存 AlexNet.pth
└─ main_alexnet_brain_tumor.py
"""

import csv
import json
from pathlib import Path

import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms, datasets
from torch.utils.data import DataLoader
from tqdm import tqdm

try:
    import matplotlib.pyplot as plt
    import numpy as np
    HAS_PLOT = True
except Exception:
    HAS_PLOT = False


BASE_DIR = Path(__file__).resolve().parent
TRAIN_DIR = BASE_DIR / "train"
VAL_DIR = BASE_DIR / "val"
MODEL_DIR = BASE_DIR / "model"
RESULT_DIR = BASE_DIR / "predict" / "alexnet_outputs"
CLASS_JSON = BASE_DIR / "class_indices.json"
WEIGHT_PATH = MODEL_DIR / "AlexNet.pth"
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class AlexNet(nn.Module):
    def __init__(self, num_classes=2, init_weights=True):
        super(AlexNet, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 48, kernel_size=11, stride=4, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(48, 128, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(128, 192, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(192, 192, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(192, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )
        self.classifier = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Linear(128 * 6 * 6, 2048),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(2048, 2048),
            nn.ReLU(inplace=True),
            nn.Linear(2048, num_classes),
        )
        if init_weights:
            self._initialize_weights()

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, start_dim=1)
        x = self.classifier(x)
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)


def ensure_dirs():
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    RESULT_DIR.mkdir(parents=True, exist_ok=True)


def read_class_indices():
    if CLASS_JSON.exists():
        with open(CLASS_JSON, "r", encoding="utf-8") as f:
            data = json.load(f)
        return {int(k): v for k, v in data.items()}
    return {0: "no", 1: "yes"}


def write_class_indices_from_dataset(dataset):
    idx_to_class = {v: k for k, v in dataset.class_to_idx.items()}
    with open(CLASS_JSON, "w", encoding="utf-8") as f:
        json.dump({str(k): v for k, v in idx_to_class.items()}, f, ensure_ascii=False, indent=1)
    return idx_to_class


def get_transform(train=False):
    if train:
        return transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])


def check_dataset():
    print("\n========== 数据集检查 ==========")
    print("项目目录：", BASE_DIR)
    print("运行设备：", DEVICE)
    ok = True
    for p in [TRAIN_DIR, VAL_DIR]:
        if not p.exists():
            print("缺少文件夹：", p)
            ok = False
    if not ok:
        return False

    for root in [TRAIN_DIR, VAL_DIR]:
        print(f"\n{root.name} 数据集：")
        for cls in ["no", "yes"]:
            cls_dir = root / cls
            if cls_dir.exists():
                files = [x for x in cls_dir.iterdir() if x.suffix.lower() in [".jpg", ".jpeg", ".png", ".bmp"]]
                print(f"  {cls}: {len(files)} 张")
            else:
                print(f"  缺少类别文件夹：{cls_dir}")
                ok = False
    for img in ["test_yes.jpg", "test_no.png"]:
        print(f"测试图片 {img}：", "存在" if (BASE_DIR / img).exists() else "缺失")
    return ok


def build_loaders(batch_size=16):
    if not TRAIN_DIR.exists() or not VAL_DIR.exists():
        raise FileNotFoundError("未找到 train 或 val 文件夹，请确认目录完整。")

    train_dataset = datasets.ImageFolder(str(TRAIN_DIR), transform=get_transform(train=True))
    val_dataset = datasets.ImageFolder(str(VAL_DIR), transform=get_transform(train=False))
    class_map = write_class_indices_from_dataset(train_dataset)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    return train_dataset, val_dataset, train_loader, val_loader, class_map


def load_model(require_weight=True):
    model = AlexNet(num_classes=2, init_weights=False).to(DEVICE)
    if require_weight:
        if not WEIGHT_PATH.exists():
            raise FileNotFoundError(f"没有找到权重文件：{WEIGHT_PATH}，请先训练模型。")
        try:
            state_dict = torch.load(WEIGHT_PATH, map_location=DEVICE, weights_only=True)
        except TypeError:
            state_dict = torch.load(WEIGHT_PATH, map_location=DEVICE)
        model.load_state_dict(state_dict)
    model.eval()
    return model


def evaluate_model(model, loader):
    idx_to_class = read_class_indices()
    model.eval()
    correct = 0
    total = 0
    rows = []
    class_total = {0: 0, 1: 0}
    class_correct = {0: 0, 1: 0}
    confusion = [[0, 0], [0, 0]]

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(DEVICE)
            labels = labels.to(DEVICE)
            outputs = model(images)
            probs = torch.softmax(outputs, dim=1)
            preds = torch.argmax(probs, dim=1)
            for label, pred, prob in zip(labels.cpu(), preds.cpu(), probs.cpu()):
                label_i = int(label.item())
                pred_i = int(pred.item())
                conf = float(prob[pred_i].item())
                ok = label_i == pred_i
                total += 1
                correct += int(ok)
                class_total[label_i] = class_total.get(label_i, 0) + 1
                class_correct[label_i] = class_correct.get(label_i, 0) + int(ok)
                confusion[label_i][pred_i] += 1
                rows.append({
                    "true": idx_to_class.get(label_i, str(label_i)),
                    "pred": idx_to_class.get(pred_i, str(pred_i)),
                    "confidence": conf,
                    "correct": ok,
                })
    acc = correct / total if total else 0.0
    return acc, correct, total, class_correct, class_total, confusion, rows


def train_alexnet():
    ensure_dirs()
    if not check_dataset():
        return
    raw_epochs = input("请输入训练轮数 epochs，直接回车默认 10：").strip()
    epochs = int(raw_epochs) if raw_epochs else 10
    raw_batch = input("请输入 batch_size，直接回车默认 16：").strip()
    batch_size = int(raw_batch) if raw_batch else 16

    train_dataset, val_dataset, train_loader, val_loader, class_map = build_loaders(batch_size=batch_size)
    model = AlexNet(num_classes=2, init_weights=True).to(DEVICE)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0002)

    print("\n========== 开始训练 AlexNet ==========")
    print("训练图片数量：", len(train_dataset))
    print("验证图片数量：", len(val_dataset))
    print("类别对应关系：", class_map)
    print("epochs：", epochs)
    print("batch_size：", batch_size)
    print("权重保存位置：", WEIGHT_PATH)

    best_acc = 0.0
    log_rows = []
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        bar = tqdm(train_loader, desc=f"epoch {epoch + 1}/{epochs}")
        for images, labels in bar:
            images = images.to(DEVICE)
            labels = labels.to(DEVICE)
            optimizer.zero_grad()
            outputs = model(images)
            loss = loss_fn(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            bar.set_postfix(loss=f"{loss.item():.3f}")
        train_loss = running_loss / max(1, len(train_loader))
        val_acc, correct, total, *_ = evaluate_model(model, val_loader)
        log_rows.append([epoch + 1, train_loss, val_acc, correct, total])
        print(f"第 {epoch + 1} 轮：loss={train_loss:.4f}, val_acc={val_acc * 100:.2f}% ({correct}/{total})")
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), WEIGHT_PATH)

    log_path = RESULT_DIR / "alexnet_training_log.csv"
    with open(log_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["epoch", "train_loss", "val_acc", "correct", "total"])
        writer.writerows(log_rows)
    print("\n训练完成。")
    print(f"最佳验证准确率：{best_acc * 100:.2f}%")
    print("训练日志已保存：", log_path)
    if HAS_PLOT:
        save_training_curve(log_rows)


def evaluate_val_set():
    ensure_dirs()
    _, val_dataset, _, val_loader, _ = build_loaders(batch_size=16)
    model = load_model(require_weight=True)
    acc, correct, total, class_correct, class_total, confusion, rows = evaluate_model(model, val_loader)
    idx_to_class = read_class_indices()

    print("\n========== AlexNet 完整验证集评估 ==========")
    print("权重：", WEIGHT_PATH)
    print("验证集路径：", VAL_DIR)
    print(f"整体准确率：{correct}/{total} = {acc * 100:.2f}%")
    for idx in sorted(class_total.keys()):
        name = idx_to_class.get(idx, str(idx))
        c = class_correct.get(idx, 0)
        n = class_total.get(idx, 0)
        rate = c / n if n else 0
        print(f"类别 {name}：{c}/{n} = {rate * 100:.2f}%")

    print("\n前 10 条预测明细：")
    for i, r in enumerate(rows[:10], 1):
        flag = "正确" if r["correct"] else "错误"
        print(f"{i:02d}. 真实={r['true']}, 预测={r['pred']}, 置信度={r['confidence']*100:.2f}%, {flag}")

    txt_path = RESULT_DIR / "alexnet_val_eval.txt"
    csv_path = RESULT_DIR / "alexnet_val_detail.csv"
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write("AlexNet 完整验证集评估结果\n")
        f.write(f"整体准确率：{correct}/{total} = {acc * 100:.2f}%\n")
        for idx in sorted(class_total.keys()):
            name = idx_to_class.get(idx, str(idx))
            c = class_correct.get(idx, 0)
            n = class_total.get(idx, 0)
            rate = c / n if n else 0
            f.write(f"类别 {name}：{c}/{n} = {rate * 100:.2f}%\n")
        f.write("\n混淆矩阵（行=真实类别，列=预测类别）\n")
        f.write(str(confusion))
    with open(csv_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["index", "true", "pred", "confidence", "correct"])
        for i, r in enumerate(rows, 1):
            writer.writerow([i, r["true"], r["pred"], f"{r['confidence']:.6f}", r["correct"]])
    print("\n评估结果已保存：", txt_path)
    print("逐张明细已保存：", csv_path)
    if HAS_PLOT:
        save_eval_charts(acc, class_correct, class_total, confusion)


def predict_one_image(image_name):
    ensure_dirs()
    image_path = BASE_DIR / image_name
    if not image_path.exists():
        print("没有找到图片：", image_path)
        return None
    model = load_model(require_weight=True)
    idx_to_class = read_class_indices()
    transform = get_transform(train=False)
    img = Image.open(image_path).convert("RGB")
    tensor = transform(img).unsqueeze(0).to(DEVICE)
    model.eval()
    with torch.no_grad():
        output = model(tensor).squeeze(0).cpu()
        prob = torch.softmax(output, dim=0)
        pred_idx = int(torch.argmax(prob).item())
        pred_name = idx_to_class.get(pred_idx, str(pred_idx))
    print("\n========== 单张图片预测 ==========")
    print("图片：", image_name)
    print("预测类别：", pred_name)
    print(f"置信度：{prob[pred_idx].item() * 100:.2f}%")
    print("各类别概率：")
    for i in range(len(prob)):
        print(f"  {idx_to_class.get(i, str(i))}: {prob[i].item() * 100:.2f}%")
    return image_name, pred_name, float(prob[pred_idx].item()), [float(x.item()) for x in prob]


def predict_two_images():
    results = []
    for img in ["test_yes.jpg", "test_no.png"]:
        r = predict_one_image(img)
        if r is not None:
            results.append(r)
    if results:
        csv_path = RESULT_DIR / "alexnet_test_predictions.csv"
        idx_to_class = read_class_indices()
        with open(csv_path, "w", encoding="utf-8", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["image", "pred", "confidence", "prob_no", "prob_yes"])
            for image_name, pred, conf, probs in results:
                writer.writerow([image_name, pred, f"{conf:.6f}", f"{probs[0]:.6f}", f"{probs[1]:.6f}"])
        print("\n测试图片预测结果已保存：", csv_path)


def save_training_curve(log_rows):
    if not log_rows:
        return
    epochs = [r[0] for r in log_rows]
    losses = [r[1] for r in log_rows]
    accs = [r[2] * 100 for r in log_rows]
    fig, ax1 = plt.subplots(figsize=(8, 5))
    ax1.plot(epochs, losses, marker="o", label="train loss")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Train Loss")
    ax1.grid(True, linestyle="--", alpha=0.35)
    ax2 = ax1.twinx()
    ax2.plot(epochs, accs, marker="s", label="val acc")
    ax2.set_ylabel("Val Accuracy (%)")
    plt.title("AlexNet Training Curve")
    fig.tight_layout()
    out = RESULT_DIR / "alexnet_training_curve.png"
    fig.savefig(out, dpi=200)
    plt.close(fig)
    print("训练曲线已保存：", out)


def save_eval_charts(acc, class_correct, class_total, confusion):
    idx_to_class = read_class_indices()
    names = [idx_to_class.get(0, "no"), idx_to_class.get(1, "yes")]
    values = []
    for idx in [0, 1]:
        n = class_total.get(idx, 0)
        values.append((class_correct.get(idx, 0) / n * 100) if n else 0)

    fig, ax = plt.subplots(figsize=(6, 4))
    bars = ax.bar(["overall"] + names, [acc * 100] + values)
    ax.set_ylim(0, 100)
    ax.set_ylabel("Accuracy (%)")
    ax.set_title("AlexNet Validation Accuracy")
    for b in bars:
        h = b.get_height()
        ax.text(b.get_x() + b.get_width() / 2, h + 1, f"{h:.2f}%", ha="center")
    fig.tight_layout()
    out1 = RESULT_DIR / "alexnet_val_accuracy.png"
    fig.savefig(out1, dpi=200)
    plt.close(fig)

    cm = np.array(confusion)
    fig, ax = plt.subplots(figsize=(4.5, 4))
    im = ax.imshow(cm)
    ax.set_title("AlexNet Confusion Matrix")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_xticks([0, 1]); ax.set_xticklabels(names)
    ax.set_yticks([0, 1]); ax.set_yticklabels(names)
    for i in range(2):
        for j in range(2):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center", fontsize=16)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    out2 = RESULT_DIR / "alexnet_confusion_matrix.png"
    fig.savefig(out2, dpi=200)
    plt.close(fig)
    print("验证集可视化已保存：", out1, out2)


def run_all():
    train_alexnet()
    evaluate_val_set()
    predict_two_images()


def print_menu():
    print("\n========== AlexNet 脑肿瘤 MRI 二分类系统 ==========")
    print("当前目录：", BASE_DIR)
    print("运行设备：", DEVICE)
    print("1. 检查数据集")
    print("2. 训练 AlexNet")
    print("3. 评估完整 val 数据集")
    print("4. 预测 test_yes.jpg 和 test_no.png")
    print("5. 自定义图片预测")
    print("6. 一键运行：训练 + 评估 + 测试图片预测")
    print("0. 退出")


def main():
    ensure_dirs()
    while True:
        print_menu()
        choice = input("请选择功能编号：").strip()
        try:
            if choice == "1":
                check_dataset()
            elif choice == "2":
                train_alexnet()
            elif choice == "3":
                evaluate_val_set()
            elif choice == "4":
                predict_two_images()
            elif choice == "5":
                name = input("请输入图片文件名，例如 test_yes.jpg：").strip()
                predict_one_image(name)
            elif choice == "6":
                run_all()
            elif choice == "0":
                print("程序结束。")
                break
            else:
                print("输入无效，请重新输入。")
        except Exception as e:
            print("运行出错：", e)


if __name__ == "__main__":
    main()
