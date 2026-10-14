import os
import csv
import time
import random
from pathlib import Path
from typing import List, Tuple, Optional

import numpy as np
from PIL import Image

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from tqdm import tqdm

try:
    import matplotlib.pyplot as plt
except Exception:
    plt = None


# =========================
# 1. 全局路径与默认参数
# =========================
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
TRAIN_DIR = DATA_DIR / "train"
VAL_DIR = DATA_DIR / "val"
PREDICT_INPUT_DIR = DATA_DIR / "predict"
MODEL_DIR = BASE_DIR / "model"
OUTPUT_DIR = BASE_DIR / "predict"

MODEL_DIR.mkdir(exist_ok=True)
OUTPUT_DIR.mkdir(exist_ok=True)

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

IMAGE_SIZE = 512          # U-Net 下采样 4 次，建议使用 16 的倍数
BATCH_SIZE = 2            # CPU 环境建议 1 或 2
EPOCHS = 5                # 课程演示可先用 5；想提高效果可改大
LR = 1e-4
THRESHOLD = 0.5
SEED = 42


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# =========================
# 2. 数据集读取
# =========================
def list_images(folder: Path) -> List[Path]:
    exts = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
    if not folder.exists():
        return []
    return sorted([p for p in folder.iterdir() if p.suffix.lower() in exts])


def make_pairs(root: Path) -> List[Tuple[Path, Path]]:
    """读取 Data 与 Ground 文件夹，按同名文件配对。"""
    data_dir = root / "Data"
    mask_dir = root / "Ground"
    images = list_images(data_dir)
    pairs = []
    missing = []

    for img_path in images:
        mask_path = mask_dir / img_path.name
        if mask_path.exists():
            pairs.append((img_path, mask_path))
        else:
            missing.append(img_path.name)

    if missing:
        print("以下图像没有找到同名标签文件，已跳过：")
        for name in missing[:10]:
            print("  ", name)
        if len(missing) > 10:
            print(f"  ……共 {len(missing)} 个缺失标签")
    return pairs


class LiverSegDataset(Dataset):
    def __init__(self, root: Path, image_size: int = 512, augment: bool = False):
        self.root = root
        self.pairs = make_pairs(root)
        self.image_size = image_size
        self.augment = augment
        self.img_tf = transforms.Compose([
            transforms.Resize((image_size, image_size), interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.ToTensor(),
        ])
        self.mask_tf = transforms.Compose([
            transforms.Resize((image_size, image_size), interpolation=transforms.InterpolationMode.NEAREST),
            transforms.ToTensor(),
        ])

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, index):
        img_path, mask_path = self.pairs[index]
        img = Image.open(img_path).convert("L")
        mask = Image.open(mask_path).convert("L")

        # 简单同步增强：水平翻转。医学分割演示中保持轻量，不做复杂形变。
        if self.augment and random.random() < 0.5:
            img = img.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
            mask = mask.transpose(Image.Transpose.FLIP_LEFT_RIGHT)

        img = self.img_tf(img)
        mask = self.mask_tf(mask)
        mask = (mask > 0.5).float()
        return img, mask, img_path.name


class PredictDataset(Dataset):
    def __init__(self, root: Path, image_size: int = 512):
        self.images = list_images(root)
        self.image_size = image_size
        self.tf = transforms.Compose([
            transforms.Resize((image_size, image_size), interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.ToTensor(),
        ])

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        img_path = self.images[index]
        img = Image.open(img_path).convert("L")
        original_size = img.size  # (W, H)
        x = self.tf(img)
        return x, img_path.name, original_size


# =========================
# 3. U-Net 模型定义
# =========================
class DoubleConv(nn.Module):
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.net(x)


class UNet(nn.Module):
    def __init__(self, in_ch: int = 1, out_ch: int = 1, base_ch: int = 32, dropout: float = 0.1):
        super().__init__()
        # 这里使用 base_ch=32，比原始 64 更轻，适合 CPU；结构仍是标准 U-Net 编码-解码。
        self.conv1 = DoubleConv(in_ch, base_ch)
        self.pool1 = nn.MaxPool2d(2)
        self.conv2 = DoubleConv(base_ch, base_ch * 2)
        self.pool2 = nn.MaxPool2d(2)
        self.conv3 = DoubleConv(base_ch * 2, base_ch * 4)
        self.pool3 = nn.MaxPool2d(2)
        self.conv4 = DoubleConv(base_ch * 4, base_ch * 8)
        self.pool4 = nn.MaxPool2d(2)
        self.conv5 = DoubleConv(base_ch * 8, base_ch * 16)

        self.up6 = nn.ConvTranspose2d(base_ch * 16, base_ch * 8, kernel_size=2, stride=2)
        self.conv6 = DoubleConv(base_ch * 16, base_ch * 8)
        self.up7 = nn.ConvTranspose2d(base_ch * 8, base_ch * 4, kernel_size=2, stride=2)
        self.conv7 = DoubleConv(base_ch * 8, base_ch * 4)
        self.up8 = nn.ConvTranspose2d(base_ch * 4, base_ch * 2, kernel_size=2, stride=2)
        self.conv8 = DoubleConv(base_ch * 4, base_ch * 2)
        self.up9 = nn.ConvTranspose2d(base_ch * 2, base_ch, kernel_size=2, stride=2)
        self.conv9 = DoubleConv(base_ch * 2, base_ch)
        self.conv10 = nn.Conv2d(base_ch, out_ch, kernel_size=1)
        self.dropout = nn.Dropout2d(p=dropout)

    def forward(self, x):
        c1 = self.conv1(x)
        p1 = self.dropout(self.pool1(c1))
        c2 = self.conv2(p1)
        p2 = self.dropout(self.pool2(c2))
        c3 = self.conv3(p2)
        p3 = self.dropout(self.pool3(c3))
        c4 = self.conv4(p3)
        p4 = self.dropout(self.pool4(c4))
        c5 = self.conv5(p4)

        u6 = self.up6(c5)
        c6 = self.conv6(torch.cat([u6, c4], dim=1))
        u7 = self.up7(c6)
        c7 = self.conv7(torch.cat([u7, c3], dim=1))
        u8 = self.up8(c7)
        c8 = self.conv8(torch.cat([u8, c2], dim=1))
        u9 = self.up9(c8)
        c9 = self.conv9(torch.cat([u9, c1], dim=1))
        return self.conv10(c9)  # 输出 logits；训练时配合 BCEWithLogitsLoss


# =========================
# 4. 指标、保存与可视化
# =========================
def dice_score_from_logits(logits: torch.Tensor, masks: torch.Tensor, threshold: float = 0.5, eps: float = 1e-7) -> float:
    probs = torch.sigmoid(logits)
    preds = (probs > threshold).float()
    masks = (masks > 0.5).float()
    intersection = (preds * masks).sum(dim=(1, 2, 3))
    union = preds.sum(dim=(1, 2, 3)) + masks.sum(dim=(1, 2, 3))
    dice = (2 * intersection + eps) / (union + eps)
    return dice.mean().item()


def tensor_to_uint8_img(tensor: torch.Tensor) -> np.ndarray:
    arr = tensor.detach().cpu().numpy()
    if arr.ndim == 3:
        arr = arr[0]
    arr = np.clip(arr * 255, 0, 255).astype(np.uint8)
    return arr


def save_mask(array_01: np.ndarray, save_path: Path):
    arr = (array_01.astype(np.float32) * 255).clip(0, 255).astype(np.uint8)
    Image.fromarray(arr).save(save_path)


def save_triplet(image_arr: np.ndarray, gt_arr: Optional[np.ndarray], pred_arr: np.ndarray, save_path: Path, title: str = ""):
    if plt is None:
        return
    if gt_arr is None:
        fig, axes = plt.subplots(1, 2, figsize=(8, 4))
        axes[0].imshow(image_arr, cmap="gray")
        axes[0].set_title("Original")
        axes[1].imshow(pred_arr, cmap="gray")
        axes[1].set_title("Prediction")
    else:
        fig, axes = plt.subplots(1, 3, figsize=(12, 4))
        axes[0].imshow(image_arr, cmap="gray")
        axes[0].set_title("Original")
        axes[1].imshow(gt_arr, cmap="gray")
        axes[1].set_title("Ground Truth")
        axes[2].imshow(pred_arr, cmap="gray")
        axes[2].set_title("Prediction")
    for ax in axes:
        ax.axis("off")
    if title:
        fig.suptitle(title)
    plt.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)


def plot_loss_curve(train_losses: List[float], val_losses: List[float], save_path: Path):
    if plt is None or not train_losses:
        return
    plt.figure(figsize=(8, 5))
    plt.plot(range(1, len(train_losses) + 1), train_losses, marker="o", label="train loss")
    if val_losses:
        plt.plot(range(1, len(val_losses) + 1), val_losses, marker="o", label="val loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("U-Net Training Curve")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()


# =========================
# 5. 核心功能：检查、训练、预测、评估
# =========================
def check_dataset():
    print("\n========== 数据集检查 ==========")
    print("项目目录：", BASE_DIR)
    print("运行设备：", DEVICE)
    for name, folder in [("训练集", TRAIN_DIR), ("验证集", VAL_DIR)]:
        data_dir = folder / "Data"
        mask_dir = folder / "Ground"
        imgs = list_images(data_dir)
        masks = list_images(mask_dir)
        pairs = make_pairs(folder)
        print(f"\n{name}：{folder}")
        print("  Data 图像数量：", len(imgs))
        print("  Ground 标签数量：", len(masks))
        print("  可配对样本数量：", len(pairs))
        if pairs:
            print("  示例：", pairs[0][0].name)
    pred_imgs = list_images(PREDICT_INPUT_DIR)
    print(f"\n待预测文件夹：{PREDICT_INPUT_DIR}")
    print("  待预测图像数量：", len(pred_imgs))


def evaluate(model: nn.Module, loader: DataLoader, criterion: nn.Module) -> Tuple[float, float]:
    model.eval()
    total_loss = 0.0
    total_dice = 0.0
    steps = 0
    with torch.no_grad():
        for images, masks, _ in loader:
            images = images.to(DEVICE)
            masks = masks.to(DEVICE)
            logits = model(images)
            loss = criterion(logits, masks)
            dice = dice_score_from_logits(logits, masks, THRESHOLD)
            total_loss += loss.item()
            total_dice += dice
            steps += 1
    if steps == 0:
        return 0.0, 0.0
    return total_loss / steps, total_dice / steps


def train_unet():
    set_seed(SEED)
    train_dataset = LiverSegDataset(TRAIN_DIR, image_size=IMAGE_SIZE, augment=True)
    val_dataset = LiverSegDataset(VAL_DIR, image_size=IMAGE_SIZE, augment=False)
    if len(train_dataset) == 0:
        print("没有可用训练样本，请检查 data/train/Data 与 data/train/Ground。")
        return
    if len(val_dataset) == 0:
        print("没有可用验证样本，请检查 data/val/Data 与 data/val/Ground。")
        return

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    model = UNet(in_ch=1, out_ch=1, base_ch=32, dropout=0.1).to(DEVICE)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    best_dice = -1.0
    train_losses, val_losses, val_dices = [], [], []
    best_path = MODEL_DIR / "best_unet.pth"
    last_path = MODEL_DIR / "last_unet.pth"

    print("\n========== 开始训练 U-Net ==========")
    print("训练样本：", len(train_dataset))
    print("验证样本：", len(val_dataset))
    print("image_size：", IMAGE_SIZE)
    print("batch_size：", BATCH_SIZE)
    print("epochs：", EPOCHS)
    print("device：", DEVICE)

    for epoch in range(1, EPOCHS + 1):
        model.train()
        total_loss = 0.0
        steps = 0
        bar = tqdm(train_loader, desc=f"epoch {epoch}/{EPOCHS}")
        for images, masks, _ in bar:
            images = images.to(DEVICE)
            masks = masks.to(DEVICE)
            optimizer.zero_grad()
            logits = model(images)
            loss = criterion(logits, masks)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            steps += 1
            bar.set_postfix(loss=f"{loss.item():.4f}")

        train_loss = total_loss / max(1, steps)
        val_loss, val_dice = evaluate(model, val_loader, criterion)
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        val_dices.append(val_dice)

        print(f"第 {epoch} 轮：train_loss={train_loss:.4f}, val_loss={val_loss:.4f}, val_dice={val_dice:.4f}")
        if val_dice > best_dice:
            best_dice = val_dice
            torch.save(model.state_dict(), best_path)
            print("  已保存最佳模型：", best_path.name)

    torch.save(model.state_dict(), last_path)
    plot_loss_curve(train_losses, val_losses, OUTPUT_DIR / "loss_curve.png")

    # 保存训练日志
    log_path = OUTPUT_DIR / "train_log.csv"
    with open(log_path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f)
        writer.writerow(["epoch", "train_loss", "val_loss", "val_dice"])
        for i, (tl, vl, vd) in enumerate(zip(train_losses, val_losses, val_dices), 1):
            writer.writerow([i, f"{tl:.6f}", f"{vl:.6f}", f"{vd:.6f}"])

    print("\n训练完成")
    print("最佳 Dice：", f"{best_dice:.4f}")
    print("最佳权重：", best_path)
    print("最后一轮权重：", last_path)
    print("训练日志：", log_path)
    print("损失曲线：", OUTPUT_DIR / "loss_curve.png")


def load_trained_model(weight_name: str = "best_unet.pth") -> nn.Module:
    weight_path = MODEL_DIR / weight_name
    if not weight_path.exists():
        raise FileNotFoundError(f"没有找到权重文件：{weight_path}，请先训练或检查 model 文件夹。")
    model = UNet(in_ch=1, out_ch=1, base_ch=32, dropout=0.0).to(DEVICE)
    state = torch.load(weight_path, map_location=DEVICE)
    # 兼容原 notebook 中 DataParallel 保存的 module. 前缀
    if any(k.startswith("module.") for k in state.keys()):
        state = {k.replace("module.", ""): v for k, v in state.items()}
    model.load_state_dict(state)
    model.eval()
    return model


def predict_val(weight_name: str = "best_unet.pth"):
    val_dataset = LiverSegDataset(VAL_DIR, image_size=IMAGE_SIZE, augment=False)
    if len(val_dataset) == 0:
        print("没有可用验证样本，请检查 data/val/Data 与 data/val/Ground。")
        return
    loader = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=0)
    model = load_trained_model(weight_name)

    save_dir = OUTPUT_DIR / "val_results"
    save_dir.mkdir(exist_ok=True)
    dice_rows = []

    print("\n========== 预测验证集并保存结果 ==========")
    with torch.no_grad():
        for idx, (image, mask, name_list) in enumerate(tqdm(loader)):
            name = name_list[0]
            image = image.to(DEVICE)
            mask = mask.to(DEVICE)
            logits = model(image)
            prob = torch.sigmoid(logits)
            pred = (prob > THRESHOLD).float()

            dice = dice_score_from_logits(logits, mask, THRESHOLD)
            dice_rows.append([idx, name, f"{dice:.6f}"])

            img_arr = tensor_to_uint8_img(image[0])
            gt_arr = tensor_to_uint8_img(mask[0])
            pred_arr = tensor_to_uint8_img(pred[0])

            Image.fromarray(img_arr).save(save_dir / f"{idx:03d}_image_{name}.png")
            Image.fromarray(gt_arr).save(save_dir / f"{idx:03d}_gt_{name}.png")
            Image.fromarray(pred_arr).save(save_dir / f"{idx:03d}_pred_{name}.png")
            save_triplet(img_arr, gt_arr, pred_arr, save_dir / f"{idx:03d}_compare_{name}.png", title=f"Dice={dice:.4f}")

    mean_dice = float(np.mean([float(r[2]) for r in dice_rows])) if dice_rows else 0.0
    csv_path = save_dir / "dice_results.csv"
    with open(csv_path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f)
        writer.writerow(["index", "filename", "dice"])
        writer.writerows(dice_rows)
        writer.writerow([])
        writer.writerow(["mean_dice", "", f"{mean_dice:.6f}"])

    print("预测结果保存至：", save_dir)
    print("Dice 结果文件：", csv_path)
    print("平均 Dice：", f"{mean_dice:.4f}")


def predict_unlabeled(weight_name: str = "best_unet.pth"):
    pred_dataset = PredictDataset(PREDICT_INPUT_DIR, image_size=IMAGE_SIZE)
    if len(pred_dataset) == 0:
        print("data/predict 中没有待预测图像。")
        return
    loader = DataLoader(pred_dataset, batch_size=1, shuffle=False, num_workers=0)
    model = load_trained_model(weight_name)
    save_dir = OUTPUT_DIR / "predict_results"
    save_dir.mkdir(exist_ok=True)

    print("\n========== 预测 data/predict 文件夹 ==========")
    with torch.no_grad():
        for idx, (image, name_list, original_size) in enumerate(tqdm(loader)):
            name = name_list[0]
            image = image.to(DEVICE)
            logits = model(image)
            pred = (torch.sigmoid(logits) > THRESHOLD).float()
            img_arr = tensor_to_uint8_img(image[0])
            pred_arr = tensor_to_uint8_img(pred[0])
            Image.fromarray(pred_arr).save(save_dir / f"{idx:03d}_pred_{name}.png")
            save_triplet(img_arr, None, pred_arr, save_dir / f"{idx:03d}_compare_{name}.png", title=name)
    print("待预测图像结果保存至：", save_dir)


def run_all():
    train_unet()
    predict_val("best_unet.pth")
    predict_unlabeled("best_unet.pth")


# =========================
# 6. 菜单入口
# =========================
def print_menu():
    print("\n========== U-Net 肝脏医学图像分割系统 ==========")
    print("当前目录：", BASE_DIR)
    print("运行设备：", DEVICE)
    print("1. 检查数据集")
    print("2. 训练 U-Net")
    print("3. 使用 best_unet.pth 预测验证集并计算 Dice")
    print("4. 使用 best_unet.pth 预测 data/predict 文件夹")
    print("5. 一键运行：训练 + 验证集预测评估 + 待预测图像输出")
    print("6. 使用指定权重预测验证集")
    print("0. 退出")


def main():
    set_seed(SEED)
    while True:
        print_menu()
        choice = input("请选择功能编号：").strip()
        try:
            if choice == "1":
                check_dataset()
            elif choice == "2":
                train_unet()
            elif choice == "3":
                predict_val("best_unet.pth")
            elif choice == "4":
                predict_unlabeled("best_unet.pth")
            elif choice == "5":
                run_all()
            elif choice == "6":
                weight = input("请输入 model 文件夹中的权重文件名，例如 best_unet.pth：").strip()
                predict_val(weight)
            elif choice == "0":
                print("程序结束。")
                break
            else:
                print("输入无效，请重新输入。")
        except Exception as e:
            print("运行出错：", e)


if __name__ == "__main__":
    main()
