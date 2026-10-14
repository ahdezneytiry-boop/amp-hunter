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
import torch.nn.utils.prune as prune
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from tqdm import tqdm

try:
    import matplotlib.pyplot as plt
except Exception:
    plt = None


# =========================================================
# U-Net 肝脏分割 CPU 极速版
# 适用目录：
# P2_liver_seg0919/
# ├─ data/
# │  ├─ train/Data, train/Ground
# │  ├─ val/Data, val/Ground
# │  └─ predict
# ├─ model/
# └─ predict/
# =========================================================

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

# =========================
# CPU 极速参数
# =========================
IMAGE_SIZE = 256          # 原来 512 太慢，CPU 建议 256；想更快可改 192
BATCH_SIZE = 1            # CPU 最稳；显存/内存够可改 2
EPOCHS = 3                # 演示版先 3 轮；需要效果再改 5~10
LR = 2e-4
BASE_CH = 8               # 原来 32，这里降到 8，是核心提速点
THRESHOLD = 0.5
SEED = 42
EARLY_STOP_PATIENCE = 2


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# =========================
# 数据读取
# =========================
def list_images(folder: Path) -> List[Path]:
    exts = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
    if not folder.exists():
        return []
    return sorted([p for p in folder.iterdir() if p.suffix.lower() in exts])


def make_pairs(root: Path) -> List[Tuple[Path, Path]]:
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
        print("以下图像没有找到同名标签，已跳过：")
        for name in missing[:10]:
            print("  ", name)
        if len(missing) > 10:
            print(f"  ……共 {len(missing)} 个缺失标签")
    return pairs


class LiverSegDataset(Dataset):
    def __init__(self, root: Path, image_size: int = IMAGE_SIZE, augment: bool = False):
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

        if self.augment:
            if random.random() < 0.5:
                img = img.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
                mask = mask.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
            if random.random() < 0.3:
                img = img.transpose(Image.Transpose.FLIP_TOP_BOTTOM)
                mask = mask.transpose(Image.Transpose.FLIP_TOP_BOTTOM)

        img = self.img_tf(img)
        mask = self.mask_tf(mask)
        mask = (mask > 0.5).float()
        return img, mask, img_path.name


class PredictDataset(Dataset):
    def __init__(self, root: Path, image_size: int = IMAGE_SIZE):
        self.images = list_images(root)
        self.tf = transforms.Compose([
            transforms.Resize((image_size, image_size), interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.ToTensor(),
        ])

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        img_path = self.images[index]
        img = Image.open(img_path).convert("L")
        x = self.tf(img)
        return x, img_path.name


# =========================
# 轻量 U-Net
# =========================
class DoubleConvLite(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.net(x)


class FastUNet(nn.Module):
    """
    CPU 轻量版 U-Net：
    1. base_ch 默认 8，远小于原来的 32/64；
    2. 只下采样 3 次，减少计算；
    3. 上采样用 bilinear + conv，避免 ConvTranspose2d 在 CPU 上过慢；
    4. 输出 logits，训练时配合 BCEWithLogitsLoss。
    """
    def __init__(self, in_ch=1, out_ch=1, base_ch=BASE_CH):
        super().__init__()
        self.enc1 = DoubleConvLite(in_ch, base_ch)
        self.pool1 = nn.MaxPool2d(2)
        self.enc2 = DoubleConvLite(base_ch, base_ch * 2)
        self.pool2 = nn.MaxPool2d(2)
        self.enc3 = DoubleConvLite(base_ch * 2, base_ch * 4)
        self.pool3 = nn.MaxPool2d(2)

        self.mid = DoubleConvLite(base_ch * 4, base_ch * 8)

        self.up3 = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)
        self.dec3 = DoubleConvLite(base_ch * 8 + base_ch * 4, base_ch * 4)

        self.up2 = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)
        self.dec2 = DoubleConvLite(base_ch * 4 + base_ch * 2, base_ch * 2)

        self.up1 = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)
        self.dec1 = DoubleConvLite(base_ch * 2 + base_ch, base_ch)

        self.out = nn.Conv2d(base_ch, out_ch, 1)

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool1(e1))
        e3 = self.enc3(self.pool2(e2))

        m = self.mid(self.pool3(e3))

        d3 = self.up3(m)
        d3 = self.dec3(torch.cat([d3, e3], dim=1))

        d2 = self.up2(d3)
        d2 = self.dec2(torch.cat([d2, e2], dim=1))

        d1 = self.up1(d2)
        d1 = self.dec1(torch.cat([d1, e1], dim=1))

        return self.out(d1)


# =========================
# 指标和工具
# =========================
def dice_score_from_logits(logits, masks, threshold=THRESHOLD, eps=1e-7):
    probs = torch.sigmoid(logits)
    preds = (probs > threshold).float()
    masks = (masks > 0.5).float()
    intersection = (preds * masks).sum(dim=(1, 2, 3))
    union = preds.sum(dim=(1, 2, 3)) + masks.sum(dim=(1, 2, 3))
    dice = (2 * intersection + eps) / (union + eps)
    return dice.mean().item()


def tensor_to_uint8(tensor):
    arr = tensor.detach().cpu().numpy()
    if arr.ndim == 3:
        arr = arr[0]
    return np.clip(arr * 255, 0, 255).astype(np.uint8)


def save_triplet(image_arr, gt_arr, pred_arr, save_path, title=""):
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
    fig.savefig(save_path, dpi=120)
    plt.close(fig)


def plot_loss_curve(train_losses, val_losses, val_dices, save_path):
    if plt is None:
        return
    fig, ax1 = plt.subplots(figsize=(8, 5))
    ax1.plot(range(1, len(train_losses) + 1), train_losses, marker="o", label="train loss")
    ax1.plot(range(1, len(val_losses) + 1), val_losses, marker="o", label="val loss")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.grid(alpha=0.3)
    ax2 = ax1.twinx()
    ax2.plot(range(1, len(val_dices) + 1), val_dices, marker="s", label="val dice")
    ax2.set_ylabel("Dice")
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2)
    plt.title("Fast U-Net Training Curve")
    plt.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)


# =========================
# 剪枝与轻量化
# =========================
def apply_unstructured_pruning(model, amount=0.30):
    """
    非结构化 L1 剪枝：把卷积权重中较小的部分置零。
    注意：普通 PyTorch CPU 上，这种 mask 剪枝不一定明显加速；
    真正提速主要来自本代码的 base_ch=8、image_size=256 和浅层结构。
    """
    for module in model.modules():
        if isinstance(module, nn.Conv2d):
            prune.l1_unstructured(module, name="weight", amount=amount)
            prune.remove(module, "weight")
    return model


def try_dynamic_quantize(model):
    """
    PyTorch 动态量化主要对 Linear/RNN 有明显作用。
    本模型几乎全是 Conv2d，因此该函数只做兼容性尝试，不承诺 INT8 卷积加速。
    真正需要 Conv INT8 要走静态量化校准，U-Net 的上采样、拼接、医学分割后处理会复杂很多。
    """
    try:
        q_model = torch.quantization.quantize_dynamic(model, {nn.Linear}, dtype=torch.qint8)
        return q_model
    except Exception as e:
        print("动态量化失败，继续使用 FP32。原因：", e)
        return model


# =========================
# 核心功能
# =========================
def check_dataset():
    print("\n========== 数据集检查 ==========")
    print("当前目录：", BASE_DIR)
    print("运行设备：", DEVICE)
    print("image_size：", IMAGE_SIZE)
    print("base_ch：", BASE_CH)
    for name, folder in [("训练集", TRAIN_DIR), ("验证集", VAL_DIR)]:
        pairs = make_pairs(folder)
        print(f"\n{name}：{folder}")
        print("  Data 数量：", len(list_images(folder / "Data")))
        print("  Ground 数量：", len(list_images(folder / "Ground")))
        print("  可配对数量：", len(pairs))
        if pairs:
            print("  示例：", pairs[0][0].name)
    print("\n待预测图片数量：", len(list_images(PREDICT_INPUT_DIR)))


def evaluate(model, loader, criterion):
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
            dice = dice_score_from_logits(logits, masks)
            total_loss += loss.item()
            total_dice += dice
            steps += 1
    if steps == 0:
        return 0.0, 0.0
    return total_loss / steps, total_dice / steps


def train_fast_unet():
    set_seed(SEED)
    train_dataset = LiverSegDataset(TRAIN_DIR, IMAGE_SIZE, augment=True)
    val_dataset = LiverSegDataset(VAL_DIR, IMAGE_SIZE, augment=False)
    if len(train_dataset) == 0:
        print("没有可用训练样本。")
        return
    if len(val_dataset) == 0:
        print("没有可用验证样本。")
        return

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    model = FastUNet().to(DEVICE)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    best_dice = -1.0
    patience_count = 0
    train_losses, val_losses, val_dices = [], [], []

    best_path = MODEL_DIR / "fast_unet_best.pth"
    last_path = MODEL_DIR / "fast_unet_last.pth"

    print("\n========== 开始训练 Fast U-Net ==========")
    print("训练样本：", len(train_dataset))
    print("验证样本：", len(val_dataset))
    print("image_size：", IMAGE_SIZE)
    print("base_ch：", BASE_CH)
    print("batch_size：", BATCH_SIZE)
    print("epochs：", EPOCHS)
    print("device：", DEVICE)

    start_time = time.time()
    for epoch in range(1, EPOCHS + 1):
        model.train()
        running_loss = 0.0
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
            running_loss += loss.item()
            steps += 1
            bar.set_postfix(loss=f"{loss.item():.4f}")

        train_loss = running_loss / max(1, steps)
        val_loss, val_dice = evaluate(model, val_loader, criterion)
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        val_dices.append(val_dice)

        print(f"第 {epoch} 轮：train_loss={train_loss:.4f}, val_loss={val_loss:.4f}, val_dice={val_dice:.4f}")

        if val_dice > best_dice:
            best_dice = val_dice
            patience_count = 0
            torch.save(model.state_dict(), best_path)
            print("  已保存最佳模型：", best_path.name)
        else:
            patience_count += 1
            if patience_count >= EARLY_STOP_PATIENCE:
                print("  验证 Dice 连续未提升，提前停止训练。")
                break

    torch.save(model.state_dict(), last_path)
    plot_loss_curve(train_losses, val_losses, val_dices, OUTPUT_DIR / "fast_loss_curve.png")

    log_path = OUTPUT_DIR / "fast_train_log.csv"
    with open(log_path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f)
        writer.writerow(["epoch", "train_loss", "val_loss", "val_dice"])
        for i, (tl, vl, vd) in enumerate(zip(train_losses, val_losses, val_dices), 1):
            writer.writerow([i, f"{tl:.6f}", f"{vl:.6f}", f"{vd:.6f}"])

    print("\n训练完成")
    print("耗时：", f"{time.time() - start_time:.1f} 秒")
    print("最佳 Dice：", f"{best_dice:.4f}")
    print("最佳权重：", best_path)
    print("日志：", log_path)


def load_model(weight_name="fast_unet_best.pth", do_prune=False, do_quant=False):
    weight_path = MODEL_DIR / weight_name
    if not weight_path.exists():
        raise FileNotFoundError(f"没有找到权重文件：{weight_path}")
    model = FastUNet().to(DEVICE)
    state = torch.load(weight_path, map_location=DEVICE)
    if any(k.startswith("module.") for k in state.keys()):
        state = {k.replace("module.", ""): v for k, v in state.items()}
    model.load_state_dict(state)

    if do_prune:
        model = apply_unstructured_pruning(model, amount=0.30)
    if do_quant:
        model = try_dynamic_quantize(model)

    model.eval()
    return model


def predict_val(weight_name="fast_unet_best.pth", do_prune=False, do_quant=False):
    dataset = LiverSegDataset(VAL_DIR, IMAGE_SIZE, augment=False)
    if len(dataset) == 0:
        print("没有可用验证样本。")
        return

    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)
    model = load_model(weight_name, do_prune=do_prune, do_quant=do_quant)

    tag = "fast"
    if do_prune:
        tag += "_pruned"
    if do_quant:
        tag += "_quant"
    save_dir = OUTPUT_DIR / f"{tag}_val_results"
    save_dir.mkdir(exist_ok=True)

    rows = []
    infer_times = []

    print("\n========== 预测验证集 ==========")
    with torch.no_grad():
        for idx, (image, mask, names) in enumerate(tqdm(loader)):
            name = names[0]
            image = image.to(DEVICE)
            mask = mask.to(DEVICE)

            t0 = time.time()
            logits = model(image)
            infer_times.append(time.time() - t0)

            pred = (torch.sigmoid(logits) > THRESHOLD).float()
            dice = dice_score_from_logits(logits, mask)
            rows.append([idx, name, f"{dice:.6f}"])

            img_arr = tensor_to_uint8(image[0])
            gt_arr = tensor_to_uint8(mask[0])
            pred_arr = tensor_to_uint8(pred[0])

            Image.fromarray(pred_arr).save(save_dir / f"{idx:03d}_pred_{name}.png")
            save_triplet(img_arr, gt_arr, pred_arr, save_dir / f"{idx:03d}_compare_{name}.png", title=f"Dice={dice:.4f}")

    mean_dice = float(np.mean([float(r[2]) for r in rows])) if rows else 0.0
    avg_ms = float(np.mean(infer_times) * 1000) if infer_times else 0.0

    csv_path = save_dir / "dice_results.csv"
    with open(csv_path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f)
        writer.writerow(["index", "filename", "dice"])
        writer.writerows(rows)
        writer.writerow([])
        writer.writerow(["mean_dice", "", f"{mean_dice:.6f}"])
        writer.writerow(["avg_inference_ms", "", f"{avg_ms:.3f}"])

    print("结果目录：", save_dir)
    print("平均 Dice：", f"{mean_dice:.4f}")
    print("平均单张推理时间：", f"{avg_ms:.2f} ms")
    print("结果文件：", csv_path)


def save_pruned_weight(weight_name="fast_unet_best.pth"):
    model = load_model(weight_name, do_prune=True, do_quant=False)
    save_path = MODEL_DIR / "fast_unet_pruned_30.pth"
    torch.save(model.state_dict(), save_path)
    print("剪枝权重已保存：", save_path)
    print("说明：这是非结构化剪枝权重，文件更稀疏，但普通 PyTorch CPU 推理不一定明显变快。")


def predict_unlabeled(weight_name="fast_unet_best.pth"):
    dataset = PredictDataset(PREDICT_INPUT_DIR, IMAGE_SIZE)
    if len(dataset) == 0:
        print("data/predict 中没有图片。")
        return

    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)
    model = load_model(weight_name)
    save_dir = OUTPUT_DIR / "fast_predict_results"
    save_dir.mkdir(exist_ok=True)

    with torch.no_grad():
        for idx, (image, names) in enumerate(tqdm(loader)):
            name = names[0]
            image = image.to(DEVICE)
            logits = model(image)
            pred = (torch.sigmoid(logits) > THRESHOLD).float()
            img_arr = tensor_to_uint8(image[0])
            pred_arr = tensor_to_uint8(pred[0])
            Image.fromarray(pred_arr).save(save_dir / f"{idx:03d}_pred_{name}.png")
            save_triplet(img_arr, None, pred_arr, save_dir / f"{idx:03d}_compare_{name}.png", title=name)

    print("预测结果保存至：", save_dir)


def print_menu():
    print("\n========== U-Net 肝脏分割 CPU 极速版 ==========")
    print("当前目录：", BASE_DIR)
    print("运行设备：", DEVICE)
    print("当前参数：image_size =", IMAGE_SIZE, ", base_ch =", BASE_CH, ", batch_size =", BATCH_SIZE)
    print("1. 检查数据集")
    print("2. 训练 Fast U-Net")
    print("3. 预测验证集并计算 Dice")
    print("4. 剪枝后预测验证集并计算 Dice")
    print("5. 保存 30% 剪枝权重")
    print("6. 尝试动态量化后预测验证集")
    print("7. 预测 data/predict 文件夹")
    print("8. 一键运行：训练 + 验证集预测")
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
                train_fast_unet()
            elif choice == "3":
                predict_val("fast_unet_best.pth")
            elif choice == "4":
                predict_val("fast_unet_best.pth", do_prune=True)
            elif choice == "5":
                save_pruned_weight("fast_unet_best.pth")
            elif choice == "6":
                predict_val("fast_unet_best.pth", do_quant=True)
            elif choice == "7":
                predict_unlabeled("fast_unet_best.pth")
            elif choice == "8":
                train_fast_unet()
                predict_val("fast_unet_best.pth")
            elif choice == "0":
                print("程序结束。")
                break
            else:
                print("输入无效，请重新输入。")
        except Exception as e:
            print("运行出错：", e)


if __name__ == "__main__":
    main()
