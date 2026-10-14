import json
import sys
from pathlib import Path

import torch
from PIL import Image
from torchvision import transforms

from model import LeNet, AlexNet, resnet34
from model_cpu_light import TinyCNN

ROOT = Path(__file__).resolve().parent
CLASS_JSON = ROOT / "class_indices.json"
DEFAULT_IMAGE = ROOT / "test_no.png"


def build_model(model_name):
    model_name = model_name.lower()
    if model_name == "tinycnn":
        model = TinyCNN(num_classes=2)
        weight = ROOT / "TinyCNN_cpu.pth"
        img_size = 128
        norm = ((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    elif model_name == "lenet":
        model = LeNet(num_classes=2)
        weight = ROOT / "LeNet.pth"
        img_size = 224
        norm = ((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    elif model_name == "alexnet":
        model = AlexNet(num_classes=2)
        weight = ROOT / "AlexNet.pth"
        img_size = 224
        norm = ((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    elif model_name == "resnet34":
        model = resnet34(num_classes=2)
        weight = ROOT / "ResNet34.pth"
        img_size = 224
        norm = ((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    else:
        raise ValueError("模型名称只能是 tinycnn、lenet、alexnet、resnet34")
    return model, weight, img_size, norm


def main():
    model_name = sys.argv[1] if len(sys.argv) >= 2 else "lenet"
    img_path = Path(sys.argv[2]) if len(sys.argv) >= 3 else DEFAULT_IMAGE
    if not img_path.is_absolute():
        img_path = ROOT / img_path

    if not img_path.exists():
        raise FileNotFoundError(f"图片不存在：{img_path}")
    if not CLASS_JSON.exists():
        raise FileNotFoundError(f"类别文件不存在：{CLASS_JSON}")

    model, weight_path, img_size, norm = build_model(model_name)
    if not weight_path.exists():
        raise FileNotFoundError(f"权重文件不存在：{weight_path}。如果使用 tinycnn，请先运行 train_cpu_light.py。")

    device = torch.device("cpu")
    model.load_state_dict(torch.load(weight_path, map_location=device))
    model.to(device)
    model.eval()

    with open(CLASS_JSON, "r", encoding="utf-8") as f:
        class_dict = json.load(f)

    transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(*norm)
    ])

    image = Image.open(img_path).convert("RGB")
    x = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(x).squeeze(0).cpu()
        prob = torch.softmax(output, dim=0)
        pred = int(torch.argmax(prob).item())

    print("预测图片：", img_path.name)
    print("使用模型：", model_name)
    print("预测类别：", class_dict[str(pred)])
    print("预测概率：", f"{prob[pred].item():.3f}")
    for i in range(len(prob)):
        print(f"class: {class_dict[str(i)]:10} prob: {prob[i].item():.3f}")


if __name__ == "__main__":
    main()
