"""
该文件用于测试模型（AlexNet/LeNet）
通过修改以下几处可切换两个网络的训练(两类测试图片)：
1、
    # 选择测试图片，这是在网上随便找的两张图片，已包含在文件夹中，可自行用其他图片测试
    #img_path = "./BRAIN_TUMOR/test_yes.jpg"
    img_path = "./BRAIN_TUMOR/test_no.png"
2、
    #此处选择模型
    #model = AlexNet(num_classes=2).to(device)
    model = LeNet(num_classes=2).to(device)
3、
    #此处选择训练好的权重文件（模型）
    #weights_path = "./BRAIN_TUMOR/AlexNet.pth"
    weights_path = "./BRAIN_TUMOR/LeNet.pth"
"""




import os
import json

import torch
from PIL import Image
from torchvision import transforms
import matplotlib.pyplot as plt

from model import AlexNet
from model import LeNet

def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    data_transform = transforms.Compose(
        [transforms.Resize((224, 224)),
         transforms.ToTensor(),
         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    # 选择测试图片，这是在网上随便找的两张图片，已包含在文件夹中，可自行用其他图片测试
    #img_path = "./BRAIN_TUMOR/test_yes.jpg"
    img_path = "./BRAIN_TUMOR/test_no.png"

    assert os.path.exists(img_path), "file: '{}' dose not exist.".format(img_path)
    img = Image.open(img_path)

    plt.imshow(img)
    # [N, C, H, W]
    img = data_transform(img)
    # expand batch dimension
    img = torch.unsqueeze(img, dim=0)

    # read class_indict
    json_path = './BRAIN_TUMOR/class_indices.json'
    assert os.path.exists(json_path), "file: '{}' dose not exist.".format(json_path)

    with open(json_path, "r") as f:
        class_indict = json.load(f)

    #此处选择模型
    #model = AlexNet(num_classes=2).to(device)
    model = LeNet(num_classes=2).to(device)

    #此处选择训练好的权重文件（模型）
    #weights_path = "./BRAIN_TUMOR/AlexNet.pth"
    weights_path = "./BRAIN_TUMOR/LeNet.pth"

    assert os.path.exists(weights_path), "file: '{}' dose not exist.".format(weights_path)
    model.load_state_dict(torch.load(weights_path))

    model.eval()
    with torch.no_grad():
        # predict class
        output = torch.squeeze(model(img.to(device))).cpu()
        predict = torch.softmax(output, dim=0)
        predict_cla = torch.argmax(predict).numpy()

    print_res = "class: {}   prob: {:.3}".format(class_indict[str(predict_cla)],
                                                 predict[predict_cla].numpy())
    plt.title(print_res)
    for i in range(len(predict)):
        print("class: {:10}   prob: {:.3}".format(class_indict[str(i)],
                                                  predict[i].numpy()))
    plt.show()


if __name__ == '__main__':
    main()