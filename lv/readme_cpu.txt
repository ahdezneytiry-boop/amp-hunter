脑肿瘤MRI二分类识别项目说明（CPU轻量版）

一、运行顺序
如果只是演示预测效果，先运行：
python predict_cpu.py lenet test_no.png
python predict_cpu.py lenet test_yes.jpg

如果需要重新训练轻量模型，运行：
python train_cpu_light.py
训练完成后再运行：
python predict_cpu.py tinycnn test_no.png
python predict_cpu.py tinycnn test_yes.jpg

ResNet34模型较大，普通无GPU电脑不建议重新训练。若已有ResNet34.pth，可只做预测：
python predict_cpu.py resnet34 test_no.png

二、文件放置要求
所有代码文件、权重文件、测试图片和数据集文件夹应放在同一个BRAIN_TUMOR目录下。
目录结构如下：
BRAIN_TUMOR
├─ model.py
├─ model_cpu_light.py
├─ train_cpu_light.py
├─ predict_cpu.py
├─ class_indices.json
├─ LeNet.pth
├─ ResNet34.pth
├─ test_no.png
├─ test_yes.jpg
├─ train
│  ├─ no
│  └─ yes
└─ val
   ├─ no
   └─ yes

三、本设备环境说明
本项目按普通Windows电脑环境进行配置，不依赖GPU。程序默认使用CPU运行，适合课程设计演示、单张图片预测和小规模训练。由于没有GPU，ResNet34等较深网络训练耗时较长，因此训练部分采用轻量化TinyCNN模型，降低图片尺寸、批量大小和数据加载线程数，保证在普通电脑上更容易运行。

四、环境依赖
Python 3.x
PyTorch
torchvision
Pillow
tqdm

五、项目功能
本项目用于脑部MRI图像二分类识别，判断输入图像属于no或yes类别。其中no表示未检测到脑肿瘤，yes表示检测到脑肿瘤。项目支持使用已有LeNet权重进行快速预测，也支持使用CPU轻量模型进行重新训练。

六、注意事项
本项目仅用于课程设计和学习演示，输出结果不能作为医学诊断依据。
