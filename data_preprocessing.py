import os
import torch
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as transforms

class FaceDataset(Dataset):
    """
    自定义的FaceDataset类，用于人脸识别任务的数据集加载。
    该类会从指定的数据目录中加载图片及其对应的标签，适用于人脸识别训练。

    初始化参数:
    - data_dir: 数据目录的路径，要求每个人的图片放在单独的文件夹中，
                文件夹名称为类别名称（例如，人员的名字或 ID）。

    功能：
    - 自动遍历 data_dir 目录下的所有文件夹，每个文件夹中的图片属于同一个人。
    - 加载图片并将其转换为 160x160 的大小，同时进行标准化，以符合 FaceNet 的预处理方式。
    """

    def __init__(self, data_dir):
        """
        初始化数据集。设置数据目录，加载所有图片路径及其对应的类别标签。

        Args:
            data_dir (str): 包含人脸图片的数据目录路径。
                            目录结构要求如下：
                            └── data_dir
                                ├── person1
                                │   ├── img1.jpg
                                │   ├── img2.jpg
                                ├── person2
                                │   ├── img1.jpg
                                │   ├── img2.jpg
                            每个文件夹（如 person1 和 person2）代表一个类别。
        """
        self.data_dir = data_dir
        # 获取目录下所有文件夹名称，每个文件夹对应一个类别
        self.classes = os.listdir(data_dir)
        # 用于存储所有图片的路径及其对应的标签
        self.data = []
        self.labels = []
        # 定义图像变换，包含大小调整和标准化
        self.transform = transforms.Compose([
            transforms.Resize((160, 160)),   # 将图片调整为 160x160
            transforms.ToTensor(),           # 转换为 PyTorch 的张量格式
            transforms.Normalize([0.5], [0.5])  # 将像素值标准化到 [-1, 1]
        ])

        # 遍历每个类别文件夹，将图片路径和对应标签加入列表
        for idx, label in enumerate(self.classes):
            user_dir = os.path.join(data_dir, label)
            for img_name in os.listdir(user_dir):
                img_path = os.path.join(user_dir, img_name)
                self.data.append(img_path)
                self.labels.append(idx)

    def __len__(self):
        """
        返回数据集的总图片数量。
        """
        return len(self.data)

    def __getitem__(self, idx):
        """
        根据索引返回一张图片及其对应的标签。

        Args:
            idx (int): 图片索引

        Returns:
            image (torch.Tensor): 经过预处理的图片张量
            label (int): 图片对应的类别标签
        """
        img_path = self.data[idx]
        label = self.labels[idx]
        # 加载图片并转换为 RGB 模式
        image = Image.open(img_path).convert('RGB')
        # 应用预定义的图像变换
        image = self.transform(image)
        return image, label
