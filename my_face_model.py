import torch
import torch.nn as nn
import torch.nn.functional as F


class MyFaceRecognitionModel(nn.Module):
    """
    MyFaceRecognitionModel 类是一个用于人脸识别的卷积神经网络模型，
    输入图像后输出一个固定长度的嵌入向量，可用于人脸的特征表示和比对。
    """

    def __init__(self, embedding_size=128):
        """
        初始化 MyFaceRecognitionModel 网络结构。

        Args:
            embedding_size (int): 嵌入向量的维度，默认为 128。
        """
        super(MyFaceRecognitionModel, self).__init__()

        # 第一层卷积层，将输入通道数从3扩展到64，卷积核大小为3x3，步长为1，填充为1
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1)

        # 第二层卷积层，将通道数从64扩展到128
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)

        # 第三层卷积层，将通道数从128扩展到256
        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1)

        # 全连接层，将卷积特征映射到 1024 维的隐藏层
        self.fc1 = nn.Linear(256 * 20 * 20, 1024)  # 假设输入图像的尺寸是 160x160

        # 输出层，将隐藏层映射到指定的嵌入维度
        self.fc2 = nn.Linear(1024, embedding_size)

    def forward(self, x):
        """
        定义前向传播过程。

        Args:
            x (Tensor): 输入张量，形状为 (batch_size, 3, 160, 160)。

        Returns:
            Tensor: 输出的嵌入向量，形状为 (batch_size, embedding_size)。
        """
        # 卷积和激活操作
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2)  # 2x2 最大池化，降低维度
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2)
        x = F.relu(self.conv3(x))
        x = F.max_pool2d(x, 2)

        # 展平操作，将卷积特征展平成一维向量
        x = x.view(x.size(0), -1)

        # 全连接层和激活操作
        x = F.relu(self.fc1(x))

        # 最后一层输出特征嵌入向量
        x = self.fc2(x)

        # 对输出进行 L2 归一化，以便进行余弦相似度计算
        return F.normalize(x, p=2, dim=1)
