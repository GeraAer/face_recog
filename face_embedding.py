import torch
import torchvision.transforms as transforms
from PIL import Image
from facenet_pytorch import InceptionResnetV1

# 使用 FaceNet 提取人脸特征向量
class FaceEmbedder:
    """
    FaceEmbedder 类用于提取人脸特征向量。
    该类使用了 FaceNet 模型（InceptionResnetV1）来生成图像的嵌入向量，
    可以作为人脸识别和相似度计算的基础。
    """

    def __init__(self, device='cuda' if torch.cuda.is_available() else 'cpu'):
        """
        初始化 FaceEmbedder 类并加载 FaceNet 模型。

        Args:
            device (str): 设备名称（'cuda' 或 'cpu'）。默认为 'cuda'，如果没有可用的 GPU，则使用 'cpu'。
        """
        self.device = device
        # 加载预训练的 FaceNet 模型，该模型在 VGGFace2 数据集上进行了预训练
        self.model = InceptionResnetV1(pretrained='vggface2').eval().to(self.device)
        # 定义图像预处理步骤，确保输入符合模型的要求
        self.transform = transforms.Compose([
            transforms.Resize((160, 160)),  # 将图像调整为 160x160 的大小
            transforms.ToTensor(),  # 转换为张量
            transforms.Normalize([0.5], [0.5])  # 归一化，使像素值在 -1 到 1 之间
        ])

    def get_embedding(self, image):
        """
        获取输入图像的人脸嵌入特征向量。

        Args:
            image (PIL.Image): 输入的 PIL 图像。

        Returns:
            numpy array: 人脸嵌入特征向量。
        """
        # 对图像进行预处理并转换为 4D 张量 [batch_size, channels, height, width]
        image = self.transform(image).unsqueeze(0).to(self.device)
        with torch.no_grad():  # 不计算梯度，加速并减少显存占用
            embedding = self.model(image)  # 获取人脸嵌入向量
        return embedding.cpu().numpy().flatten()  # 将向量转换为 numpy 数组并拉平成一维
