from facenet_pytorch import MTCNN
import cv2
import torch

# 使用 MTCNN 进行人脸检测
class FaceDetector:
    """
    FaceDetector 类用于使用 MTCNN（Multi-task Cascaded Convolutional Networks）进行人脸检测。
    MTCNN 是一个基于深度学习的模型，能够精确地检测人脸并标记其边界框。
    """

    def __init__(self, device='cuda' if torch.cuda.is_available() else 'cpu'):
        """
        初始化 FaceDetector 类，加载 MTCNN 模型。

        Args:
            device (str): 设备名称（'cuda' 或 'cpu'）。默认为 'cuda'，如果没有可用的 GPU，则使用 'cpu'。
        """
        self.device = device
        # 初始化 MTCNN 模型
        self.mtcnn = MTCNN(keep_all=True, device=self.device)

    def detect_faces(self, image):
        """
        检测图像中的人脸，并返回人脸的边界框坐标。

        Args:
            image (numpy array): 输入图像（BGR 格式）。

        Returns:
            boxes (numpy array): 检测到的每个人脸的边界框坐标，格式为 [x1, y1, x2, y2]。
                                 如果没有检测到人脸，返回 None。
        """
        # 使用 MTCNN 检测人脸
        boxes, _ = self.mtcnn.detect(image)
        return boxes

    def draw_boxes(self, image, boxes):
        """
        在图像上绘制人脸的边界框。

        Args:
            image (numpy array): 输入图像（BGR 格式）。
            boxes (numpy array): 人脸的边界框坐标，格式为 [x1, y1, x2, y2]。

        Returns:
            image (numpy array): 包含绘制边界框的图像。
        """
        if boxes is not None:
            for box in boxes:
                # 绘制矩形框，每个框的坐标为 [x1, y1, x2, y2]
                cv2.rectangle(image,
                              (int(box[0]), int(box[1])),
                              (int(box[2]), int(box[3])),
                              (0, 255, 0), 2)  # 绿色框，线条宽度为 2
        return image
