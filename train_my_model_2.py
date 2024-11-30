import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader
from data_preprocessing import FaceDataset  # 自定义数据集加载模块
from my_face_model import MyFaceRecognitionModel  # 自定义模型
from sklearn.metrics import accuracy_score, f1_score  # 评估指标
import os
import math
from tqdm import tqdm  # 进度条显示

class ArcFaceLoss(nn.Module):
    def __init__(self, scale=30.0, margin=0.5):
        super(ArcFaceLoss, self).__init__()
        self.scale = scale  # 缩放因子
        self.margin = margin  # 角度间隔
        self.cos_m = torch.cos(margin)
        self.sin_m = torch.sin(margin)
        self.thresh = torch.cos(math.pi - margin)
        self.mm = torch.sin(math.pi - margin) * margin

    def forward(self, cosine, label):
        sine = torch.sqrt(1.0 - torch.pow(cosine, 2))
        phi = cosine * self.cos_m - sine * self.sin_m  # 更新后的角度余弦值
        phi = torch.where(cosine > self.thresh, phi, cosine - self.mm)  # 应用角度约束

        # 通过标签进行角度更新
        one_hot = torch.zeros(cosine.size(), device=cosine.device)
        one_hot.scatter_(1, label.view(-1, 1).long(), 1)
        output = (one_hot * phi) + ((1.0 - one_hot) * cosine)
        output *= self.scale
        return output

def train_model(dataset_dir, test_dataset_dir, save_model_path, epochs=20, batch_size=32, embedding_size=128):
    """
    训练人脸识别模型

    参数:
        dataset_dir (str): 训练集路径
        test_dataset_dir (str): 测试集路径
        save_model_path (str): 模型保存路径
        epochs (int): 训练轮数
        batch_size (int): 批次大小
        embedding_size (int): 嵌入向量大小
    """
    # 加载数据集
    train_dataset = FaceDataset(dataset_dir)
    test_dataset = FaceDataset(test_dataset_dir)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # 初始化模型和优化器
    model = MyFaceRecognitionModel(embedding_size=embedding_size).cuda()  # GPU 加速
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    arcface_criterion = ArcFaceLoss(scale=30, margin=0.5)  # ArcFace Loss

    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        for images, labels in tqdm(train_loader, desc=f"Training Epoch {epoch + 1}/{epochs}"):
            images, labels = images.cuda(), labels.cuda()
            optimizer.zero_grad()
            embeddings = model(images)
            cosine = nn.functional.linear(embeddings, model.weight)  # Cosine similarity
            output = arcface_criterion(cosine, labels)  # 应用 ArcFace Loss
            loss = nn.CrossEntropyLoss()(output, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        # 测试集评估
        model.eval()
        all_preds, all_labels = [], []
        with torch.no_grad():
            for images, labels in test_loader:
                images, labels = images.cuda(), labels.cuda()
                embeddings = model(images)
                cosine = nn.functional.linear(embeddings, model.weight)
                _, predicted = torch.max(cosine, 1)
                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        accuracy = accuracy_score(all_labels, all_preds)
        f1 = f1_score(all_labels, all_preds, average='weighted')
        print(f"Epoch [{epoch + 1}/{epochs}], Loss: {train_loss / len(train_loader):.4f}, Accuracy: {accuracy:.4f}, F1 Score: {f1:.4f}")

        # 保存模型
        torch.save(model.state_dict(), save_model_path)
        print(f"模型已保存至 {save_model_path}")

if __name__ == "__main__":
    # 开始训练
    train_model(
        dataset_dir="./casia_webface",  # 训练集路径
        test_dataset_dir="./lfw",       # 测试集路径
        save_model_path="./saved_models/my_face_model.pth",  # 保存模型的路径
        epochs=20,
        batch_size=32,
        embedding_size=128
    )
