import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader
from data_preprocessing import FaceDataset
from inception_resnet_v1 import InceptionResnetV1
from sklearn.metrics import accuracy_score, f1_score
import os
from tqdm import tqdm
from torch.optim.lr_scheduler import StepLR


def train_model(dataset_dir, test_dataset_dir, save_model_path, epochs=20, batch_size=32, embedding_size=128, lr=0.001, step_size=10):
    """
    训练人脸识别模型 InceptionResnetV1。

    参数:
        dataset_dir (str): 训练集路径。
        test_dataset_dir (str): 测试集路径。
        save_model_path (str): 模型保存路径。
        epochs (int): 训练轮数。
        batch_size (int): 批次大小。
        embedding_size (int): 嵌入向量大小。
        lr (float): 学习率。
        step_size (int): 学习率调整间隔。
    """
    # 加载数据集
    train_dataset = FaceDataset(dataset_dir)
    test_dataset = FaceDataset(test_dataset_dir)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # 初始化模型、优化器、损失函数和学习率调整器
    model = InceptionResnetV1(embedding_size=embedding_size).cuda()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = StepLR(optimizer, step_size=step_size, gamma=0.1)  # 学习率衰减
    criterion = nn.CrossEntropyLoss()

    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        for images, labels in tqdm(train_loader, desc=f"Training Epoch {epoch + 1}/{epochs}"):
            images, labels = images.cuda(), labels.cuda()

            # 前向传播、计算损失、反向传播并更新参数
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        # 测试集评估
        model.eval()
        all_preds, all_labels = [], []
        with torch.no_grad():
            for images, labels in test_loader:
                images, labels = images.cuda(), labels.cuda()
                outputs = model(images)
                _, predicted = torch.max(outputs, 1)
                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        accuracy = accuracy_score(all_labels, all_preds)
        f1 = f1_score(all_labels, all_preds, average='weighted')
        print(f"Epoch [{epoch + 1}/{epochs}], Loss: {train_loss / len(train_loader):.4f}, Accuracy: {accuracy:.4f}, F1 Score: {f1:.4f}")

        # 学习率调整
        scheduler.step()

        # 保存模型
        torch.save(model.state_dict(), save_model_path)
        print(f"模型已保存至 {save_model_path}")

if __name__ == "__main__":
    # 执行训练
    train_model(
        dataset_dir="./casia_webface",  # 训练集路径
        test_dataset_dir="./lfw",       # 测试集路径
        save_model_path="./saved_models/inception_resnet_v1.pth",  # 保存模型路径
        epochs=20,
        batch_size=32,
        embedding_size=128,
        lr=0.001,
        step_size=10
    )
