import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from data_preprocessing import FaceDataset  # 导入自定义数据集加载模块
from my_face_model import MyFaceRecognitionModel  # 导入自定义的模型模块
from sklearn.metrics import accuracy_score, f1_score  # 用于计算评估指标
import os
from tqdm import tqdm  # 用于显示进度条
# from arcface_loss import ArcFaceLoss  # 确保引入 ArcFace 损失

def train_model(dataset_dir, test_dataset_dir, save_model_path, epochs=20, batch_size=32, embedding_size=128):
    """
    使用ArcFace损失函数训练模型的主函数。

    Args:
        dataset_dir (str): 训练数据集路径。
        test_dataset_dir (str): 测试数据集路径。
        save_model_path (str): 保存模型路径。
        epochs (int): 训练的轮数。
        batch_size (int): 批次大小。
        embedding_size (int): 嵌入向量的大小。
    """
    # 加载训练集和测试集
    train_dataset = FaceDataset(dataset_dir)
    test_dataset = FaceDataset(test_dataset_dir)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # 初始化模型、优化器和ArcFace损失函数
    model = MyFaceRecognitionModel(embedding_size=embedding_size).cuda()  # 将模型加载到 GPU 上
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = ArcFaceLoss(embedding_size=embedding_size, num_classes=train_dataset.get_num_classes()).cuda()  # 使用ArcFace损失

    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        for images, labels in tqdm(train_loader, desc=f"Training Epoch {epoch + 1}/{epochs}"):
            images, labels = images.cuda(), labels.cuda()

            # 前向传播、计算损失、反向传播并更新参数
            optimizer.zero_grad()
            embeddings = model(images)
            loss = criterion(embeddings, labels)  # 使用ArcFace损失
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        # 在测试集上评估
        model.eval()
        all_preds, all_labels = [], []
        with torch.no_grad():
            for images, labels in test_loader:
                images, labels = images.cuda(), labels.cuda()
                embeddings = model(images)
                _, predicted = torch.max(embeddings, 1)
                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        accuracy = accuracy_score(all_labels, all_preds)
        f1 = f1_score(all_labels, all_preds, average='weighted')

        print(
            f"Epoch [{epoch + 1}/{epochs}], Loss: {train_loss / len(train_loader):.4f}, Accuracy: {accuracy:.4f}, F1 Score: {f1:.4f}")

        # 保存模型
        torch.save(model.state_dict(), save_model_path)
        print(f"模型已保存至 {save_model_path}")

if __name__ == "__main__":
    # 执行训练
    train_model(
        dataset_dir="./casia_webface",  # 指定训练数据集路径
        test_dataset_dir="./lfw",  # 指定测试数据集路径
        save_model_path="./saved_models/my_face_model.pth",  # 保存模型的路径
        epochs=20,  # 训练轮数
        batch_size=32,  # 批次大小
        embedding_size=128  # 嵌入向量大小
    )
