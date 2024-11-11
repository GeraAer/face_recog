import cv2
import numpy as np
import torch
from face_detection import FaceDetector  # 从 face_detection.py 中导入人脸检测模块
from face_embedding import FaceEmbedder  # 从 face_embedding.py 中导入人脸特征提取模块
from PIL import Image
import os


def load_registered_users(data_dir='./registered_users'):
    """
    加载已注册用户的特征嵌入向量。

    Args:
        data_dir (str): 已注册用户的目录，默认为 './registered_users'。

    Returns:
        dict: 键为用户姓名，值为用户的特征嵌入向量。
    """
    user_embeddings = {}
    for user_name in os.listdir(data_dir):
        user_dir = os.path.join(data_dir, user_name)
        embedding_path = os.path.join(user_dir, f"{user_name}_embedding.npy")
        if os.path.exists(embedding_path):
            user_embeddings[user_name] = np.load(embedding_path)  # 加载用户嵌入向量
    return user_embeddings


def recognize_user(embedding, user_embeddings, threshold=0.6):
    """
    根据输入的嵌入向量识别用户。

    Args:
        embedding (numpy.ndarray): 待识别人脸的嵌入向量。
        user_embeddings (dict): 已注册用户的嵌入向量。
        threshold (float): 相似度阈值，默认为 0.6。

    Returns:
        str: 识别出的用户姓名，或 "未知"（如果无法识别）。
    """
    min_dist = float('inf')
    recognized_user = "未知"
    for user_name, user_embedding in user_embeddings.items():
        dist = np.linalg.norm(embedding - user_embedding)  # 计算欧氏距离
        if dist < min_dist and dist < threshold:
            min_dist = dist
            recognized_user = user_name
    return recognized_user


def run_recognition():
    """
    运行实时人脸识别。启动摄像头，检测和识别用户。
    """
    detector = FaceDetector()  # 实例化人脸检测器
    embedder = FaceEmbedder()  # 实例化特征提取器
    user_embeddings = load_registered_users()  # 加载已注册用户的嵌入向量

    cap = cv2.VideoCapture(0)
    print("按 'q' 退出识别。")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # 使用 MTCNN 检测人脸
        boxes = detector.detect_faces(frame)
        frame = detector.draw_boxes(frame, boxes)  # 在人脸周围绘制方框

        # 对每一张检测到的人脸进行识别
        for box in boxes:
            x1, y1, x2, y2 = [int(b) for b in box]
            face = frame[y1:y2, x1:x2]  # 裁剪人脸区域
            face_pil = Image.fromarray(cv2.cvtColor(face, cv2.COLOR_BGR2RGB))  # 转换为 PIL 格式
            embedding = embedder.get_embedding(face_pil)  # 获取特征向量
            user_name = recognize_user(embedding, user_embeddings)  # 识别用户姓名

            # 在图像上显示识别结果
            cv2.putText(frame, f"{user_name}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        # 显示实时人脸识别画面
        cv2.imshow("Face Recognition", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    run_recognition()
