import cv2
import numpy as np
import torch
from face_detection import FaceDetector  # 导入人脸检测模块
from data_preprocessing import FaceDataset  # 导入数据预处理模块
from my_face_model import MyFaceRecognitionModel  # 导入自定义模型
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
        str: 识别出的用户姓名，或 "Unregistered"（如果无法识别）。
        float: 计算出的相似度
    """
    min_dist = float('inf')
    recognized_user = "Unregistered"  # 默认值为 "Unregistered"
    for user_name, user_embedding in user_embeddings.items():
        dist = np.linalg.norm(embedding - user_embedding)  # 计算欧氏距离
        if dist < min_dist and dist < threshold:
            min_dist = dist
            recognized_user = user_name  # 如果相似度足够高，则认为是该用户
    return recognized_user, min_dist

def get_embedding_from_custom_model(image, model, device='cuda' if torch.cuda.is_available() else 'cpu'):
    """
    使用自定义模型提取人脸特征嵌入向量。

    Args:
        image (PIL.Image): 输入的人脸图像。
        model (torch.nn.Module): 自定义人脸识别模型。
        device (str): 运行模型的设备，默认为 GPU（如果可用）。

    Returns:
        numpy.ndarray: 生成的特征嵌入向量。
    """
    transform = FaceDataset.get_transform()  # 调用数据预处理模块的图像变换
    image_tensor = transform(image).unsqueeze(0).to(device)
    with torch.no_grad():
        embedding = model(image_tensor)
    return embedding.cpu().numpy().flatten()


def run_recognition(model_path='./saved_models/my_face_model.pth'):
    """
    运行实时人脸识别。使用自定义模型进行特征提取，启动摄像头检测和识别用户。

    Args:
        model_path (str): 自定义模型的路径。
    """
    # 1. 加载自定义模型
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = MyFaceRecognitionModel().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    # 2. 加载已注册用户的特征嵌入
    user_embeddings = load_registered_users()

    # 3. 初始化人脸检测模块
    detector = FaceDetector()

    # 4. 打开摄像头
    cap = cv2.VideoCapture(0)
    print("Press 'q' to exit recognition.")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # 使用 MTCNN 检测人脸
        boxes = detector.detect_faces(frame)

        # 如果检测到人脸
        if boxes is not None:
            frame = detector.draw_boxes(frame, boxes)  # 在人脸周围绘制方框

            # 对每一张检测到的人脸进行识别
            for box in boxes:
                x1, y1, x2, y2 = [int(b) for b in box]
                face = frame[y1:y2, x1:x2]  # 裁剪人脸区域
                face_pil = Image.fromarray(cv2.cvtColor(face, cv2.COLOR_BGR2RGB))  # 转换为 PIL 格式
                embedding = get_embedding_from_custom_model(face_pil, model)  # 获取自定义模型特征嵌入
                user_name, similarity = recognize_user(embedding, user_embeddings)  # 识别用户姓名

                # 在图像上显示识别结果
                cv2.putText(frame, f"Recognition Result: {user_name}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                cv2.putText(frame, f"Similarity: {similarity:.2f}, Threshold: 0.6", (x1, y2 + 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        # 如果没有检测到人脸
        else:
            cv2.putText(frame, "No face detected", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        # 显示实时人脸识别画面
        cv2.imshow("Face Recognition", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # 5. 释放资源
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    run_recognition()
