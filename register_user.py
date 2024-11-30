import os
import cv2
import numpy as np
from face_detection import FaceDetector  # 从 face_detection.py 中导入人脸检测模块
from face_embedding import FaceEmbedder  # 从 face_embedding.py 中导入人脸特征提取模块
from PIL import Image  # 用于图像格式转换

def register_user(user_name, data_dir='./registered_users', num_images=5):
    """
    注册新用户并保存多张人脸特征嵌入向量。用户按下 'c' 键来捕捉每张图像。

    Args:
        user_name (str): 用户的姓名，用于创建用户文件夹。
        data_dir (str): 注册用户数据的保存目录，默认为 './registered_users'。
        num_images (int): 要捕捉的图像数量，默认为5张。
    """
    # 为新用户创建文件夹
    user_dir = os.path.join(data_dir, user_name)
    os.makedirs(user_dir, exist_ok=True)

    # 实例化人脸检测器和特征提取器
    detector = FaceDetector()
    embedder = FaceEmbedder()

    # 启动摄像头
    cap = cv2.VideoCapture(0)
    print(f"Press 'c' to capture a face image. You need to capture {num_images} images.")
    print("Press 'q' to quit the registration process.")

    embeddings = []  # 用于存储用户的嵌入向量

    while len(embeddings) < num_images:
        # 捕捉帧
        ret, frame = cap.read()
        if not ret:
            break

        # 检测并绘制人脸框
        boxes = detector.detect_faces(frame)
        frame = detector.draw_boxes(frame, boxes)

        # 显示图像窗口
        cv2.imshow("Register User", frame)

        # 获取键盘输入
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break  # 退出注册过程
        elif key == ord('c') and boxes is not None:
            for box in boxes:
                x1, y1, x2, y2 = [int(b) for b in box]
                face = frame[y1:y2, x1:x2]  # 提取人脸区域
                face_pil = Image.fromarray(cv2.cvtColor(face, cv2.COLOR_BGR2RGB))  # 转换为 PIL 格式
                embedding = embedder.get_embedding(face_pil)  # 获取特征嵌入向量

                embeddings.append(embedding)  # 将特征向量添加到列表
                print(f"Captured {len(embeddings)}/{num_images} images")

                # 如果已捕捉到足够的图像，停止捕捉
                if len(embeddings) >= num_images:
                    break

    # 保存所有嵌入特征向量
    if len(embeddings) == num_images:
        # 将所有嵌入向量取平均作为最终的用户特征
        avg_embedding = np.mean(embeddings, axis=0)
        np.save(os.path.join(user_dir, f"{user_name}_embedding.npy"), avg_embedding)
        print(f"{user_name} has been registered. Feature vector has been saved.")

    # 释放摄像头并关闭所有窗口
    cap.release()
    cv2.destroyAllWindows()


# 测试调用
if __name__ == "__main__":
    user_name = input("Please enter the user name: ")
    register_user(user_name)
