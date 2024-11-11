import os
import cv2
import numpy as np
from face_detection import FaceDetector  # 从 face_detection.py 中导入人脸检测模块
from face_embedding import FaceEmbedder  # 从 face_embedding.py 中导入人脸特征提取模块
from PIL import Image  # 用于图像格式转换


def register_user(user_name, data_dir='./registered_users'):
    """
    注册新用户并保存人脸特征嵌入向量。

    Args:
        user_name (str): 用户的姓名，用于创建用户文件夹。
        data_dir (str): 注册用户数据的保存目录，默认为 './registered_users'。
    """
    # 为新用户创建文件夹
    user_dir = os.path.join(data_dir, user_name)
    os.makedirs(user_dir, exist_ok=True)

    # 实例化人脸检测器和特征提取器
    detector = FaceDetector()
    embedder = FaceEmbedder()

    # 启动摄像头
    cap = cv2.VideoCapture(0)
    print("按 'q' 退出注册。按 'c' 捕捉图像。")

    while True:
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
            break
        elif key == ord('c') and boxes is not None:
            for box in boxes:
                x1, y1, x2, y2 = [int(b) for b in box]
                face = frame[y1:y2, x1:x2]  # 提取人脸区域
                face_pil = Image.fromarray(cv2.cvtColor(face, cv2.COLOR_BGR2RGB))  # 转换为 PIL 格式
                embedding = embedder.get_embedding(face_pil)  # 获取特征嵌入向量

                # 保存嵌入特征向量
                np.save(os.path.join(user_dir, f"{user_name}_embedding.npy"), embedding)
                print(f"{user_name} 已注册.")

    # 释放摄像头并关闭所有窗口
    cap.release()
    cv2.destroyAllWindows()


# 测试调用
if __name__ == "__main__":
    user_name = input("请输入用户名: ")
    register_user(user_name)
