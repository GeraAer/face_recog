import torch
from facenet_pytorch import InceptionResnetV1
from my_face_model import MyFaceRecognitionModel
from data_preprocessing import FaceDataset
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, f1_score
import numpy as np
#未完成，没有什么意义 e e e
def evaluate_model(model, dataloader):
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for images, labels in dataloader:
            images, labels = images.cuda(), labels.cuda()
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    accuracy = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average='weighted')
    return accuracy, f1

if __name__ == "__main__":
    # Load datasets
    test_dataset = FaceDataset("./lfw")
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    # Evaluate FaceNet model
    facenet_model = InceptionResnetV1(pretrained='vggface2').eval().cuda()
    accuracy, f1 = evaluate_model(facenet_model, test_loader)
    print(f"FaceNet Model - Accuracy: {accuracy:.4f}, F1 Score: {f1:.4f}")

    # Evaluate custom trained model
    custom_model = MyFaceRecognitionModel(embedding_size=128).cuda()
    custom_model.load_state_dict(torch.load("./saved_models/my_face_model.pth"))
    accuracy, f1 = evaluate_model(custom_model, test_loader)
    print(f"Custom Model - Accuracy: {accuracy:.4f}, F1 Score: {f1:.4f}")
