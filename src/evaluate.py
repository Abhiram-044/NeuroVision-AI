import torch
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

from dataset import get_dataloaders
from model import build_model

def evaluate():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    _, val_loader, classes= get_dataloaders()

    model = build_model(num_classes=len(classes), freeze_backbone=False)
    model.load_state_dict(torch.load("resnet50_brain_tumor.pth"))
    model.to(device)

    model.eval()

    all_preds = []
    all_labels = []

    with torch.no_grad():
        for images, labels in val_loader:
            images = images.to(device)
            
            outputs = model(images)
            _, preds = torch.max(outputs, 1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.numpy())

        print("\nClassification report:\n")
        print(classification_report(all_labels, all_preds,
                                    target_names=classes))
        
        cm = confusion_matrix(all_labels, all_preds)

        plt.figure(figsize=(8, 6))
        sns.heatmap(cm,
                    annot=True,
                    fmt="d",
                    xticklabels=classes,
                    yticklabels=classes)
        plt.xlabel("Predicted")
        plt.ylabel("Actual")
        plt.title("Confusion Matrix")
        plt.show()

if __name__ == "__main__":
    evaluate()