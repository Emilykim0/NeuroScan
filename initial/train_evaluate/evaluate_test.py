import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pickle

# 혼동행렬 시각화 함수
def plot_confusion_matrix(cm, labels=['Normal', 'Seizure']):
    plt.figure(figsize=(5,4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=labels, yticklabels=labels)
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.title("Confusion Matrix (Test Set)")
    plt.tight_layout()
    plt.show()

# 평가 함수
def evaluate_on_test(model, test_loader, device):
    model.eval()
    preds, trues = [], []

    with torch.no_grad():
        for x, y in test_loader:
            x, y = x.to(device), y.to(device)
            output = model(x)
            pred = output.argmax(dim=1)
            preds.extend(pred.cpu().numpy())
            trues.extend(y.cpu().numpy())

    acc = accuracy_score(trues, preds)
    prec = precision_score(trues, preds)
    rec = recall_score(trues, preds)
    f1 = f1_score(trues, preds)
    cm = confusion_matrix(trues, preds)

    print("\n[Test Evaluation]")
    print(f"Accuracy  : {acc:.4f}")
    print(f"Precision : {prec:.4f}")
    print(f"Recall    : {rec:.4f}")
    print(f"F1 Score  : {f1:.4f}")
    print("Confusion Matrix:\n", cm)
    plot_confusion_matrix(cm)

