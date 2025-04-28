import os
import torch
import pickle
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from evaluate import evaluate, plot_confusion_matrix

def train_model(model, train_loader, val_loader, criterion, optimizer, device,
                model_name="model", save_model_dir="./saved_models",
                num_epochs=100, patience=7):

    os.makedirs(save_model_dir, exist_ok=True)

    # 지표 저장용 리스트
    train_losses, val_losses = [], []
    train_accs, val_accs = [], []
    train_f1s, val_f1s = [], []
    train_precs, val_precs = [], []
    train_recs, val_recs = [], []

    # Early Stopping 변수
    best_acc = 0.0
    best_cm = None
    patience_counter = 0

    for epoch in range(1, num_epochs+1):
        model.train()
        total_loss = 0
        all_preds = []
        all_targets = []

        loop = tqdm(train_loader, desc=f"Epoch {epoch}/{num_epochs}")

        for x, y in loop:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            outputs = model(x)
            loss = criterion(outputs, y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * x.size(0)

            preds = torch.argmax(outputs, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(y.cpu().numpy())
            loop.set_postfix(loss=loss.item())

        avg_train_loss = total_loss / len(train_loader.dataset)
        train_acc = accuracy_score(all_targets, all_preds)
        train_prec = precision_score(all_targets, all_preds)
        train_rec = recall_score(all_targets, all_preds)
        train_f1 = f1_score(all_targets, all_preds)

        # Validation
        val_loss, val_acc, val_prec, val_rec, val_f1, cm = evaluate(model, val_loader, criterion, device)

        # 기록 저장
        train_losses.append(avg_train_loss)
        val_losses.append(val_loss)
        train_accs.append(train_acc)
        val_accs.append(val_acc)
        train_f1s.append(train_f1)
        val_f1s.append(val_f1)
        train_precs.append(train_prec)
        val_precs.append(val_prec)
        train_recs.append(train_rec)
        val_recs.append(val_rec)

        print(f"\n[Epoch {epoch}]")
        print(f"Train Loss  : {avg_train_loss:.4f}")
        print(f"Val Loss    : {val_loss:.4f}")
        print(f"Train Acc   : {train_acc:.4f} | Val Acc : {val_acc:.4f}")
        print(f"Train F1    : {train_f1:.4f} | Val F1  : {val_f1:.4f}")
        print(f"Train Rec   : {train_rec:.4f} | Val Rec : {val_rec:.4f}")
        print(f"Train Prec  : {train_prec:.4f} | Val Prec: {val_prec:.4f}")

        # Best model 저장
        if val_acc > best_acc:
            best_acc = val_acc
            best_cm = cm  # Save best confusion matrix
            model_filename = f"{model_name}_acc_{best_acc:.4f}.pt"
            cm_filename = f"{model_name}_cm_acc_{best_acc:.4f}.pkl"

            torch.save(model.state_dict(), os.path.join(save_model_dir, model_filename))
            with open(os.path.join(save_model_dir, cm_filename), "wb") as f:
                pickle.dump(cm, f)

            print(f"✅ Best model saved: {model_filename}")
            patience_counter = 0
        else:
            patience_counter += 1
            print(f"⏳ EarlyStopping patience: {patience_counter}/{patience}")
            if patience_counter >= patience:
                print("Early stopping triggered!")
                break

    # 학습 그래프 시각화
    epochs = range(1, len(train_losses) + 1)

    plt.figure(figsize=(16, 12))

    plt.subplot(2, 2, 1)
    plt.plot(epochs, train_losses, label='Train Loss')
    plt.plot(epochs, val_losses, label='Val Loss')
    plt.legend(); plt.title("Loss")

    plt.subplot(2, 2, 2)
    plt.plot(epochs, train_accs, label='Train Acc')
    plt.plot(epochs, val_accs, label='Val Acc')
    plt.legend(); plt.title("Accuracy")

    plt.subplot(2, 2, 3)
    plt.plot(epochs, train_precs, label='Train Prec')
    plt.plot(epochs, val_precs, label='Val Prec')
    plt.legend(); plt.title("Precision")

    plt.subplot(2, 2, 4)
    plt.plot(epochs, train_recs, label='Train Recall')
    plt.plot(epochs, val_recs, label='Val Recall')
    plt.legend(); plt.title("Recall")

    plt.tight_layout()
    plt.savefig(os.path.join(save_model_dir, "training_metrics.png"))
    plt.show()

    # 최종 best confusion matrix 시각화도 함께 저장
    if best_cm is not None:
        plt.figure(figsize=(5, 4))
        plot_confusion_matrix(best_cm, labels=["Normal", "Seizure"])
        plt.title("Best Val Confusion Matrix")
        plt.savefig(os.path.join(save_model_dir, f"{model_name}_best_val_confusion.png"))
        plt.close()

    return model
