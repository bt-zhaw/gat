from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
import torch

def evaluate_model(model, test_data):
    model.eval()
    with torch.no_grad():
        logits = model(test_data)
        preds = logits.max(1)[1]

    cm = confusion_matrix(test_data.y.numpy(), preds.numpy())
    accuracy = accuracy_score(test_data.y.numpy(), preds.numpy())
    precision = precision_score(test_data.y.numpy(), preds.numpy(), average='binary')
    recall = recall_score(test_data.y.numpy(), preds.numpy(), average='binary')
    f1 = f1_score(test_data.y.numpy(), preds.numpy(), average='binary')

    print("Confusion Matrix:")
    print(cm)
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1-score: {f1:.4f}")
