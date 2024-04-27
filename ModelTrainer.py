import torch
import torch.nn.functional as F
from torch.optim import Adam
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score, accuracy_score

from gat.data_processing import train_test_splitting, convert_to_graph
from gat.load_data import load_data
from gat.model import GAT


class ModelTrainer:
    def __init__(self, X, y, model_params, optimizer_params):
        # Process the data using the provided functions
        X_train, X_test, y_train, y_test = train_test_splitting(X, y)
        self.train_data = convert_to_graph(X_train, y_train)
        self.test_data = convert_to_graph(X_test, y_test)

        self.model = GAT(num_features=self.train_data.num_features, **model_params)
        #self.optimizer = torch.optim.Adagrad(self.model.parameters(), **optimizer_params)
        #self.optimizer = torch.optim.RMSprop(self.model.parameters(), **optimizer_params)
        #self.optimizer = torch.optim.SGD(self.model.parameters(), **optimizer_params)
        self.optimizer = torch.optim.AdamW(self.model.parameters(), **optimizer_params)

        self.loss_history = []
        self.accuracy_history = []
        self.best_accuracy = 0.0
        self.best_epoch = 0
        self.best_metrics = {}

    def train(self, num_epochs):
        for epoch in range(num_epochs):
            loss = self._train_one_epoch()
            self.loss_history.append(loss)
            accuracy = self._evaluate(epoch)
            self.accuracy_history.append(accuracy)
            print(f'Epoch {epoch + 1}, Loss: {loss:.4f}, Accuracy: {accuracy:.4f}')

        self._plot_loss_and_accuracy()
        self._print_best_epoch_metrics()

    def _train_one_epoch(self):
        self.model.train()
        self.optimizer.zero_grad()
        output = self.model(self.train_data)
        loss = F.nll_loss(output[self.train_data.train_mask], self.train_data.y[self.train_data.train_mask])
        loss.backward()
        self.optimizer.step()
        return loss.item()

    def _evaluate(self, epoch):
        self.model.eval()
        with torch.no_grad():
            output = self.model(self.test_data)
            _, predicted = torch.max(output, 1)
            correct = predicted[self.test_data.test_mask].eq(self.test_data.y[self.test_data.test_mask]).sum().item()
            accuracy = correct / int(self.test_data.test_mask.sum())

            if accuracy > self.best_accuracy:
                self.best_accuracy = accuracy
                self.best_epoch = epoch + 1
                self._update_best_metrics(predicted)

            return accuracy

    def _update_best_metrics(self, predicted):
        all_labels = np.unique(self.test_data.y.numpy())
        self.best_metrics = {
            'conf_matrix': confusion_matrix(self.test_data.y.numpy(), predicted.numpy(), labels=all_labels),
            'precision': precision_score(self.test_data.y.numpy(), predicted.numpy(), average='weighted', labels=all_labels),
            'recall': recall_score(self.test_data.y.numpy(), predicted.numpy(), average='weighted', labels=all_labels),
            'f1': f1_score(self.test_data.y.numpy(), predicted.numpy(), average='weighted', labels=all_labels),
            'test_accuracy': accuracy_score(self.test_data.y.numpy(), predicted.numpy())
        }

    def _plot_loss_and_accuracy(self):
        plt.figure(figsize=(12, 6))
        plt.subplot(1, 2, 1)
        plt.plot(self.loss_history, label='Loss', marker='o')
        plt.title('Loss over Epochs')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.grid(True)
        plt.legend()

        plt.subplot(1, 2, 2)
        plt.plot(self.accuracy_history, label='Accuracy', color='r', marker='x')
        plt.title('Accuracy over Epochs')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.grid(True)
        plt.legend()

        plt.tight_layout()
        plt.show()

    def _print_best_epoch_metrics(self):
        print("Best Epoch Results:")
        print("Confusion Matrix:\n", self.best_metrics['conf_matrix'])
        print(f"Test Accuracy: {self.best_metrics['test_accuracy']:.4f}")
        print(f"Precision: {self.best_metrics['precision']:.4f}")
        print(f"Recall: {self.best_metrics['recall']:.4f}")
        print(f"F1 Score: {self.best_metrics['f1']:.4f}")

    def report(self):
        print(f'Best Epoch: {self.best_epoch} with Accuracy: {self.best_accuracy:.4f}')

# Usage Example:
# Assuming 'X' and 'y' are defined somewhere in your project
X, y, header = load_data()
trainer = ModelTrainer(X, y, model_params={'num_classes': 2}, optimizer_params={'lr': 0.01, 'weight_decay': 0.01})
#trainer = ModelTrainer(X, y, model_params={'num_classes': 2}, optimizer_params={'lr': 0.01, 'alpha': 0.99})
trainer.train(80)
trainer.report()
