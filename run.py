import matplotlib.pyplot as plt
from gat.load_data import load_data, split_data
from gat.graph_utils import create_knn_graph
from gat.model import initialize_model, GCN
from gat.evaluation import evaluate_model
from gat.converter import label_converter
from torch_geometric.data import Data
from pathlib import Path
import torch

EPOCHS = 150
FILE_PATH = './data/traces.csv'

def main():
    df = load_data(Path(FILE_PATH))

    df['is_anomaly'] = df['is_anomaly'].replace({'True': 1, 'False': 0}).astype(int)
    df = label_converter(df)

    X = df[['diversity_index']]
    y = df['is_anomaly']

    X_train, X_test, y_train, y_test = split_data(X, y)

    train_edge_index = create_knn_graph(X_train)
    test_edge_index = create_knn_graph(X_test)

    train_data = Data(x=torch.tensor(X_train.values, dtype=torch.float), 
                      edge_index=train_edge_index, 
                      y=torch.tensor(y_train.values, dtype=torch.long))

    test_data = Data(x=torch.tensor(X_test.values, dtype=torch.float), 
                     edge_index=test_edge_index, 
                     y=torch.tensor(y_test.values, dtype=torch.long))

    model, optimizer, loss_func = initialize_model(train_data)

    # Training loop
    loss_values = []
    for epoch in range(EPOCHS):
        model.train()
        optimizer.zero_grad()
        out = model(train_data)
        loss = loss_func(out, train_data.y)
        loss.backward()
        optimizer.step()
        loss_values.append(loss.item())
        print(f'Epoch {epoch+1}, Loss: {loss:.4f}')

    # Plotting the training loss
    plt.figure(figsize=(10, 5))
    plt.plot(loss_values, label='Training Loss')
    plt.title('Training Loss Over Epochs')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig('loss_graph.png')

    evaluate_model(model, test_data)

if __name__ == "__main__":
    main()
