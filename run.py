# Importing necessary libraries and modules for the project
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split  # Splitting data
from torch_geometric.data import Data  # PyTorch Geometric data handling

from gat.encoder import ip_encoder, number_normalizer, string_encoder
from gat.load_data import load_data  # Custom function to load data
from gat.model import GAT  # Graph Attention Network model from the gat package

# Load data using a custom function from the gat package
X, y, header = load_data()

# Mapping for converting y string values to integers
mapping = {'False': 0, 'True': 1}
# Apply the mapping and directly convert to integer
y_int = y.map(mapping).astype(int)

# Now you can proceed to split the data
# stratify makes sure that the distribution of classes is similar in both train and test sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y_int, test_size=0.25, stratify=y_int, random_state=42)


def convert_to_graph(X, y):

    encoder_map = {
        'ip_source': 'ip classes to int',
        'ip_destination': 'ip classes to int',
        'source_pod_label': 'hash',
        'destination_pod_label': 'hash',
        'source_namespace_label': 'hash',
        'destination_namespace_label': 'hash',
        'source_port_label': 'string to int',
        'destination_port_label': 'string to int',
        'ack_flag': 'stringbool to int',
        'psh_flag': 'stringbool to int'
    }   
    
    
    X = ip_encoder(X, 'ip_source')
    X = ip_encoder(X, 'ip_destination')
    X = string_encoder(X, 'source_pod_label')
    X = string_encoder(X, 'destination_pod_label')
    X = string_encoder(X, 'source_namespace_label')
    X = string_encoder(X, 'destination_namespace_label')
    X = number_normalizer(X, 'source_port_label')
    X = number_normalizer(X, 'destination_port_label')


    # Mapping for converting y string values to integers
    mapping = {'False': 0, 'True': 1}
    X['ack_flag'] = X['ack_flag'].map(mapping).astype(int)
    X['psh_flag'] = X['psh_flag'].map(mapping).astype(int)
    
    # Create a DataFrame with the correct column names
    X = X.apply(pd.to_numeric, errors='coerce').fillna(0)

    # Convert the DataFrame to a tensor
    X_tensor = torch.tensor(X.values, dtype=torch.float)

    # Convert the label data to a tensor
    y_tensor = torch.tensor(y.values, dtype=torch.long)

    # Total number of nodes calculated from the number of rows in the tensor
    num_nodes = len(X_tensor)

    # Initialize masks as boolean tensors for training, testing, and validation
    masks = [torch.zeros(num_nodes, dtype=torch.bool) for _ in range(3)]
    limits = [int(0.8 * num_nodes), int(0.9 * num_nodes), num_nodes]

    # Assign True up to the specified limit for each mask and shuffle
    for mask, limit in zip(masks, limits):
        mask[:limit] = True
        np.random.shuffle(mask.numpy())

    # Defines a simple bi-directional connection between two consecutive nodes
    edge_index = torch.tensor([[i, i+1] for i in range(num_nodes-1) for _ in (0, 1)], dtype=torch.long).t().contiguous()

    # Return a Data object containing node features, edge connections, labels, and masks
    return Data(x=X_tensor, edge_index=edge_index, y=y_tensor,
                train_mask=masks[0], test_mask=masks[1], val_mask=masks[2])

    # Note: This function assumes that X is a DataFrame and y is a Series or similar.


# Convert both training and testing data to graph format
train_data = convert_to_graph(X_train, y_train)
test_data = convert_to_graph(X_test, y_test)

model = GAT(torch.optim.Adam, num_features=train_data.num_features, num_classes=len(np.unique(y_train)))

# Train model for a specified number of epochs
for epoch in range(20):
    loss = model.train_model(train_data)
    print(f'Epoch {epoch+1}, Loss: {loss:.4f}')

# Test the model and capture predictions
accuracy, pred = model.test_model(test_data)

# Compute confusion matrix and additional metrics
all_labels = np.unique(y_int)  # Ensure all_labels is an integer array
conf_matrix = confusion_matrix(test_data.y.numpy(), pred.numpy(), labels=all_labels)
precision = precision_score(test_data.y.numpy(), pred.numpy(), average='weighted', labels=all_labels)
recall = recall_score(test_data.y.numpy(), pred.numpy(), average='weighted', labels=all_labels)
f1 = f1_score(test_data.y.numpy(), pred.numpy(), average='weighted', labels=all_labels)
test_accuracy = accuracy_score(test_data.y.numpy(), pred.numpy())

# Print all computed metrics for model evaluation
print("Confusion Matrix:\n", conf_matrix)
print(f"Test Accuracy: {test_accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1 Score: {f1:.4f}")
