import numpy as np
import torch
from sklearn.neighbors import kneighbors_graph

def create_knn_graph(X, k=5):
    A = kneighbors_graph(X, n_neighbors=k, mode='connectivity', include_self=True)
    A = A.tocoo()  # Convert to COO format
    row = A.row.astype(np.int64)
    col = A.col.astype(np.int64)
    edge_index = np.vstack([row, col])
    return torch.tensor(edge_index, dtype=torch.long)
