import os

import numpy as np
import torch
from sklearn.model_selection import train_test_split

from gat.converter import convert_to_graph
from gat.model import GAT
from gat.preprocesser import preprocess_df, preprocess_X, preprocess_y

TEST_SIZE = 0.25
RANDOM_STATE = 42

class Config:
    optimizer = torch.optim.AdamW
    lr = 0.0425
    weight_decay = 0.0004807430799298252
    epochs = 30
    patience = 5
    hidden_dim = 30
    dropout = 0.425

def split_data():
    print("Start feature engineering...")
    df = preprocess_df()
    print("Feature engineering done.")
    print("Start preprocessing...")
    X = preprocess_X(df)
    y = preprocess_y(df)
    print("Preprocessing done.")
    print("Start splitting data...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, stratify=y, random_state=RANDOM_STATE)
    print("Splitting data done.")
    print("Start converting to graph...")
    train_data = convert_to_graph(X_train, y_train)
    test_data = convert_to_graph(X_test, y_test)
    print("Converting to graph done.")
    return train_data, test_data, y_train

def initialize_model(train_data, y_train):
    config = Config()
    model = GAT(
        optimizer=config.optimizer,
        num_features=train_data.num_features,
        num_classes=len(np.unique(y_train)),
        weight_decay=config.weight_decay,
        dropout=config.dropout,
        hidden_dim=config.hidden_dim,
        epochs=config.epochs,
        lr=config.lr,
        patience=config.patience
    )
    return model

if not os.path.exists("./results"):
    os.makedirs("./results")
train_data, test_data, y_train = split_data()
model = initialize_model(train_data, y_train)
model.train_model(train_data, test_data)
