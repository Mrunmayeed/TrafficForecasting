import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from tqdm import tqdm
import json
from collections import defaultdict

# --- Shared CNN Model ---
class SharedCNNRegressor(nn.Module):
    def __init__(self, input_seq, output_seq):
        super(SharedCNNRegressor, self).__init__()
        self.conv1 = nn.Conv1d(input_seq, 20, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(20, 16, kernel_size=3, padding=1)
        self.fc = nn.Linear(16 , output_seq)

    def forward(self, x):
        x = x.unsqueeze(-1)
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = x.view(x.size(0), -1)
        return self.fc(x)

# def create_shared_dataset(X_all_nodes, y_all_nodes, input_seq=24, output_seq=12):
#     num_nodes, total_time, num_features = X_all_nodes.shape
#     X_list, y_list = [], []
#
#     for node in range(num_nodes):
#         for t in range(total_time - input_seq - output_seq + 1):
#             X_list.append(X_all_nodes[node, t:t+input_seq])
#             y_list.append(y_all_nodes[node, t+input_seq:t+input_seq+output_seq])
#
#     X_all = torch.tensor(np.array(X_list), dtype=torch.float32)
#     y_all = torch.tensor(np.array(y_list), dtype=torch.float32)
#     return X_all, y_all

# --- Training ---
# def train_shared_cnn_model(X_all, y_all, input_seq=24, output_seq=12, batch_size=32, lr=0.01, epochs=10):
#     # input_channels = X_all.shape[1]
#     model = SharedCNNRegressor( input_seq, output_seq)
#     optimizer = torch.optim.Adam(model.parameters(), lr=lr)
#     loss_fn = nn.MSELoss()
#     loader = DataLoader(TensorDataset(X_all, y_all), batch_size=batch_size, shuffle=True)
#     losses= []
#     for epoch in range(epochs):
#         model.train()
#         total_loss = 0
#         for xb, yb in loader:
#             optimizer.zero_grad()
#             preds = model(xb)
#             loss = loss_fn(preds, yb)
#             loss.backward()
#             optimizer.step()
#             total_loss += loss.item()
#         # if epoch%2 == 0:
#         # print(f"Epoch {epoch+1}, Loss: {total_loss:.4f}")
#         losses.append(total_loss)
#     return model, losses

def test_shared_cnn_model(model, X_test, y_test, batch_size=32):
    model.eval()
    loader = DataLoader(TensorDataset(X_test, y_test), batch_size=batch_size)
    loss_fn = torch.nn.MSELoss()
    total_loss = 0
    preds_all = []
    targets_all = []
    with torch.no_grad():
        for xb, yb in loader:
            preds = model(xb)
            loss = loss_fn(preds, yb)
            total_loss += loss.item()
            preds_all.append(preds)
            targets_all.append(yb)

    # Concatenate all predictions and targets for later analysis if needed
    preds_all = torch.cat(preds_all, dim=0)
    targets_all = torch.cat(targets_all, dim=0)

    print(f"Test Loss: {total_loss:.4f}")
    return total_loss, preds_all, targets_all


def train_shared_cnn_model(X_all, y_all, input_seq=24, output_seq=12, batch_size=32, lr=0.01, epochs=20):
    # input_channels = X_all.shape[1]
    model = SharedCNNRegressor( input_seq, output_seq)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()
    loader = DataLoader(TensorDataset(X_all, y_all), batch_size=batch_size, shuffle=True)
    losses= []
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for xb, yb in loader:
            optimizer.zero_grad()
            preds = model(xb)
            loss = loss_fn(preds, yb)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch+1}, Loss: {total_loss:.4f}")
        losses.append(total_loss)
    return model, losses


def trainCNN(embeddings, args, seperate_nodes=False):
    X,Y,Z=[],[],[]
    for node in embeddings:
        # print(f"Running CNN for node {node}")

        embed = embeddings[node]
        input = np.hstack((np.array(embed['x']),np.array(embed['z'])))
        X.extend(input)
        Y.extend(embed['y'])
        # loss_table[node] = losses

    x = torch.tensor(X, dtype=torch.float32)
    y = torch.tensor(Y, dtype=torch.float32)
    model,_ = train_shared_cnn_model(x, y, args['enc_out_channels']+args['enc_in_channels'], args['prediction'], epochs=args['epoch'])


    if seperate_nodes:
        loss_table = {}
        for node in embeddings:
            print(f"Running CNN for node {node}")
            embed = embeddings[node]
            x= torch.tensor(embed['x'], dtype=torch.float32)
            y= torch.tensor(embed['y'], dtype=torch.float32)
            _, losses = train_shared_cnn_model(x, y, args['enc_out_channels'], args['prediction'])
            loss_table[node] = losses

        pd.DataFrame(loss_table).T.to_csv(f"cnn_losses.csv")

    return model


def testCNN(model, embeddings, seperate_nodes=False):
    X, Y, Z = [], [], []
    for node in embeddings:
        # print(f"Running CNN for node {node}")

        embed = embeddings[node]
        input = np.hstack((np.array(embed['x']), np.array(embed['z'])))
        X.extend(input)
        Y.extend(embed['y'])
        # loss_table[node] = losses

    x = torch.tensor(X, dtype=torch.float32)
    y = torch.tensor(Y, dtype=torch.float32)
    test_shared_cnn_model(model, x, y)

    if seperate_nodes:
        loss_table = {}
        for node in embeddings:
            print(f"Running CNN for node {node}")
            embed = embeddings[node]
            x = torch.tensor(embed['x'], dtype=torch.float32)
            y = torch.tensor(embed['y'], dtype=torch.float32)
            _, losses = test_shared_cnn_model(model, x, y)
            loss_table[node] = losses

        # pd.DataFrame(loss_table).T.to_csv(f"cnn_losses.csv")