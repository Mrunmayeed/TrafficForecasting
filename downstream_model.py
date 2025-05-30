
import torch
import pandas as pd
import torch.nn as nn
import torch.nn.functional as F
import time
from torch.utils.data import DataLoader, TensorDataset
import logging
from concurrent.futures import ProcessPoolExecutor, as_completed
from torchinfo import summary


input_seq1=288
input_seq2=32
output_seq=12

class TwoInputRegressor(nn.Module):
    def __init__(self, input_seq1, input_seq2, output_seq):
        super(TwoInputRegressor, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=input_seq1, out_channels=128, kernel_size=5, padding=2)
        self.conv2 = nn.Conv1d(128, 32, kernel_size=5, padding=2)
        # self.conv3 = nn.Conv1d(64, 32, kernel_size=5, padding=2)

        # Output from conv2 has shape: [B, 32, 1] → flatten to B x (32*288)
        self.flatten_dim = 32 * 1
        self.time_comp = nn.Linear(self.flatten_dim, input_seq2)  # project to 24
        self.final1 = nn.Linear(input_seq2 * 2, input_seq2)
        self.final2 = nn.Linear(input_seq2, output_seq)

    def forward(self, x, z):
        # x: [B, 288] → [B, 1, 288]
        x = x.unsqueeze(2)
        x = F.relu(self.conv1(x))  # → [B, 128, 288]
        x = F.relu(self.conv2(x))  # → [B, 32, 288]
        # x = F.relu(self.conv3(x))  # → [B, 32, 288]
        x = x.view(x.size(0), -1)  # → [B, 32*288]
        x = F.relu(self.time_comp(x))  # → [B, 24]

        # z: [B, 24]
        xz = torch.cat((x, z), dim=1)  # → [B, 48]
        xz = self.final1(xz)
        xz = self.final2(xz)
        return xz


def train_thread_func(args, node, train_encoding, test_encoding):
    start = time.time()
    dm = DownstreamModel(input_seq1=args['enc_in_channels'], input_seq2=args['enc_out_channels'], output_seq=args['prediction'])

    x_train = torch.tensor(train_encoding['x'], dtype=torch.float32)
    y_train = torch.tensor(train_encoding['y'], dtype=torch.float32)
    z_train = torch.tensor(train_encoding['z'], dtype=torch.float32)
    training_losses = dm.train(x_train, y_train, z_train, **args)

    x_test = torch.tensor(test_encoding['x'], dtype=torch.float32)
    y_test = torch.tensor(test_encoding['y'], dtype=torch.float32)
    z_test = torch.tensor(test_encoding['z'], dtype=torch.float32)
    loss, _, _ = dm.test(x_test, y_test, z_test)
    end = time.time()
    duration = end - start
    return node, loss, duration, training_losses


def train_test_each(train_encoding, test_encoding, args):

    loss_table = {}
    training_loss_table = {}
    num_nodes = len(train_encoding)
    # models = {}
    total_time = 0

    # for node in range(num_nodes):
    #     logging.info(f"Running Model for node {node}")
    #     loss,duration = train_thread_func(args,train_encoding[node], test_encoding[node])
    #     loss_table[node] = loss
    #     total_time+=duration

    with ProcessPoolExecutor(max_workers=30) as executor:
        futures = []
        for node in range(num_nodes):
            logging.info(f"Submitting model for node {node}")
            future = executor.submit(train_thread_func,args, node, train_encoding[node], test_encoding[node])
            futures.append(future)

        for future in as_completed(futures):
            node, loss, duration, training_losses = future.result()
            loss_table[node] = loss
            training_loss_table[node]=training_losses
            total_time += duration
            logging.info(f"Node {node} done. Loss: {loss:.4f} | Time: {duration:.2f}s")

    avg_loss = sum(loss_table.values())/228
    logging.info(f"Final Test Loss for Model: {avg_loss:.4f}| Time: {total_time:.2f}s")
    pd.DataFrame(loss_table.values()).to_csv(f"downstream_losses.csv")
    pd.DataFrame(training_loss_table.values()).to_csv(f"downstream_losses_train.csv")
    return loss_table

# class ModelBuilder:
#     def __init__(self, args):
#         self.input_seq1 = args['enc_in_channels']
#         self.input_seq2 = args['enc_out_channels']
#         self.output_seq = args['prediction']
#
#     def build(self):
#         dm = DownstreamModel(input_seq1=self.input_seq1, input_seq2=self.input_seq2, output_seq=self.output_seq)
#         return dm

class DownstreamModel:
    def __init__(self, **args):
        self.model = TwoInputRegressor(args['input_seq1'], args['input_seq2'], args['output_seq'])


    def train(self, X_train, y_train, Z_train, **args):
        optimizer = torch.optim.Adam(self.model.parameters(), lr=args['lr'])
        loss_fn = nn.MSELoss()
        self.loader = DataLoader(TensorDataset(X_train, y_train,Z_train), shuffle=True)
        losses= []

        # logging.info(f"Model Summary")
        # logging.info(summary(self.model, input_data=(X_train, Z_train)))
        for epoch in range(args['epoch']):
            self.model.train()
            total_loss = 0
            for xb, yb, zb in self.loader:
                optimizer.zero_grad()
                preds = self.model(xb, zb)
                loss = loss_fn(preds, yb)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            avg_loss = total_loss / len(self.loader)
            logging.info(f"Epoch {epoch + 1}, Loss: {avg_loss:.4f}")
            losses.append(avg_loss)
        return losses

    def test(self, X_test, y_test, Z_test):
        self.model.eval()
        loader = DataLoader(TensorDataset(X_test, y_test,Z_test))
        loss_fn = torch.nn.MSELoss()
        total_loss = 0
        preds_all = []
        targets_all = []
        with torch.no_grad():
            for xb, yb, zb in loader:
                preds = self.model(xb, zb)
                loss = loss_fn(preds, yb)
                total_loss += loss.item()
                preds_all.append(preds)
                targets_all.append(yb)

            # Concatenate all predictions and targets for later analysis if needed

        avg_loss = total_loss / len(loader)
        preds_all = torch.cat(preds_all, dim=0)
        targets_all = torch.cat(targets_all, dim=0)
        logging.info(f"Test Loss: {avg_loss:.4f}")
        return avg_loss, preds_all, targets_all
