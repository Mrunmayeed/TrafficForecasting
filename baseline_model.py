import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv

from data import RoadDataset, get_data
import logging

#
class SimpleGCN(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(SimpleGCN, self).__init__()
        self.gcn1 = GCNConv(input_dim, hidden_dim)
        self.gcn2 = GCNConv(hidden_dim, output_dim)

    def forward(self, x, edge_index):
        x = F.relu(self.gcn1(x, edge_index))
        x = self.gcn2(x, edge_index)
        return x


class BaselineModel:
    def __init__(self, **args):
        self.model = SimpleGCN(args['input_dim'], args['hidden_dim'], args['output_dim'])

    def train(self, dataset, epochs=5, lr=0.01):
        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        loss_fn = nn.MSELoss()

        for epoch in range(epochs):
            self.model.train()
            total_loss = 0
            for data in dataset:
                optimizer.zero_grad()
                out = self.model(data.x, data.edge_index)
                loss = loss_fn(out, data.y)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            logging.info(f"Epoch {epoch+1}, Loss: {total_loss:.4f}")

    def test(self, dataset):

        self.model.eval()
        loss_fn = torch.nn.MSELoss()
        total_loss = 0

        with torch.no_grad():
            for data in dataset:
                out = self.model(data.x, data.edge_index)  # Shape: [N, pred_size]
                loss = loss_fn(out, data.y)
                total_loss += loss.item()

        logging.info(f"Final Test Loss for Baseline: {total_loss:.4f}")

    # return total_loss

# T = 200         # total time steps
# N = 20          # number of nodes

if __name__ == "__main__":
    args = {
        "input_dim": 288,
        "hidden_dim": 128,
        "output_dim": 12,
        "stride":12,
        "lr": 0.001,
        "epochs": 5,
        "device": "cuda:0" if torch.cuda.is_available() else "cpu"
    }


    # Prepare windowed graph dataset
    rd = RoadDataset(window_size=args['input_dim'], pred_size=args['output_dim'], stride=args['stride'])
    train_dataset,test_dataset = rd.train_dataset,rd.test_dataset
    # Initialize model and train
    bm = BaselineModel(**args)
    bm.train(train_dataset)
    bm.test(test_dataset)
