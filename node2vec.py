import torch
from torch.optim import Adam
import json
from collections import defaultdict
# from embedding_model import DeepVGAE
# from torchinfo import summary
from torch_geometric.nn import Node2Vec


from data import RoadDataset

import logging

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


args = {
    "enc_in_channels": 288 ,
    "enc_hidden_channels": 128,
    "enc_out_channels": 32,
    "stride":12,
    "lr": 0.001,
    "epoch": 10,
    "dataset": "Road",
    "prediction":12,
    "device": device,
    "input_dim": 288,
    "hidden_dim": 128,
    "output_dim": 12,
}

class Node2VecClass:
    def __init__(self, args, dataset):
        self.args = args
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.dataset = dataset

    def get_dataset(self):

        return None



    def train(self):


        self.model = Node2Vec(
            edge_index=self.dataset[0].edge_index,
            embedding_dim=32,
            walk_length=12,
            context_size=12
        )
        self.model = self.model.to(self.device)
        self.loader = self.model.loader(batch_size=128, shuffle=True, num_workers=0)
        self.optimizer = Adam(self.model.parameters(), lr=self.args['lr'])

        self.model.train()
        for epoch in range(self.args['epoch']):
            total_loss = 0
            for pos_rw, neg_rw in self.loader:
                self.optimizer.zero_grad()
                loss = self.model.loss(pos_rw.to(self.device), neg_rw.to(self.device))
                loss.backward()
                self.optimizer.step()
                total_loss += loss.item()
            print(f"Epoch {epoch:03d}, Loss: {total_loss:.4f}")

    def encodings(self, dataset):
        self.model.eval()
        all_node_embeddings = defaultdict(lambda: {"x": [], "y": [], "z":[]})
        with torch.no_grad():
            embeddings_tensor = self.model().cpu().numpy()

        for t, data in enumerate(dataset):
            z = embeddings_tensor
            y = data.y.cpu().numpy()
            x = data.x.cpu().numpy()

            for node_id, embedding in enumerate(z):
                # if node_id not in all_node_embeddings:
                #     all_node_embeddings[node_id] = [{"x": [], "y": []}]
                all_node_embeddings[node_id]["z"].append(embedding.tolist())
                all_node_embeddings[node_id]["y"].append(y[node_id].tolist())
                all_node_embeddings[node_id]["x"].append(x[node_id].tolist())

        return all_node_embeddings


if __name__ == "__main__":

    rd = RoadDataset(window_size=args['enc_in_channels'], pred_size=args['prediction'], stride=args['stride'])
    train_dataset, test_dataset = rd.train_dataset, rd.test_dataset

    nv = Node2VecClass(args, train_dataset)
    nv.train()
    train_encodings = nv.encodings(train_dataset)
    test_encodings = nv.encodings(test_dataset)
    print('Printing train encodings:')