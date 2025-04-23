import torch
from torch.optim import Adam
import json
from collections import defaultdict
from embedding_model import DeepVGAE
from torchinfo import summary

import logging


class GraphEmbeddings:

    def __init__(self, args, dataset):
        self.device = args['device']
        self.args = args
        self.model = DeepVGAE(args).to(self.device)
        self.optimizer = Adam(self.model.parameters(), lr=args['lr'])
        self.dataset = dataset
        # logging.info(f"Summary:")
        # logging.info(summary(self.model, input_data=(dataset[0].x,dataset[0].edge_index)))

    def train(self):

        for epoch in range(self.args['epoch']):
            self.model.train()
            total_loss = 0
            # loss = model.loss(data.x, data.train_pos_edge_index)

            for i,data in enumerate(self.dataset):
                data = data.to(self.device)
                self.optimizer.zero_grad()
                z = self.model.encode(data.x, data.edge_index)
                loss = self.model.recon_loss(z, data.edge_index)
                loss = loss + (1 / data.num_nodes) * self.model.kl_loss()
                loss.backward()
                self.optimizer.step()
                total_loss += loss.item()
                # if i%1000 == 0:
                    # print(f'Epoch {epoch}, Batch {i}, Loss: {loss.item():.4f}')

            # print(f'Epoch {epoch} for Embeddings, Loss: {total_loss:.4f}')

    def encodings(self,dataset):

        self.model.eval()
        all_node_embeddings = defaultdict(lambda: {"x": [], "y": [], "z":[]})

        # model.eval()
        # all_node_embeddings = defaultdict(lambda: {"x": [], "y": None})

        with torch.no_grad():
            for t, data in enumerate(dataset):
                z = self.model.encode(data.x, data.edge_index).cpu().numpy()  # shape: (N, latent_dim)
                y = data.y.cpu().numpy()
                x = data.x.cpu().numpy()

                for node_id, embedding in enumerate(z):
                    # if node_id not in all_node_embeddings:
                    #     all_node_embeddings[node_id] = [{"x": [], "y": []}]
                    all_node_embeddings[node_id]["z"].append(embedding.tolist())
                    all_node_embeddings[node_id]["y"].append(y[node_id].tolist())
                    all_node_embeddings[node_id]["x"].append(x[node_id].tolist())

        # self.all_node_embeddings = all_node_embeddings
        self.write_embeddings(all_node_embeddings)
        return all_node_embeddings


    def write_embeddings(self, all_node_embeddings):
        with open(f"temp/all_node_embeddings", "w") as file:
            json.dump(dict(all_node_embeddings), file, indent=2)