
import pandas as pd
import numpy as np
import torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch_geometric.data import Data


def scale_graph_data(data_list, scaler):
    scaled_list = []
    for data in data_list:
        x_scaled = torch.tensor(scaler.transform(data.x.T.numpy()).T, dtype=torch.float32)
        y_scaled = torch.tensor(scaler.transform(data.y.T.numpy()).T, dtype=torch.float32)
        scaled_list.append(Data(x=x_scaled, edge_index=data.edge_index, y=y_scaled))
    return scaled_list

class RoadDataset:
    def __init__(self, window_size, pred_size, stride):
        self.window_size = window_size
        self.pred_size = pred_size
        self.stride = stride
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.create_dataset()
        
    def get_raw_data(self):

        folder_path = 'dataset/'
        self.node_features = pd.read_csv(folder_path + 'PeMSD7_V_228.csv', header=None).values
        self.adj_matrix = pd.read_csv(folder_path + 'PeMSD7_W_228.csv', header=None).values



    def adjacency_to_edge_index(self, adj_matrix):
        self.adj_matrix = self.adj_matrix<16862.77
        edge_index = np.nonzero(adj_matrix)  # Find non-zero entries (edges) in the adjacency matrix
        edge_weight = adj_matrix[edge_index]  # Get the weights of these edges
        edge_index = np.vstack(edge_index)  # Convert to the correct format for PyTorch Geometric
        self.edge_index= torch.tensor(edge_index, dtype=torch.long)
        self.edge_index = self.edge_index.to(self.device)
        self.edge_weight = torch.tensor(edge_weight, dtype=torch.float)
        self.edge_weight = self.edge_weight.to(self.device)

    def scale_dataset(self, dataset, train=False):

        if train:
            self.scaler = StandardScaler()
            # Stack all x and y tensors for scaler fitting
            x_train_all = torch.cat([data.x.T for data in dataset], dim=0).numpy()  # shape: [#windows * window_size, N]
            y_train_all = torch.cat([data.y.T for data in dataset], dim=0).numpy()  # shape: [#windows * pred_size, N]
            self.scaler.fit(np.vstack([x_train_all, y_train_all]))

        scaled_list = []
        for data in dataset:
            x_scaled = torch.tensor(self.scaler.transform(data.x.T.numpy()).T, dtype=torch.float32)
            x_scaled = x_scaled.to(self.device)
            y_scaled = torch.tensor(self.scaler.transform(data.y.T.numpy()).T, dtype=torch.float32)
            y_scaled = y_scaled.to(self.device)
            scaled_list.append(Data(x=x_scaled, edge_index=data.edge_index, y=y_scaled))

        return scaled_list

    def scale_test_data(self, test_dataset):
        scaled_list = []
        for data in test_dataset:
            x_scaled = torch.tensor(self.scaler.transform(data.x.T.numpy()).T, dtype=torch.float32)
            y_scaled = torch.tensor(self.scaler.transform(data.y.T.numpy()).T, dtype=torch.float32)
            scaled_list.append(Data(x=x_scaled, edge_index=data.edge_index, y=y_scaled))

        self.test_dataset = scaled_list


    def create_windowed_graphs(self, node_features):
        """
        X_all: shape (T, N)
        Y_all: shape (T, N)
        Returns: list of PyG Data objects, one per time window
        """
        data_list = []
        T, N = node_features.shape

        for t in range(0, T - self.window_size - self.pred_size+1, self.stride):
            e = t + self.window_size
            node_features_window = node_features[t:e]  # shape (W, N)
            x = torch.tensor(node_features_window.T, dtype=torch.float32)  # (N, W)
            node_pred_window = node_features[e: e + self.pred_size]
            y = torch.tensor(node_pred_window.T, dtype=torch.float32)  # node labels at t + W

            data = Data(x=x, edge_index=self.edge_index, y=y)
            data_list.append(data)
        return data_list

    def split_data(self, dataset, test_ratio=0.2):
        train_list, test_list = train_test_split(dataset, test_size=test_ratio, random_state=42)
        return train_list, test_list

    def create_dataset(self):
        self.get_raw_data()
        self.adjacency_to_edge_index(self.adj_matrix)
        dataset = self.create_windowed_graphs(self.node_features)
        train_dataset, test_dataset = self.split_data(dataset)
        self.train_dataset = self.scale_dataset(train_dataset, train=True)
        self.test_dataset = self.scale_dataset(test_dataset)


def get_data():
    # Load
    train_dataset = torch.load("temp/train_data.pt", weights_only=False)
    test_dataset = torch.load("temp/test_data.pt", weights_only=False)

    return train_dataset,test_dataset

if __name__ == "__main__":
    rd = RoadDataset(window_size=288, pred_size=12, stride=12)
    train_dataset,test_dataset = rd.train_dataset,rd.test_dataset
    # torch.save(train_dataset, "temp/train_data.pt")
    # torch.save(test_dataset, "temp/test_data.pt")
    print("Updated train and test datasets")

