import os

import torch
import pandas as pd
import json
from collections import defaultdict

from torch_geometric.datasets import Planetoid
import torch_geometric.transforms as T
from torch_geometric.utils import train_test_split_edges
from node2vec import Node2VecClass
from data import RoadDataset, get_data
from graph_embedding import GraphEmbeddings
from downstream_model import DownstreamModel, train_test_each
from cnn_model import trainCNN,testCNN
import time


import logging

logging.basicConfig(
    filename='temp/training_node2vec.log',
    filemode='w',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

torch.manual_seed(12345)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# args = parse_args()
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

def flatten_embeddings(embeddings):
    X, Y, Z = [], [], []
    for node in embeddings:
        # print(f"Running CNN for node {node}")

        embed = embeddings[node]
        X.extend(embed['x'])
        Y.extend(embed['y'])
        Z.extend(embed['z'])

    x = torch.tensor(X, dtype=torch.float32)
    y = torch.tensor(Y, dtype=torch.float32)
    z = torch.tensor(Z, dtype=torch.float32)

    return x, y, z

def save_encodings(name, encodings):
    with open(f"{name}_encodings.json", "w") as f:
        json.dump(encodings, f)

def get_encodings(name):
    with open(f"{name}_encodings.json", "r") as f:
        encodings = json.load(f)
    return encodings


#TODO: Work on dataset


# BASELINE


if __name__ == "__main__":

    rd = RoadDataset(window_size=args['enc_in_channels'], pred_size=args['prediction'], stride=args['stride'])
    train_dataset, test_dataset = rd.train_dataset, rd.test_dataset


    # GRAPH ENCODING
    # start = time.time()
    # logging.info("Training Graph Embeddings")
    # ge = GraphEmbeddings(args, train_dataset)
    # ge.train()
    # train_encodings = ge.encodings(train_dataset)
    # test_encodings = ge.encodings(test_dataset)
    # # save_encodings("train_encodings", train_encodings)
    # # save_encodings("test_encodings", test_encodings)
    # end = time.time()
    # logging.info(f"Time taken for Graph Embeddings: {end-start}")

    # NODE_2_VEC
    start = time.time()
    logging.info("Training Node2Vec Embeddings")
    nv = Node2VecClass(args, train_dataset)
    nv.train()
    train_encodings = nv.encodings(train_dataset)
    test_encodings = nv.encodings(test_dataset)
    # save_encodings("train_encodings", train_encodings)
    # save_encodings("test_encodings", test_encodings)
    end = time.time()
    logging.info(f"Time taken for Graph Embeddings: {end-start}")


    start = time.time()
    # train_encodings = get_encodings("train_encodings")
    # test_encodings = get_encodings("test_encodings")
    losses = train_test_each(train_encodings, test_encodings, args)
    end = time.time()
    logging.info(f"Time taken for Training and Testing: {end-start}")



## SharedDownstream Model
    # logging.info("")
# X_train, Y_train, Z_train = flatten_embeddings(train_encodings)
# X_test, Y_test, Z_test = flatten_embeddings(test_encodings)
# dm = DownstreamModel(input_seq1=args['enc_in_channels'], output_seq=args['prediction'], input_seq2=args['enc_out_channels'])
# dm.train(X_train,Y_train, Z_train, epoch=args['epoch'], lr=args['lr'])
# dm.test(X_test,Y_test, Z_test)



