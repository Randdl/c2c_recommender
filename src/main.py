# import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import roc_auc_score, roc_curve, classification_report
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.utils import to_dense_adj
from torch_sparse import SparseTensor

from graphsage import GraphSAGE


def read_from_txt(file):
    list_from = []
    list_to = []
    list_weights = []
    with open(file) as inp:
        for line in inp.readlines():
            if ' ' in line:
                node_from, node_to, weight = line.split(' ', 2)
                list_from.append(int(node_from))
                list_to.append(int(node_to))
                list_weights.append(int(weight))
    return [list_from, list_to, list_weights]


def main():
    device = f'cuda' if torch.cuda.is_available() else 'cpu'
    device = torch.device(device)
    num_buyers = 8922
    num_items = 3119
    num_sellers = 4056
    num_nodes = num_buyers + num_items

    edge_list = read_from_txt('data/buyer_item.txt')
    edge_index = torch.tensor([edge_list[0], [i + num_buyers for i in edge_list[1]]])
    buyers = torch.cat([F.one_hot(torch.arange(0, num_buyers)), torch.zeros(num_buyers)[:, None]], dim=-1)
    items = torch.cat([F.one_hot(torch.arange(0, num_items), num_classes=num_buyers), torch.zeros(num_items)[:, None]],
                      dim=-1)
    x = torch.cat([buyers, items], dim=0)

    adj = SparseTensor(row=edge_index[0], col=edge_index[1], sparse_sizes=(num_nodes, num_nodes))
    edges = adj.to_dense()
    y = edges.clone()
    edges = edges == 1
    adj_t = adj.t()

    adj_dense = to_dense_adj(edge_index=edge_index)

    dataset = Data(x=x, edge_index=edge_index, adj_t=adj_t)
    print(dataset)
    data = dataset
    train_idx = list(range(8000)) + list(range(8922, 8922+2000))

    dataloader = DataLoader([dataset], batch_size=32)

    lr = 1e-3
    epochs = 20
    hidden_dim = 75
    evaluator = None
    criterion = nn.BCEWithLogitsLoss()

    model = GraphSAGE(in_dim=data.num_node_features,
                      hidden_dim=hidden_dim,
                      out_dim=10)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    for epoch in range(1, 1 + epochs):
        model.train()

        optimizer.zero_grad()
        # out = model(data)[train_idx]
        out = model(data)
        # print(out.shape)
        all_pairs = torch.mm(out, out.t())
        # print(all_pairs.shape)
        # pred =
        # print(edges)
        scores = all_pairs

        # scores = all_pairs[data.adj_t]
        # loss = criterion(scores, data.y.squeeze(1)[train_idx].float())
        loss = criterion(scores.squeeze(1).float(), y.squeeze(1).float())
        print(loss)
        loss.backward()
        optimizer.step()

        # result = test(model, data, split_idx, evaluator)


if __name__ == '__main__':
    main()
