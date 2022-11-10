import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import roc_auc_score, roc_curve, classification_report
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.utils import to_dense_adj

from graphsage import GraphSAGE


def read_from_txt(file):
    list_from = []
    list_to = []
    list_weights = []
    with open(file) as inp:
        for line in inp.readlines():
            if ' ' in line:
                node_from, node_to, weight = line.split(' ', 2)
                list_from.append(node_from)
                list_to.append(node_to)
                list_weights.append(weight)
    return [list_from, list_to, list_weights]


def main():
    device = f'cuda' if torch.cuda.is_available() else 'cpu'
    device = torch.device(device)
    num_buyers = 8922
    num_items = 3119
    num_sellers = 4056

    edge_list = read_from_txt('data/buyer_item.txt')
    edge_index = torch.tensor([edge_list[0], [i + num_buyers for i in edge_list[1]]])
    buyers = torch.cat([F.one_hot(torch.arange(0, num_buyers)), torch.zeros(num_buyers)[:, None]], dim=-1)
    items = torch.cat([F.one_hot(torch.arange(0, num_items), num_classes=num_buyers), torch.zeros(num_items)[:, None]],
                      dim=-1)
    x = torch.cat([buyers, items], dim=0)

    adj_dense = to_dense_adj(edge_index=edge_index)

    dataset = Data(x=x, edge_index=edge_index)
    print(dataset)
    data = dataset
    train_idx = []

    dataloader = DataLoader([dataset], batch_size=32)

    lr = 1e-4
    epochs = 2
    hidden_dim = 75
    evaluator = None
    criterion = nn.BCEWithLogitsLoss()

    model = GraphSAGE(in_dim=data.num_node_features,
                      hidden_dim=hidden_dim,
                      out_dim=dataset.num_classes)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    for epoch in range(1, 1 + epochs):
        model.train()

        optimizer.zero_grad()
        out = model(data)[train_idx]
        all_pairs = torch.mm(out, out.t())
        scores = all_pairs
        # scores = all_pairs[edges.T]
        loss = criterion(scores, data.y.squeeze(1)[train_idx].float())
        loss.backward()
        optimizer.step()

        # result = test(model, data, split_idx, evaluator)


if __name__ == '__main__':
    main()
