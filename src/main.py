import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import roc_auc_score, roc_curve, classification_report
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader

from graphsage import GraphSAGE


def main():
    device = f'cuda' if torch.cuda.is_available() else 'cpu'
    device = torch.device(device)
    dataset = None
    data = dataset
    train_idx = []

    dataloader = DataLoader([dataset], batch_size=32)

    lr = 1e-4
    epochs = 50
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
        scores = all_pairs[edges.T]
        loss = criterion(scores, data.y.squeeze(1)[train_idx].float())
        loss.backward()
        optimizer.step()

        # result = test(model, data, split_idx, evaluator)


if __name__ == '__main__':
    main()