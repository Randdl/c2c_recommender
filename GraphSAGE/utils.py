import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric as pyg
from torch_geometric.data import Data
from torch_geometric.data import DataLoader
from torch_geometric.utils import negative_sampling
import matplotlib.pyplot as plt
import numpy as np


# GraphSAGE model to generate node embeddings
class GNNStack(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers, dropout, emb=False):
        super(GNNStack, self).__init__()
        conv_model = pyg.nn.SAGEConv

        self.convs = nn.ModuleList()
        self.convs.append(conv_model(input_dim, hidden_dim))
        self.dropout = dropout
        self.num_layers = num_layers
        self.emb = emb

        # Create num_layers GraphSAGE convs
        assert (self.num_layers >= 1), 'Number of layers is not >=1'
        for l in range(self.num_layers - 1):
            self.convs.append(conv_model(hidden_dim, hidden_dim))

        # post-message-passing processing
        self.post_mp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim), nn.Dropout(self.dropout),
            nn.Linear(hidden_dim, output_dim))

    def forward(self, x, edge_index):
        for i in range(self.num_layers):
            x = self.convs[i](x, edge_index)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)

        x = self.post_mp(x)

        # Return final layer of embeddings if specified
        if self.emb:
            return x

        # Else return class probabilities
        return F.log_softmax(x, dim=1)

    def loss(self, pred, label):
        return F.nll_loss(pred, label)


# MLP model to classify the relationship between two nodes as positive or negative
class LinkPredictor(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers,
                 dropout):
        super(LinkPredictor, self).__init__()

        # Create linear layers
        self.lins = nn.ModuleList()
        self.lins.append(nn.Linear(in_channels, hidden_channels))
        for _ in range(num_layers - 2):
            self.lins.append(nn.Linear(hidden_channels, hidden_channels))
        self.lins.append(nn.Linear(hidden_channels, out_channels))

        self.dropout = dropout

    def reset_parameters(self):
        for lin in self.lins:
            lin.reset_parameters()

    def forward(self, x_i, x_j):
        # x_i and x_j are both of shape (E, D)
        x = x_i * x_j
        for lin in self.lins[:-1]:
            x = lin(x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.lins[-1](x)
        return torch.sigmoid(x)


# train an epoch
def train(model, link_predictor, emb, edge_index, pos_train_edge, batch_size, optimizer, scheduler):
    """
    Runs offline training for model, link_predictor and node embeddings given the message
    edges and supervision edges.
    :param model: Torch Graph model used for updating node embeddings based on message passing
    :param link_predictor: Torch model used for predicting whether edge exists or not
    :param emb: (N, d) Initial node embeddings for all N nodes in graph
    :param edge_index: (2, E) Edge index for all edges in the graph
    :param pos_train_edge: (PE, 2) Positive edges used for training supervision loss
    :param batch_size: Number of positive (and negative) supervision edges to sample per batch
    :param optimizer: Torch Optimizer to update model parameters
    :return: Average supervision loss over all positive (and correspondingly sampled negative) edges
    """
    model.train()
    link_predictor.train()

    train_losses = []

    for edge_id in DataLoader(range(pos_train_edge.shape[0]), batch_size, shuffle=True):
        optimizer.zero_grad()

        # Run message passing on the inital node embeddings to get updated embeddings
        node_emb = model(emb, edge_index)  # (N, d)

        # Predict the class probabilities on the batch of positive edges using link_predictor
        pos_edge = pos_train_edge[edge_id].T  # (2, B)
        pos_pred = link_predictor(node_emb[pos_edge[0]], node_emb[pos_edge[1]])  # (B, )

        # Sample negative edges (same number as number of positive edges) and predict class probabilities
        neg_edge = negative_sampling(edge_index, num_nodes=emb.shape[0],
                                     num_neg_samples=edge_id.shape[0], method='dense')  # (Ne,2)
        neg_pred = link_predictor(node_emb[neg_edge[0]], node_emb[neg_edge[1]])  # (Ne,)

        # Compute the corresponding negative log likelihood loss on the positive and negative edges
        # print(-torch.log(pos_pred + 1e-15).mean())
        # print(- torch.log(1 - neg_pred + 1e-15).mean())
        loss = -torch.log(pos_pred + 1e-15).mean() - torch.log(1 - neg_pred + 1e-15).mean()
        # scores = torch.cat([pos_pred, neg_pred]).view(-1)
        # labels = torch.cat([torch.ones(pos_pred.shape[0]), torch.zeros(neg_pred.shape[0])]).to(pos_pred.device)
        # loss = F.binary_cross_entropy(scores, labels)
        # alpha = 0.5
        # gamma = 1
        #
        # scores = torch.cat([pos_pred, neg_pred]).view(-1)
        # labels = torch.cat([torch.ones(pos_pred.shape[0]), torch.zeros(neg_pred.shape[0])]).to(pos_pred.device)
        # BCE_loss = F.binary_cross_entropy(scores, labels, reduction='none')
        # pt = torch.exp(-BCE_loss)  # prevents nans when probability 0
        # focal_loss = alpha * (1 - pt) ** gamma * BCE_loss
        # loss = focal_loss.mean()

        # Backpropagate and update parameters
        loss.backward()
        optimizer.step()
        scheduler.step()

        train_losses.append(loss.item())
    return sum(train_losses) / len(train_losses)


# test and output the link prediction accuracy
def test(model, predictor, emb, edge_index, split_edge, batch_size):
    """
    Evaluates graph model on validation and test edges
    :param model: Torch Graph model used for updating node embeddings based on message passing
    :param predictor: Torch model used for predicting whether edge exists or not
    :param emb: (N, d) Initial node embeddings for all N nodes in graph
    :param edge_index: (2, E) Edge index for all edges in the graph
    :param split_edge: Dictionary of (e, 2) edges for val pos/neg and test pos/neg edges
    :param batch_size: Number of positive (and negative) supervision edges to sample per batch
    :param evaluator: OGB evaluator to calculate hits @ k metric
    :return: hits @ k results
    """
    model.eval()
    predictor.eval()

    node_emb = model(emb, edge_index)

    # pos_valid_edge = split_edge['valid']['edge'].to(emb.device)
    # neg_valid_edge = split_edge['valid']['edge_neg'].to(emb.device)
    pos_test_edge = split_edge['test']['edge'].to(emb.device)
    neg_test_edge = split_edge['test']['edge_neg'].to(emb.device)

    pos_test_preds = []
    for perm in DataLoader(range(pos_test_edge.size(0)), batch_size):
        edge = pos_test_edge[perm].t()
        pos_test_preds += [predictor(node_emb[edge[0]], node_emb[edge[1]]).squeeze().cpu()]
    pos_test_pred = torch.cat(pos_test_preds, dim=0)
    # print(pos_test_pred)

    neg_test_preds = []
    for perm in DataLoader(range(neg_test_edge.size(0)), batch_size):
        edge = neg_test_edge[perm].t()
        neg_test_preds += [predictor(node_emb[edge[0]], node_emb[edge[1]]).squeeze().cpu()]
    neg_test_pred = torch.cat(neg_test_preds, dim=0)
    # print(neg_test_pred)

    pos_hits = (pos_test_pred[pos_test_pred > 0.5]).shape[0]
    neg_hits = (neg_test_pred[neg_test_pred < 0.5]).shape[0]
    # print('Positive:', pos_hits / pos_test_edge.size(0),
    #       'Negative:', neg_hits / neg_test_edge.size(0))
    return pos_hits / pos_test_edge.size(0), neg_hits / neg_test_edge.size(0)


# evaluate the output the top k recommendations
def evaluate(model, predictor, emb, edge_index, test_nodes, pair_nodes, batch_size):
    model.eval()
    predictor.eval()
    node_emb = model(emb, edge_index)
    result = {}
    for node in test_nodes:
        test_preds = []
        for perm in DataLoader(range(pair_nodes.size), batch_size):
            nodes1 = pair_nodes[perm]
            nodes0 = np.zeros(nodes1.size, dtype=int) + node
            # print(nodes1)
            # print(nodes0)
            test_preds += [predictor(node_emb[nodes1], node_emb[nodes0]).squeeze().cpu()]
        test_pred = torch.cat(test_preds, dim=0)
        # print(test_pred[torch.topk(test_pred, 50).indices])
        # result[node] = test_pred.detach()
        result[node] = torch.topk(test_pred, 50).indices
    return result


# read from the edge file
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


# generate recommendations based on purchase history
def predict_baseline(test_nodes, edge_index, train_indices):
    gt_nodes = {}
    for i in train_indices:
        i = int(i)
        node = edge_index[0][i]
        target = edge_index[1][i]
        if node not in test_nodes:
            continue
        if node not in gt_nodes.keys():
            gt_nodes[node] = [target]
        else:
            gt_nodes[node].append(target)
    return gt_nodes
