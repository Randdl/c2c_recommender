import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric as pyg
from torch.optim.lr_scheduler import MultiStepLR
from torch_geometric.data import Data
from torch_geometric.data import DataLoader
from torch_geometric.utils import negative_sampling
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm


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

    # pos_valid_preds = []
    # for perm in DataLoader(range(pos_valid_edge.size(0)), batch_size):
    #     edge = pos_valid_edge[perm].t()
    #     pos_valid_preds += [predictor(node_emb[edge[0]], node_emb[edge[1]]).squeeze().cpu()]
    # pos_valid_pred = torch.cat(pos_valid_preds, dim=0)
    #
    # neg_valid_preds = []
    # for perm in DataLoader(range(neg_valid_edge.size(0)), batch_size):
    #     edge = neg_valid_edge[perm].t()
    #     neg_valid_preds += [predictor(node_emb[edge[0]], node_emb[edge[1]]).squeeze().cpu()]
    # neg_valid_pred = torch.cat(neg_valid_preds, dim=0)

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

    # results = {}
    # for K in [20, 50, 100]:
    #     evaluator.K = K
    #     valid_hits = evaluator.eval({
    #         'y_pred_pos': pos_valid_pred,
    #         'y_pred_neg': neg_valid_pred,
    #     })[f'hits@{K}']
    #     test_hits = evaluator.eval({
    #         'y_pred_pos': pos_test_pred,
    #         'y_pred_neg': neg_test_pred,
    #     })[f'hits@{K}']


#
#     results[f'Hits@{K}'] = (valid_hits, test_hits)

# return results


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


def run(optim_wd=.0,
          epochs=800,
          hidden_dim=256,
          dropout=0.1,
          num_layers=3,
          lr=3e-3,
          node_emb_dim=256,
          batch_size=64 * 512):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    num_buyers = 8922
    num_items = 3119
    num_sellers = 4056
    num_nodes_1 = num_buyers + num_items
    edge_list = read_from_txt('data/split/buyer_item.txt')
    edge_index = torch.tensor([edge_list[0], [i + num_buyers for i in edge_list[1]]])
    # print(edge_index.shape)

    train_indices = np.loadtxt('data/split/buyer_item_train.txt')
    val_indices = np.loadtxt('data/split/buyer_item_val.txt')
    test_indices = np.loadtxt('data/split/buyer_item_test.txt')

    # split_edge = dataset.get_edge_split()
    pos_train_edge = edge_index.clone()[:, train_indices].T
    # print(pos_train_edge.shape)

    # graph = dataset[0]
    edge_index = edge_index.to(device)

    # evaluator = Evaluator(name='ogbl-ddi')

    emb = torch.nn.Embedding(num_nodes_1, node_emb_dim).to(device)  # each node has an embedding that has to be learnt
    model = GNNStack(node_emb_dim, hidden_dim, hidden_dim, num_layers, dropout, emb=True).to(
        device)  # the graph neural network that takes all the node embeddings as inputs to message pass and agregate
    link_predictor = LinkPredictor(hidden_dim, hidden_dim, 1, num_layers + 1, dropout).to(
        device)  # the MLP that takes embeddings of a pair of nodes and predicts the existence of an edge between them

    split_edge = {}
    split_edge['test'] = {}
    split_edge['test']['edge'] = edge_index.clone()[:, val_indices].T
    # split_edge['test']['edge'] = pos_train_edge
    split_edge['test']['edge_neg'] = negative_sampling(edge_index, num_nodes=num_nodes_1,
                                                       num_neg_samples=split_edge['test']['edge'].shape[0],
                                                       method='dense').T
    # print(split_edge['test']['edge'].shape)

    buyer_test = np.loadtxt('data/split/buyer_val.txt')
    items_list = np.arange(num_items) + num_buyers
    gt_buyers = {}
    for i in val_indices:
        i = int(i)
        buyer = edge_list[0][i]
        item = edge_list[1][i]
        if buyer not in gt_buyers.keys():
            gt_buyers[buyer] = [item]
        else:
            gt_buyers[buyer].append(item)

    optimizer = torch.optim.Adam(
        list(model.parameters()) + list(link_predictor.parameters()) + list(emb.parameters()),
        lr=lr, weight_decay=optim_wd
    )
    scheduler = MultiStepLR(optimizer, milestones=[3000], gamma=0.1)

    train_loss = []
    val_hits = []
    test_pos_hits = []
    test_neg_hits = []
    buyer_item_accs = []
    for e in tqdm(range(epochs)):
        loss = train(model, link_predictor, emb.weight, edge_index, pos_train_edge, batch_size, optimizer, scheduler)
        # print(f"Epoch {e + 1}: loss: {round(loss, 5)}")
        train_loss.append(loss)

        if (e + 1) % 200 == 0:
            pos_hits, neg_hits = test(model, link_predictor, emb.weight, edge_index, split_edge, batch_size)
            test_pos_hits.append(pos_hits)
            test_neg_hits.append(neg_hits)
            print(pos_hits, neg_hits)
            result_buyer = evaluate(model, link_predictor, emb.weight, edge_index, buyer_test, items_list, batch_size)
            buyer_item_acc = []
            for buyer, preds in result_buyer.items():
                gts = gt_buyers[buyer]
                counts = 0
                for gt in gts:
                    if gt in preds:
                        counts += 1
                ratio = counts / len(gts)
                buyer_item_acc.append(ratio)
            print('buyer_item_acc', np.mean(buyer_item_acc))
            buyer_item_accs.append(np.mean(buyer_item_acc))
        if (e + 1) % 500 == 0:
            torch.save({
                'epoch': e,
                'model_state_dict': model.state_dict(),
                'link_predictor_state_dict': link_predictor.state_dict(),
                'emb': emb.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': loss,
            }, 'checkpoint/model5_{}.pt'.format(e))
            torch.save(emb.weight, 'checkpoint/emb5_{}.pt'.format(e))

    plt.title('Link Prediction on buyer_item using GraphSAGE GNN')
    plt.plot(train_loss, label="training loss")
    # plt.plot(np.arange(9,epochs,10),val_hits,label="Hits@20 on validation")
    plt.plot(np.arange(9, epochs, 200), test_pos_hits, label="PHits@20 on val")
    plt.plot(np.arange(9, epochs, 200), test_neg_hits, label="NHits@20 on val")
    plt.plot(np.arange(9, epochs, 200), buyer_item_accs, label="acc on val")
    plt.xlabel('Epochs')
    plt.legend()
    plt.show()


optim_wds = [1e-6]
epochss = [400, 800, 1200]
hidden_dims = [256]
# hidden_dim = 64
dropouts = [0.3]
num_layerss = [3]
lrs = [1e-2, 1e-3, 1e-4]
node_emb_dims = [8192]
# node_emb_dim = 64
batch_size = 64 * 1024

for num_layers in num_layerss:
    for optim_wd in optim_wds:
        for dropout in dropouts:
            for hidden_dim in hidden_dims:
                for node_emb_dim in node_emb_dims:
                    print(optim_wd, hidden_dim, dropout, num_layers, ':')
                    run(optim_wd=optim_wd, epochs=4000, hidden_dim=hidden_dim, dropout=dropout, num_layers=num_layers, lr=1e-3,
                        node_emb_dim=node_emb_dim)
