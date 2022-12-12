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
from utils import *

dataset = 'split'
# dataset = 'split2'


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
    edge_list = read_from_txt('data/'+dataset+'/buyer_item.txt')
    edge_index = torch.tensor([edge_list[0], [i + num_buyers for i in edge_list[1]]])
    # print(edge_index.shape)

    train_indices = np.loadtxt('data/'+dataset+'/buyer_item_train.txt')
    # val_indices = np.loadtxt('data/split/buyer_item_val.txt')
    val_indices = np.loadtxt('data/'+dataset+'/buyer_item_val.txt')
    test_indices = np.loadtxt('data/'+dataset+'/buyer_item_test.txt')

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

    buyer_test = np.loadtxt('data/'+dataset+'/buyer_val.txt')
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
        if (e + 1) % 100 == 0:
            torch.save({
                'epoch': e,
                'model_state_dict': model.state_dict(),
                'link_predictor_state_dict': link_predictor.state_dict(),
                'emb': emb.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': loss,
            }, 'checkpoint/'+dataset+'model1_{}.pt'.format(e))
            torch.save(emb.weight, 'checkpoint/'+dataset+'emb1_{}.pt'.format(e))

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
num_layerss = [5]
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
