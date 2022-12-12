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
    num_nodes_1 = num_items + num_sellers
    edge_list = read_from_txt('data/'+dataset+'/item_seller.txt')
    edge_index = torch.tensor([edge_list[0], [i + num_items for i in edge_list[1]]])
    # print(edge_index.shape)

    train_indices = np.loadtxt('data/'+dataset+'/item_seller_train.txt')
    # val_indices = np.loadtxt('data/split/item_seller_val.txt')
    val_indices = np.loadtxt('data/'+dataset+'/item_seller_val.txt')
    test_indices = np.loadtxt('data/'+dataset+'/item_seller_test.txt')

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

    gt_items = {}
    for i in val_indices:
        i = int(i)
        item = edge_list[0][i]
        seller = edge_list[1][i]
        if item not in gt_items.keys():
            gt_items[item] = [seller]
        else:
            gt_items[item].append(seller)

    item_test = np.array(list(gt_items.keys()))
    # print(item_test)
    sellers_list = np.arange(num_sellers) + num_items

    optimizer = torch.optim.Adam(
        list(model.parameters()) + list(link_predictor.parameters()) + list(emb.parameters()),
        lr=lr, weight_decay=optim_wd
    )
    scheduler = MultiStepLR(optimizer, milestones=[2500], gamma=0.1)

    train_loss = []
    val_hits = []
    test_pos_hits = []
    test_neg_hits = []
    item_seller_accs = []
    for e in tqdm(range(epochs)):
        loss = train(model, link_predictor, emb.weight, edge_index, pos_train_edge, batch_size, optimizer, scheduler)
        # print(f"Epoch {e + 1}: loss: {round(loss, 5)}")
        train_loss.append(loss)

        if (e + 1) % 200 == 0:
            pos_hits, neg_hits = test(model, link_predictor, emb.weight, edge_index, split_edge, batch_size)
            test_pos_hits.append(pos_hits)
            test_neg_hits.append(neg_hits)
            print(pos_hits, neg_hits)
            result_item = evaluate(model, link_predictor, emb.weight, edge_index, item_test, sellers_list, batch_size)
            item_seller_acc = []
            for item, preds in result_item.items():
                gts = gt_items[item]
                counts = 0
                for gt in gts:
                    if gt in preds:
                        counts += 1
                ratio = counts / len(gts)
                item_seller_acc.append(ratio)
            print('item_seller_acc', np.mean(item_seller_acc))
            item_seller_accs.append(np.mean(item_seller_acc))
        if (e + 1) % 500 == 0:
            torch.save({
                'epoch': e,
                'model_state_dict': model.state_dict(),
                'link_predictor_state_dict': link_predictor.state_dict(),
                'emb': emb.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': loss,
            }, 'checkpoint/'+dataset+'model2_{}.pt'.format(e))
            torch.save(emb.weight, 'checkpoint/'+dataset+'emb2_{}.pt'.format(e))

    plt.title('Link Prediction on item_seller using GraphSAGE GNN')
    plt.plot(train_loss, label="training loss")
    # plt.plot(np.arange(9,epochs,10),val_hits,label="Hits@20 on validation")
    plt.plot(np.arange(9, epochs, 200), test_pos_hits, label="PHits@20 on val")
    plt.plot(np.arange(9, epochs, 200), test_neg_hits, label="NHits@20 on val")
    plt.plot(np.arange(9, epochs, 200), item_seller_accs, label="acc on val")
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
node_emb_dims = [2048]
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
