import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric as pyg
from torch_geometric.data import Data
from torch_geometric.data import DataLoader
from torch_geometric.utils import negative_sampling
import matplotlib.pyplot as plt
import numpy as np
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
    val_indices = np.loadtxt('data/'+dataset+'/buyer_item_val.txt')
    test_indices = np.loadtxt('data/'+dataset+'/buyer_item_test.txt')

    # split_edge = dataset.get_edge_split()
    pos_train_edge = edge_index.clone()[:, train_indices].T
    # print(pos_train_edge.shape)

    # graph = dataset[0]
    edge_index = edge_index.to(device)

    # evaluator = Evaluator(name='ogbl-ddi')
    checkpoint = torch.load('checkpoint/'+dataset+'model1_3999.pt')

    emb = nn.Embedding.from_pretrained(torch.load('checkpoint/'+dataset+'emb1_3999.pt')).to(device)  # each node has an embedding that has to be learnt
    model = GNNStack(node_emb_dim, hidden_dim, hidden_dim, num_layers, dropout, emb=True).to(
        device)  # the graph neural network that takes all the node embeddings as inputs to message pass and agregate
    link_predictor = LinkPredictor(hidden_dim, hidden_dim, 1, num_layers + 1, dropout).to(
        device)  # the MLP that takes embeddings of a pair of nodes and predicts the existence of an edge between them
    model.load_state_dict(checkpoint['model_state_dict'])
    link_predictor.load_state_dict(checkpoint['link_predictor_state_dict'])

    split_edge = {}
    split_edge['test'] = {}
    split_edge['test']['edge'] = edge_index.clone()[:, test_indices].T
    # split_edge['test']['edge'] = pos_train_edge
    split_edge['test']['edge_neg'] = negative_sampling(edge_index, num_nodes=num_nodes_1,
                                                       num_neg_samples=split_edge['test']['edge'].shape[0],
                                                       method='dense').T
    # print(split_edge['test']['edge'].shape)

    pos_hits, neg_hits = test(model, link_predictor, emb.weight, edge_index, split_edge, batch_size)


    num_nodes2 = num_items + num_sellers
    edge_list2 = read_from_txt('data/'+dataset+'/item_seller.txt')
    edge_index2 = torch.tensor([edge_list2[0], [i + num_items for i in edge_list2[1]]])
    # print(edge_index2.shape)

    train_indices2 = np.loadtxt('data/'+dataset+'/item_seller_train.txt')
    val_indices2 = np.loadtxt('data/'+dataset+'/item_seller_val.txt')
    test_indices2 = np.loadtxt('data/'+dataset+'/item_seller_test.txt')

    # split_edge = dataset.get_edge_split()
    pos_train_edge2 = edge_index2.clone()[:, train_indices2].T
    # print(pos_train_edge2.shape)

    # graph = dataset[0]
    edge_index2 = edge_index2.to(device)

    # evaluator = Evaluator(name='ogbl-ddi')

    checkpoint2 = torch.load('checkpoint/'+dataset+'model2_3999.pt')

    emb2 = nn.Embedding.from_pretrained(torch.load('checkpoint/'+dataset+'emb2_3999.pt')).to(
        device)  # each node has an embedding that has to be learnt
    model2 = GNNStack(2048, hidden_dim, hidden_dim, num_layers, dropout, emb=True).to(
        device)  # the graph neural network that takes all the node embeddings as inputs to message pass and agregate
    link_predictor2 = LinkPredictor(hidden_dim, hidden_dim, 1, num_layers + 1, dropout).to(
        device)  # the MLP that takes embeddings of a pair of nodes and predicts the existence of an edge between them
    model2.load_state_dict(checkpoint2['model_state_dict'])
    link_predictor2.load_state_dict(checkpoint2['link_predictor_state_dict'])

    split_edge2 = {}
    split_edge2['test'] = {}
    split_edge2['test']['edge'] = edge_index2.clone()[:, test_indices2].T
    # split_edge['test']['edge'] = pos_train_edge
    split_edge2['test']['edge_neg'] = negative_sampling(edge_index2, num_nodes=num_nodes2,
                                                        num_neg_samples=split_edge2['test']['edge'].shape[0],
                                                        method='dense').T

    pos_hits, neg_hits = test(model2, link_predictor2, emb2.weight, edge_index2, split_edge2, batch_size)

    gt_items = {}
    for i in test_indices2:
        i = int(i)
        item = edge_list2[0][i]
        seller = edge_list2[1][i]
        if item not in gt_items.keys():
            gt_items[item] = [seller]
        else:
            gt_items[item].append(seller)

    item_test = np.array(list(gt_items.keys()))
    # print(item_test)
    sellers_list = np.arange(num_sellers) + num_items
    # print(sellers_list)

    result_seller = evaluate(model2, link_predictor2, emb2.weight, edge_index2, item_test, sellers_list, batch_size)
    gt_result_seller = predict_baseline(item_test, edge_list2, train_indices2)
    # print(result_seller)
    item_seller_acc = []
    item_acc_dic = {}
    for item, gts in gt_items.items():
        if item not in result_seller.keys():
            item_seller_acc.append(.0)
            item_acc_dic[item] = .0
            continue
        preds = result_seller[item]
        if item not in gt_result_seller.keys():
            bases = []
        else:
            bases = gt_result_seller[item]
        # print(preds[gts])
        # print(gts)
        counts = 0
        for gt in gts:
            if gt in preds or gt in bases:
                # print(preds)
                # print(gts)
                counts += 1
        ratio = counts / len(gts)
        item_seller_acc.append(ratio)
        item_acc_dic[item] = ratio
    # print(item_seller_acc)
    print('item_seller_acc:', np.mean(item_seller_acc))

    buyer_test = np.loadtxt('data/'+dataset+'/buyer_test.txt')
    items_list = np.arange(num_items) + num_buyers
    # print(items_list)
    # print(test_indices)
    # print(edge_list)
    gt_buyers = {}
    for i in test_indices:
        i = int(i)
        buyer = edge_list[0][i]
        item = edge_list[1][i]
        if buyer not in gt_buyers.keys():
            gt_buyers[buyer] = [item]
        else:
            gt_buyers[buyer].append(item)
    # print(gt_buyers)
    result_buyer = evaluate(model, link_predictor, emb.weight, edge_index, buyer_test, items_list, batch_size)
    gt_result_buyer = predict_baseline(buyer_test, edge_list, train_indices)
    for buyer in buyer_test:
        if buyer in gt_result_buyer.keys():
            nums = len(gt_result_buyer[buyer])
            test_nodes = result_buyer[buyer]
            result_buyer[buyer] = test_nodes[:50-nums]
    # print(result_buyer)
    buyer_item_acc = []
    total_acc = []
    for buyer, gts in gt_buyers.items():
        if buyer not in result_buyer.keys():
            buyer_item_acc.append(.0)
            total_acc.append(.0)
            continue
        preds = result_buyer[buyer]
        if buyer not in gt_result_buyer.keys():
            bases = []
        else:
            bases = gt_result_buyer[buyer]
        # print(preds[gts])
        # print(gts)
        counts = 0
        weighted_counts = 0
        for gt in gts:
            if gt in preds or gt in bases:
                # print(preds)
                # print(gts)
                counts += 1
                weighted_counts += item_acc_dic[gt]
        ratio = counts / len(gts)
        buyer_item_acc.append(ratio)
        total_acc.append(weighted_counts / len(gts))
    # print(buyer_item_acc)
    print('buyer_item_acc', np.mean(buyer_item_acc))
    # print(total_acc)
    print('total_acc', np.mean(total_acc))


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
                    run(optim_wd=optim_wd, epochs=1200, hidden_dim=hidden_dim, dropout=dropout, num_layers=num_layers, lr=1e-3,
                        node_emb_dim=node_emb_dim)
