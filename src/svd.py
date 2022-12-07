import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric as pyg
from torch_geometric.data import Data
from torch_geometric.data import DataLoader
from torch_geometric.utils import negative_sampling
import matplotlib.pyplot as plt
import numpy as np
from scipy.sparse.linalg import svds


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


def find_top_k(predictions, k=50):
    result = np.zeros((predictions.shape[0], k))
    for i in range(predictions.shape[0]):
        row = predictions[i]
        result[i] = np.argsort(row)[:50]
        if i == 1:
            print(result[i])
    return result


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

num_nodes2 = num_items + num_sellers
edge_list2 = read_from_txt('data/split/item_seller.txt')
edge_index2 = torch.tensor([edge_list2[0], [i + num_items for i in edge_list2[1]]])
# print(edge_index2.shape)

train_indices2 = np.loadtxt('data/split/item_seller_train.txt')
val_indices2 = np.loadtxt('data/split/item_seller_val.txt')
test_indices2 = np.loadtxt('data/split/item_seller_test.txt')


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


R2 = np.zeros((num_items, num_sellers))
for i in train_indices2:
    i = int(i)
    item = edge_list2[0][i]
    seller = edge_list2[1][i]
    R2[item, seller] = 1

U2, sigma2, Vt2 = svds(R2, k=100)
sigma2 = np.diag(sigma2)
predictions2 = np.dot(np.dot(U2, sigma2), Vt2)
print((predictions2 > 0.3).nonzero())
print((predictions2 > 0.3).nonzero()[0].shape)

result_item = find_top_k(predictions2)
gt_result_seller = predict_baseline(item_test, edge_list2, train_indices2)
# print(result_seller)
item_seller_acc = []
item_acc_dic = {}
for item, gts in gt_items.items():
    preds = result_item[item]
    if item not in gt_result_seller.keys():
        bases = []
    else:
        bases = gt_result_seller[item]
    # print(preds[gts])
    # print(gts)
    counts = 0
    for gt in gts:
        if gt in preds:
            # print(preds)
            # print(gts)
            counts += 1
    ratio = counts / len(gts)
    item_seller_acc.append(ratio)
    item_acc_dic[item] = ratio
# print(item_seller_acc)
print('item_seller_acc:', np.mean(item_seller_acc))

buyer_test = np.loadtxt('data/split/buyer_test.txt')
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
R = np.zeros((num_buyers, num_items))
for i in train_indices:
    i = int(i)
    buyer = edge_list[0][i]
    item = edge_list[1][i]
    R[buyer, item] = 1

U, sigma, Vt = svds(R, k=100)
sigma = np.diag(sigma)
predictions = np.dot(np.dot(U, sigma), Vt)
print((predictions > 0.3).nonzero())
print((predictions > 0.3).nonzero()[0].shape)

result_buyer = find_top_k(predictions)
gt_result_buyer = predict_baseline(buyer_test, edge_list, train_indices)
# print(result_buyer)
buyer_item_acc = []
total_acc = []
for buyer, gts in gt_buyers.items():
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
        if gt in preds:
            # print(preds)
            # print(gts)
            counts += 1
            weighted_counts += item_acc_dic[gt]
    ratio = counts / len(gts)
    buyer_item_acc.append(ratio)
    total_acc.append(weighted_counts)
# print(buyer_item_acc)
print('buyer_item_acc', np.mean(buyer_item_acc))
# print(total_acc)
print('total_acc', np.mean(total_acc))