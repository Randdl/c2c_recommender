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


num_buyers = 8922
num_items = 3119
num_sellers = 4056
num_nodes_1 = num_buyers + num_items
edge_list = read_from_txt('data/split2/buyer_item.txt')
edge_index = torch.tensor([edge_list[0], [i + num_buyers for i in edge_list[1]]])
# print(edge_index.shape)

train_indices = np.loadtxt('data/split2/buyer_item_train.txt')
val_indices = np.loadtxt('data/split2/buyer_item_val.txt')
test_indices = np.loadtxt('data/split2/buyer_item_test.txt')

num_nodes2 = num_items + num_sellers
edge_list2 = read_from_txt('data/split2/item_seller.txt')
edge_index2 = torch.tensor([edge_list2[0], [i + num_items for i in edge_list2[1]]])
# print(edge_index2.shape)

train_indices2 = np.loadtxt('data/split2/item_seller_train.txt')
val_indices2 = np.loadtxt('data/split2/item_seller_val.txt')
test_indices2 = np.loadtxt('data/split2/item_seller_test.txt')

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
print(item_test)
sellers_list = np.arange(num_sellers) + num_items
print(sellers_list)

result_seller = predict_baseline(item_test, edge_list2, train_indices2)
# print(result_seller)
item_seller_acc = []
item_acc_dic = {}
for item, gts in gt_items.items():
    if item not in result_seller.keys():
        item_seller_acc.append(.0)
        item_acc_dic[item] = .0
        continue
    preds = result_seller[item]
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
print(item_seller_acc)
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
result_buyer = predict_baseline(buyer_test, edge_list, train_indices)
# print(result_buyer)
buyer_item_acc = []
total_acc = []
for buyer, gts in gt_buyers.items():
    if buyer not in result_buyer.keys():
        buyer_item_acc.append(.0)
        total_acc.append(.0)
        continue
    preds = result_buyer[buyer]
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
print(buyer_item_acc)
print('buyer_item_acc', np.mean(buyer_item_acc))
print(total_acc)
print('total_acc', np.mean(total_acc))
