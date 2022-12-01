# -*- coding: utf-8 -*-
"""convert_to_data.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1bejvJBaCU1qJLXRcJL7ISjX73j5pF0Pn
"""

# Commented out IPython magic to ensure Python compatibility.
# %%shell
# python -m pip install snap-stanford
# pip install torch-scatter -f https://data.pyg.org/whl/torch-1.12.1+cu113.html
# pip install torch-sparse -f https://data.pyg.org/whl/torch-1.12.1+cu113.html
# pip install torch-geometric
# pip install torch-cluster -f https://data.pyg.org/whl/torch-1.12.1+cu113.html
# pip install torch-spline-conv -f https://data.pyg.org/whl/torch-1.12.1+cu113.html

import json
import random
import numpy as np
import os
import snap
from tqdm import tqdm
import torch
from torch_geometric.data import Data
import pandas as pd

buyer_item = pd.read_csv('buyer_item.txt', sep = ' ', header = None)
buyer_item.columns = ['buyer_id','item_id','num']
print(buyer_item.head())
print(len(buyer_item['buyer_id'].unique()))
print(len(buyer_item['item_id'].unique()))
print(buyer_item.describe())
print(buyer_item.num.value_counts())

# change buyer_id to make it different from item_id
buyer_item['buyer_id'] = buyer_item.buyer_id.apply(lambda x: x + 3119)
buyer_item.head()

G = snap.TUNGraph().New()

# add nodes and edges to the G
for i in buyer_item['buyer_id'].unique().tolist():
  G.AddNode(i)
for i in buyer_item['item_id'].unique().tolist():
  G.AddNode(i)  
src = buyer_item['buyer_id'].tolist()
dst = buyer_item['item_id'].tolist()
for i in range(buyer_item.shape[0]):
  G.AddEdge(src[i],dst[i])

# check 
num_buyer = len([x for x in G.Nodes() if x.GetId() >= 3119])
num_item = len([x for x in G.Nodes() if x.GetId() < 3119])
print(num_buyer)
print(num_item)

all_edges = []
for i in tqdm(G.Edges()):
  edge_info = [i.GetSrcNId(),i.GetDstNId()]
  # undirected
  all_edges.append(edge_info)
  all_edges.append(edge_info[::-1])

edge_idx = torch.LongTensor(all_edges)

num_items = [[i] * 2 for i in buyer_item.num.tolist()]
edge_features = [[i] for t in num_items for i in t ]
edge_features = torch.Tensor(edge_features)
edge_features

data = Data(edge_index = edge_idx.t().contiguous(), num_nodes=G.GetNodes(), edge_attr = edge_features)

data

torch.save(data, os.path.join('/content', 'buyer_item.pt'))

item_seller = pd.read_csv('item_seller.txt', sep = ' ', header = None)
item_seller.columns = ['item_id','seller_id','num']
print(item_seller.head())
print(len(item_seller['item_id'].unique()))
print(len(item_seller['seller_id'].unique()))
print(item_seller.describe())
print(item_seller.num.value_counts())

# change seller_id to make it different from item_id
item_seller['seller_id'] = item_seller.seller_id.apply(lambda x: x + 3119)
item_seller.head()

G = snap.TUNGraph().New()

# add nodes and edges to the G
for i in item_seller['item_id'].unique().tolist():
  G.AddNode(i)
for i in item_seller['seller_id'].unique().tolist():
  G.AddNode(i)  
src = item_seller['item_id'].tolist()
dst = item_seller['seller_id'].tolist()
for i in range(item_seller.shape[0]):
  G.AddEdge(src[i],dst[i])

# check 
num_seller = len([x for x in G.Nodes() if x.GetId() >= 3119])
num_item = len([x for x in G.Nodes() if x.GetId() < 3119])
print(num_seller)
print(num_item)

all_edges = []
for i in tqdm(G.Edges()):
  edge_info = [i.GetSrcNId(),i.GetDstNId()]
  # undirected
  all_edges.append(edge_info)
  all_edges.append(edge_info[::-1])

edge_idx = torch.LongTensor(all_edges)

num_items = [[i] * 2 for i in item_seller.num.tolist()]
edge_features = [[i] for t in num_items for i in t ]
edge_features = torch.Tensor(edge_features)
edge_features

data = Data(edge_index = edge_idx.t().contiguous(), num_nodes=G.GetNodes(), edge_attr = edge_features)

data

torch.save(data, os.path.join('/content', 'item_seller.pt'))