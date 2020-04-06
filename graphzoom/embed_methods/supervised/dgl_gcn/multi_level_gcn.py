"""GCN using DGL nn package

References:
- Semi-Supervised Classification with Graph Convolutional Networks
- Paper: https://arxiv.org/abs/1609.02907
- Code: https://github.com/tkipf/gcn
"""
import torch
import torch.nn as nn
from dgl.nn.pytorch import GraphConv
import torch.nn.functional as F


class MultiLevelGCN(nn.Module):
    def __init__(self,
                 g,
                 in_dim,
                 num_hidden,
                 num_classes,
                 num_layers,
                 activation=F.relu,
                 dropout=0.5, log_softmax=False, projection_matrix=None, *args, **kwargs):
        super(MultiLevelGCN, self).__init__()
        self.g = g
        self.projection_matrix = projection_matrix
        self.layers = nn.ModuleList()
        # input layer
        self.layers.append(
            GraphConv(in_dim, num_hidden, activation=activation))
        # hidden layers
        for i in range(num_layers - 1):
            self.layers.append(
                GraphConv(num_hidden, num_hidden, activation=activation))
        # output layer
        self.layers.append(GraphConv(num_hidden, num_classes))
        self.dropout = nn.Dropout(p=dropout)
        self.log_softmax = log_softmax

    def forward(self, features):
        h = features
        for i, layer in enumerate(self.layers):
            if i != 0:
                h = self.dropout(h)
            emb = h
            h = layer(self.g[i], h)
            if i == len(self.layers)-1:
                break
            h = self.projection_matrix[i+1].t() @ h
        if self.log_softmax:
            return nn.functional.log_softmax(h, 1), emb
        return h, emb

    def __repr__(self):
        return super().__repr__()
