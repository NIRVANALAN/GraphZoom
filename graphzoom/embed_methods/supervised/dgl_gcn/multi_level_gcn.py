"""GCN using DGL nn package

References:
- Semi-Supervised Classification with Graph Convolutional Networks
- Paper: https://arxiv.org/abs/1609.02907
- Code: https://github.com/tkipf/gcn
"""
import torch
import torch.nn as nn
from dgl.nn.pytorch import GraphConv
from .lpa import SmoothFilter
import torch.nn.functional as F


class MultiLevelGCN(nn.Module):
    def __init__(self,
                 g,
                 in_dim,
                 num_hidden,
                 num_classes,
                 num_layers,
                 activation=F.relu,
                 dropout=0.5, log_softmax=False, projection_matrix=None, smooth_filter_k=2,*args, **kwargs):
        super(MultiLevelGCN, self).__init__()
        self.g = g
        self.projections = projection_matrix
        self.layers = nn.ModuleList()
        # input layer
        self.layers.append(
            GraphConv(in_dim, num_hidden, activation=activation))
        # hidden layers
        self.layers.append(SmoothFilter())
        self.smooth_filter_k = smooth_filter_k
        # for i in range(num_layers - 1):
        #     self.layers.append(
        #         GraphConv(num_hidden, num_hidden, activation=activation)) #TODO replace with SGCN
        # output layer
        self.layers.append(GraphConv(num_hidden, num_classes))
        self.dropout = nn.Dropout(p=dropout) # TODO
        self.log_softmax = log_softmax

    def forward(self, features):
        h = features
        h = self.layers[0](self.g[0], h)
        h = self.projections[0].t() @ h
        emb = h
        # import pdb; pdb.set_trace()
        # smooth intermidiate embeddigns
        for i in range(1, len(self.projections)):
            # h = self.dropout(h) #? #TODO
            for _ in range(self.smooth_filter_k):
                h = self.layers[1](self.g[i], h)
            h = self.projections[i].t() @ h
        # output projection
        h = self.dropout(h)
        h = self.layers[2](self.g[-1], h)

        if self.log_softmax:
            return nn.functional.log_softmax(h, 1), emb
        return h, emb

    def __repr__(self):
        return super().__repr__()
