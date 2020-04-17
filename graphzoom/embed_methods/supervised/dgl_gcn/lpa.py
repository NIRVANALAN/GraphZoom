"""GCN using DGL nn package

References:
- Semi-Supervised Classification with Graph Convolutional Networks
- Paper: https://arxiv.org/abs/1609.02907
- Code: https://github.com/tkipf/gcn
"""
import torch
import torch.nn as nn
from torch.nn import init
from dgl.nn.pytorch import GraphConv
import torch.nn.functional as F

import torch as th

from dgl import function as fn
from dgl.base import DGLError

# pylint: disable=W0235
class SmoothFilter(nn.Module):
    r"""Apply label propagation algorithm
    Y_k+1 = D^(-1)AL
    yi_k+1 = yi(0)

    Parameters
    ----------
    norm : str, optional
        How to apply the normalizer. If is `'right'`, divide the aggregated messages
        by each node's in-degrees, which is equivalent to averaging the received messages.
        If is `'none'`, no normalization is applied. Default is `'both'`,
        where the :math:`c_{ij}` in the paper is applied.
    """
    def __init__(self,
                 norm='both'):
        super(SmoothFilter, self).__init__()
        if norm not in ('none', 'both', 'right'):
            raise DGLError('Invalid norm value. Must be either "none", "both" or "right".'
                           ' But got "{}".'.format(norm))
        self._norm = norm

    def forward(self, graph, embeddings):

        graph = graph.local_var()

        if self._norm == 'both':
            degs = graph.out_degrees().to(embeddings.device).float().clamp(min=1)
            norm = th.pow(degs, -0.5)
            shp = norm.shape + (1,) * (embeddings.dim() - 1)
            norm = th.reshape(norm, shp)
            embeddings = embeddings * norm

            graph.srcdata['h'] = embeddings
            graph.update_all(fn.copy_src(src='h', out='m'),
                             fn.sum(msg='m', out='h'))
            rst = graph.dstdata['h']

        if self._norm != 'none':
            degs = graph.in_degrees().to(embeddings.device).float().clamp(min=1)
            if self._norm == 'both':
                norm = th.pow(degs, -0.5)
            else:
                norm = 1.0 / degs
            shp = norm.shape + (1,) * (embeddings.dim() - 1)
            norm = th.reshape(norm, shp)
            rst = rst * norm
        return rst

class LPA(nn.Module):
    def __init__(self,
                 g,
                 in_dim,
                 num_hidden,
                 num_classes,
                 num_layers,
                 activation=F.relu,
                 dropout=0.5, log_softmax=False, *args, **kwargs):
        super(LPA, self).__init__()
        self.g = g
        self.layers = nn.ModuleList()
        # input layer
        self.layers.append(
            SmoothFilter(in_dim, num_hidden, activation=activation))
        # hidden layers
        for i in range(num_layers - 1):
            self.layers.append(
                SmoothFilter(num_hidden, num_hidden, activation=activation))
        # output layer
        self.layers.append(SmoothFilter(num_hidden, num_classes))
        self.dropout = nn.Dropout(p=dropout)
        self.log_softmax = log_softmax

    def forward(self, features):
        h = features
        for i, layer in enumerate(self.layers):
            if i != 0:
                h = self.dropout(h)
            emb = h
            h = layer(self.g, h)
        if self.log_softmax:
            return nn.functional.log_softmax(h, 1), emb
        return h, emb

    def __repr__(self):
        return super().__repr__()
