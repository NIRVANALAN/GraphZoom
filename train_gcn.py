import argparse
import time
from easydict import EasyDict
import numpy as np
import networkx as nx
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.functional import softmax
from dgl import DGLGraph
from dgl.data import register_data_args, load_data, citegrh

from dgl_gcn.gcn import GCN
from dgl_gcn.utils import _sample_mask, load_data
from graphzoom.utils import mtx2graph, construct_proj_laplacian


def evaluate(model, features, labels, mask):
    model.eval()
    with torch.no_grad():
        logits = model(features)
        logits = logits[mask]
        labels = labels[mask]
        _, indices = torch.max(logits, dim=1)
        correct = torch.sum(indices == labels)
        return correct.item() * 1.0 / len(labels)


def main(args):
    # load and preprocess dataset
    dataset = args.dataset
    dataset_dir = f'graphzoom/dataset/{dataset}'
    # data = load_data(dataset_dir, args.dataset)

    G, labels, train_ids, test_ids, train_labels, test_labels, feats = load_data(
        dataset_dir, args.dataset)
    val_ids = test_ids[1000:1500]
    test_ids = test_ids[:1000]
    labels = torch.LongTensor(labels)
    train_mask = _sample_mask(train_ids, labels.shape[0])
    test_mask = _sample_mask(test_ids, labels.shape[0])
    val_mask = _sample_mask(val_ids, labels.shape[0])
    # val_mask = _sample_mask(range(200, 500), labels.shape[0])
    onehot_labels = F.one_hot(labels)
    print(len(train_labels))
    print(len(test_ids))
    print(len(val_ids))

    data = EasyDict({
        'graph': G,
        'labels': labels,
        'onehot_labels': onehot_labels,
        'features': feats,
        'train_mask': train_mask,
        'val_mask': val_mask,
        'test_mask': test_mask,
        'num_labels': onehot_labels.shape[1],
        'coarse': False

    })

    # * load projection matrix
    levels = 2
    reduce_results = f"graphzoom/reduction_results/{dataset}"
    original_adj = nx.adj_matrix(G)
    projections, coarse_adj = construct_proj_laplacian(
        original_adj, levels, reduce_results)
    # *calculate coarse feature, labels
    coarse_feats = projections[0] @ data.features
    coarse_labels = projections[0] @ data.onehot_labels
    coarse_graph = nx.Graph(coarse_adj[1])
    rows_sum = coarse_labels.sum(axis=1)[:, np.newaxis]
    norm_coarse_labels = coarse_labels / rows_sum
    # list(map(np.shape, [coarse_embed, coarse_labels]))
    # * new train/test masks
    coarse_train_mask = _sample_mask(range(100), norm_coarse_labels.shape[0])
    coarse_test_mask = _sample_mask(
        range(100, 700), norm_coarse_labels.shape[0])
    coarse_val_mask = _sample_mask(
        range(700, 1000), norm_coarse_labels.shape[0])

    # *replace data
    coarse_data = EasyDict({
        'graph': coarse_graph,
        'labels': coarse_labels,
        #     'onehot_labels': onehot_labels,
        'features': coarse_feats,
        'train_mask': coarse_train_mask,
        'val_mask': coarse_val_mask,
        'test_mask': coarse_test_mask,
        'num_classes': norm_coarse_labels.shape[1],
        'num_labels': onehot_labels.shape[1],
        'coarse': True
    })
    data = coarse_data
    if data.coarse:
        labels = torch.FloatTensor(data.labels)
        loss_fcn = torch.nn.KLDivLoss()
        print('training coarse')
    else:
        labels = torch.LongTensor(data.labels)
        loss_fcn = torch.nn.CrossEntropyLoss()
    features = torch.FloatTensor(data.features)
    if hasattr(torch, 'BoolTensor'):
        train_mask = torch.BoolTensor(data.train_mask)
        val_mask = torch.BoolTensor(data.val_mask)
        test_mask = torch.BoolTensor(data.test_mask)
    else:
        train_mask = torch.ByteTensor(data.train_mask)
        val_mask = torch.ByteTensor(data.val_mask)
        test_mask = torch.ByteTensor(data.test_mask)
    in_feats = features.shape[1]
    n_classes = data.num_labels
    n_edges = data.graph.number_of_edges()
    print("""----Data statistics------'
      #Edges %d
      #Classes %d
      #Train samples %d
      #Val samples %d
      #Test samples %d""" %
          (n_edges, n_classes,
              train_mask.int().sum().item(),
              val_mask.int().sum().item(),
              test_mask.int().sum().item()))

    if args.gpu < 0:
        cuda = False
    else:
        cuda = True
        torch.cuda.set_device(args.gpu)
        features = features.cuda()
        labels = labels.cuda()
        train_mask = train_mask.cuda()
        val_mask = val_mask.cuda()
        test_mask = test_mask.cuda()

    # graph preprocess and calculate normalization factor
    g = data.graph
    # add self loop
    if args.self_loop:
        g.remove_edges_from(nx.selfloop_edges(g))
        g.add_edges_from(zip(g.nodes(), g.nodes()))
    g = DGLGraph(g)
    n_edges = g.number_of_edges()
    # normalization
    degs = g.in_degrees().float()
    norm = torch.pow(degs, -0.5)
    norm[torch.isinf(norm)] = 0
    if cuda:
        norm = norm.cuda()
    g.ndata['norm'] = norm.unsqueeze(1)

    # create GCN model
    model = GCN(g,
                in_feats,
                args.n_hidden,
                n_classes,
                args.n_layers,
                F.relu,
                args.dropout, log_softmax=data.coarse)

    if cuda:
        model.cuda()
    # loss_fcn = torch.nn.CrossEntropyLoss()

    # use optimizer
    optimizer = torch.optim.Adam(model.parameters(),
                                 lr=args.lr,
                                 weight_decay=args.weight_decay)

    # initialize graph
    dur = []
    acc = 0
    for epoch in range(args.n_epochs):
        model.train()
        if epoch >= 3:
            t0 = time.time()
        # forward
        logits, h = model(features)
        loss = loss_fcn(logits[train_mask], labels[train_mask])

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if epoch >= 3:
            dur.append(time.time() - t0)

        # acc = evaluate(model, features, labels, val_mask)
        print("Epoch {:05d} | Time(s) {:.4f} | Loss {:.4f} | Accuracy {:.4f} | "
              "ETputs(KTEPS) {:.2f}". format(epoch, np.mean(dur), loss.item(),
                                             acc, n_edges / np.mean(dur) / 1000))

    # acc = evaluate(model, features, labels, test_mask)
    print(h.shape)
    np.save(f'embeddings/{repr(model)}_{dataset}_emb_level_1',
            h.detach().cpu().numpy())
    print("Test accuracy {:.2%}".format(acc))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='GCN')
    register_data_args(parser)
    parser.add_argument("--dropout", type=float, default=0.5,
                        help="dropout probability")
    parser.add_argument("--gpu", type=int, default=0,
                        help="gpu")
    parser.add_argument("--lr", type=float, default=1e-2,
                        help="learning rate")
    parser.add_argument("--n-epochs", type=int, default=200,
                        help="number of training epochs")
    parser.add_argument("--n-hidden", type=int, default=128,
                        help="number of hidden gcn units")
    parser.add_argument("--n-layers", type=int, default=1,
                        help="number of hidden gcn layers")
    parser.add_argument("--weight-decay", type=float, default=5e-4,
                        help="Weight for L2 loss")
    parser.add_argument("--self-loop", action='store_true',
                        help="graph self-loop (default=False)")
    parser.set_defaults(self_loop=False)
    args = parser.parse_args()
    print(args)

    main(args)
