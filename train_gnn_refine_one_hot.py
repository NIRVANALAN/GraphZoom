import argparse
import time
from pathlib import Path
from torch.distributions import Categorical

import networkx as nx
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from dgl import DGLGraph
from dgl.data import citegrh
from dgl.data import load_data as load_dgl_data
from dgl.data import load_graphs, register_data_args
from dgl.transform import add_self_loop
from easydict import EasyDict
from networkx.readwrite import json_graph, read_gpickle
from scipy import sparse
from torch.nn.functional import softmax

from graphzoom.embed_methods.supervised.dgl_gcn import (GAT, GCN, _sample_mask,
                                                        load_data)
from graphzoom.utils import construct_proj_laplacian, mtx2graph

FACTORY = {
    'gcn': GCN,
    'gat': GAT, }


def create_model(name, g, **kwargs):
    if name not in FACTORY:
        raise NotImplementedError(f'{name} not in arch FACTORY')
    return FACTORY[name](g, **kwargs)


def evaluate(model, features, labels, mask):
    model.eval()
    with torch.no_grad():
        logits, _ = model(features)
        logits = logits[mask]
        labels = labels[mask]
        _, indices = torch.max(logits, dim=1)
        correct = torch.sum(indices == labels)
        return correct.item() * 1.0 / len(labels)


def main(args):
    # load and preprocess dataset
    # data = load_dgl_data(args)
    dataset = args.dataset
    # prefix = '/mnt/yushi/'
    # prefix = 'graphzoom'
    dataset_dir = f'{args.prefix}/dataset/{dataset}'
    # data = load_data(dataset_dir, args.dataset)

    load_data_time = time.time()
    # if dataset in ['Amazon2M', 'reddit']:
    if dataset in ['Amazon2M']:
        g, _ = load_graphs(
            f'{args.prefix}/dataset/Amazon2M/Amazon2M_dglgraph.bin')
        g = g[0]
        data = g.ndata
        features = torch.FloatTensor(data['feat'])
        onehot_labels = F.one_hot(data['label']).numpy()
        train_mask = data['train_mask'].bool()
        val_mask = data['val_mask'].bool()
        test_mask = val_mask
        data = EasyDict({
            'graph': g,
            'labels': data['label'],
            'onehot_labels': onehot_labels,
            'features': data['feat'],
            'train_mask': train_mask,
            'val_mask': val_mask,
            'test_mask': test_mask,
            'num_labels': onehot_labels.shape[1],
            'coarse': False

        })
    else:
        original_adj, labels, train_ids, test_ids, train_labels, test_labels, feats = load_data(
            dataset_dir, args.dataset)
        data = load_dgl_data(args)
        labels = torch.LongTensor(labels)
        train_mask = _sample_mask(train_ids, labels.shape[0])
        onehot_labels = F.one_hot(labels).numpy()
        if dataset == 'reddit':
            g = data.graph
        else:
            val_ids = test_ids[1000:1500]
            test_ids = test_ids[:1000]
            test_mask = _sample_mask(test_ids, labels.shape[0])
            val_mask = _sample_mask(val_ids, labels.shape[0])
            data = EasyDict({
                'graph': data.graph,
                'labels': labels,
                # 'onehot_labels': onehot_labels,
                'features': feats,
                # 'features': data.features,
                'train_mask': train_mask,
                'val_mask': val_mask,
                'test_mask': test_mask,
                'num_labels': onehot_labels.shape[1],
                'coarse': False
            })
            # g = DGLGraph(data.graph)
    print(f'load data finished: {time.time() - load_data_time}')
    # * load projection matrix
    levels = args.level
    reduce_results = f"graphzoom/reduction_results/{dataset}/{args.proj}/"
    projections, coarse_adj = construct_proj_laplacian(
        original_adj, levels, reduce_results)
    # *calculate coarse feature, labels
    label_mask = np.expand_dims(data.train_mask, 1)
    onehot_labels = onehot_labels * label_mask
    # coarse_train_mask =[]
    for i in range(levels):
        data.features = projections[i] @ data.features
        onehot_labels = projections[i] @ onehot_labels
    # coarse_labels = projections[0] @ onehot_labels
        # ! add train_mask
    rows_sum = onehot_labels.sum(axis=1)[:, np.newaxis]
    norm_coarse_labels = onehot_labels / rows_sum
    norm_label_entropy = Categorical(
        torch.Tensor(norm_coarse_labels)).entropy()
    # label_entropy_mask = torch.BoolTensor(norm_label_entropy < 0.01)  # also remove NAN
    # coarse_train_mask.append(label_entropy_mask)
    coarse_train_mask = torch.BoolTensor(onehot_labels.sum(axis=1))
    # ! entropy threshold

    # coarse_graph = nx.Graph(coarse_adj[1])
    print('creating coarse DGLGraph')
    start = time.process_time()
    g = DGLGraph()
    g.from_scipy_sparse_matrix(coarse_adj[levels])
    print(f'creating finished in {time.process_time() - start}')
    # list(map(np.shape, [coarse_embed, coarse_labels]))
    # * new train/test masks
    # coarsen_ratio = projections[0].shape[1] / projections[0].shape[0]
    coarse_test_mask = _sample_mask(
        range(0, 0), norm_coarse_labels.shape[0])
    coarse_val_mask = _sample_mask(
        range(0, 0), norm_coarse_labels.shape[0])

    # *replace data
    data = EasyDict({
        'graph': g,
        'labels': onehot_labels.argmax(1),
        #     'onehot_labels': onehot_labels,
        'features': data.features,
        'train_mask': coarse_train_mask,
        'val_mask': coarse_val_mask,
        'test_mask': coarse_test_mask,
        'num_classes': norm_coarse_labels.shape[1],
        'num_labels': onehot_labels.shape[1],
        'coarse': True
    })
# if args.coarse:
    labels = torch.LongTensor(data.labels)
    # loss_fcn = torch.nn.KLDivLoss(reduction='batchmean')
    loss_fcn = torch.nn.CrossEntropyLoss()
    print('training coarse')
    # else:
    #     labels = torch.LongTensor(data.labels)
    #     loss_fcn = torch.nn.CrossEntropyLoss()
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

    cuda = True
    torch.cuda.set_device(args.gpu)
    features = features.cuda()
    labels = labels.cuda()
    train_mask = train_mask.cuda()
    val_mask = val_mask.cuda()
    test_mask = test_mask.cuda()

    # graph preprocess and calculate normalization factor
    # add self loop
    if args.self_loop or args.arch == 'gat':
        g = add_self_loop(data.graph)
        print('add self_loop')
    n_edges = g.number_of_edges()
    print("""----Data statistics------'
      # Edges %d
      # Classes %d
      # Train samples %d
      # Val samples %d
      # Test samples %d""" %
          (n_edges, n_classes,
              train_mask.int().sum().item(),
              val_mask.int().sum().item(),
              test_mask.int().sum().item()))
    # normalization
    degs = g.in_degrees().float()
    norm = torch.pow(degs, -0.5)
    norm[torch.isinf(norm)] = 0
    if cuda:
        norm = norm.cuda()
    g.ndata['norm'] = norm.unsqueeze(1)

    # * create GCN model
    heads = ([args.num_heads] * args.num_layers) + [args.num_out_heads]
    model = create_model(args.arch, g,
                         num_layers=args.num_layers,
                         in_dim=in_feats,
                         num_hidden=args.num_hidden,
                         num_classes=n_classes,
                         heads=heads,
                         #  activation=F.elu,
                         feat_drop=args.in_drop,
                         attn_drop=args.attn_drop,
                         negative_slope=args.negative_slope,
                         residual=args.residual, log_softmax=False)

    if cuda:
        model.cuda()
    print(model)
    # loss_fcn = torch.nn.CrossEntropyLoss()

    # use optimizer
    optimizer = torch.optim.Adam(model.parameters(),
                                 lr=args.lr,
                                 weight_decay=args.weight_decay)

    # initialize graph
    dur = []
    acc = 0
    start = time.time()
    for epoch in range(args.n_epochs):
        model.train()
        if epoch >= 3:
            t0 = time.time()
        # forward
        logits, h = model(features)
        loss = loss_fcn(logits[train_mask], labels[train_mask])  # ?

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if epoch >= 3:
            dur.append(time.time() - t0)

        if not args.coarse:
            acc = evaluate(model, features, labels, val_mask)
        print("Epoch {:05d} | Time(s) {:.4f} | Loss {:.4f} | Accuracy {:.4f} | "
              "ETputs(KTEPS) {:.2f}". format(epoch, np.mean(dur), loss.item(),
                                             acc, n_edges / np.mean(dur) / 1000))
    print(f'training time: {time.time() - start}')
    if not args.coarse:
        acc = evaluate(model, features, labels, test_mask)
    print(h.shape)
    embedding_save_path = f'embeddings/{(args.arch).upper()}_{dataset}_emb_level_{levels}_mask'
    print(f'save embedding at {embedding_save_path}')
    np.save(embedding_save_path, h.detach().cpu().numpy())
    torch.save(model.state_dict(),
               f'embeddings/{(args.arch).upper()}_{dataset}_emb_level_1_params.pth.tar',)
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
    # parser.add_argument("--n-hidden", type=int, default=128,
    #                     help="number of hidden gcn units")
    # parser.add_argument("--n-layers", type=int, default=1,
    #                     help="number of hidden gcn layers")
    parser.add_argument("--weight-decay", type=float, default=5e-4,
                        help="Weight for L2 loss")
    parser.add_argument("--self-loop", action='store_true',
                        help="graph self-loop (default=False)")
    parser.set_defaults(self_loop=False)
    # * attention
    parser.add_argument("--num-heads", type=int, default=8,
                        help="number of hidden attention heads")
    parser.add_argument("--num-out-heads", type=int, default=1,
                        help="number of output attention heads")
    parser.add_argument("--num-layers", type=int, default=1,
                        help="number of hidden layers")
    parser.add_argument("--num-hidden", type=int, default=128,
                        help="number of hidden units")
    parser.add_argument("--residual", action="store_true", default=False,
                        help="use residual connection")
    parser.add_argument("--in-drop", type=float, default=.6,
                        help="input feature dropout")
    parser.add_argument("--attn-drop", type=float, default=.6,
                        help="attention dropout")
    parser.add_argument('--negative-slope', type=float, default=0.2,
                        help="the negative slope of leaky relu")
    parser.add_argument('--early-stop', action='store_true', default=False,
                        help="indicates whether to use early stop or not")
    parser.add_argument('--fastmode', action="store_true", default=False,
                        help="skip re-evaluate the validation set")
    # * MODEL
    parser.add_argument("--arch", type=str, default='gcn',
                        help='arch of gcn model, default: gcn')
    parser.add_argument("--coarse", action="store_false")
    # * dataset
    parser.add_argument("--level", type=int, default=2)
    parser.add_argument("--prefix", type=str,
                        default='graphzoom', help='dataset prefix')
    parser.add_argument("-pj", "--proj", type=str,
                        default='fusion', help="projection matrix type")
    args = parser.parse_args()
    args = EasyDict(vars(args))
    print(args)

    main(args)
