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

from graphzoom.embed_methods.supervised.dgl_gcn import (GAT, GCN, _sample_mask, MultiLevelGCN,
                                                        load_data)
from graphzoom.utils import construct_proj_laplacian, mtx2graph
import torch
import numpy as np
from scipy.sparse import coo_matrix, csr_matrix 

FACTORY = {
    'gcn': GCN,
    'multi_level_gcn': MultiLevelGCN,
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
            g = DGLGraph(data.graph)
            val_ids = test_ids[1000:1500]
            test_ids = test_ids[:1000]
            test_mask = _sample_mask(test_ids, labels.shape[0])
            val_mask = _sample_mask(val_ids, labels.shape[0])
            data = EasyDict({
                'graph': data.graph,
                'labels': labels,
                'onehot_labels': onehot_labels,
                # 'features': feats,
                'features': data.features,
                'train_mask': train_mask,
                'val_mask': val_mask,
                'test_mask': test_mask,
                'num_labels': onehot_labels.shape[1],
                'coarse': False
            })
            # g = DGLGraph(data.graph)
    print(f'load data finished: {time.time() - load_data_time}')
    if args.coarse:
        # * load projection matrix
        levels = args.level
        reduce_results = f"graphzoom/reduction_results/{dataset}/fusion/"
        projections, coarse_adj = construct_proj_laplacian(
            original_adj, levels, reduce_results)
        # *calculate coarse feature, labels
        # label_mask = np.expand_dims(data.train_mask, 1)
        # coarse_labels = projections[0] @ (onehot_labels * label_mask)
        print('creating coarse DGLGraph')
        start = time.process_time()
        # ? what will happen if g is assigned to other variables later
        multi_level_dglgraph = [g]
        data.features = projections[0] @ data.features
        for i in range(1, levels):
            g = DGLGraph()
            g.from_scipy_sparse_matrix(coarse_adj[i])
            multi_level_dglgraph.append(g)
            if i==levels-1:
                break
            data.features = projections[i] @ data.features
        multi_level_dglgraph.reverse()
        projections.reverse()
        for projection in range(len(projections)):
            coo = projections[projection].tocoo()
            # coo = coo_matrix(([3,4,5], ([0,1,1], [2,0,2])), shape=(2,3))
            values = coo.data
            indices = np.vstack((coo.row, coo.col))

            i = torch.LongTensor(indices)
            v = torch.FloatTensor(values)
            projections[projection] = torch.sparse.FloatTensor(i, v, torch.Size(coo.shape)).cuda()
        print(f'creating finished in {time.process_time() - start}')
        # * new train/test masks

        # *replace data
        # coarse_data = EasyDict({
        #     'graph': g,
        #     'labels': coarse_labels,
        #     #     'onehot_labels': onehot_labels,
        #     'features': coarse_feats,
        #     'train_mask': coarse_train_mask,
        #     'val_mask': coarse_val_mask,
        #     'test_mask': coarse_test_mask,
        #     'num_classes': norm_coarse_labels.shape[1],
        #     'num_labels': onehot_labels.shape[1],
        #     'coarse': True
        # })
        # data = coarse_data
    # if args.coarse:
    #     labels = torch.FloatTensor(data.labels)
    #     loss_fcn = torch.nn.KLDivLoss(reduction='batchmean')
    #     print('training coarse')
    # else:
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
    # add self loop
    if args.self_loop or args.arch == 'gat':
        for i in range(len(multi_level_dglgraph)):
            multi_level_dglgraph[i] = add_self_loop(multi_level_dglgraph[i])
        print('add self_loop')
    n_edges = multi_level_dglgraph[0].number_of_edges()
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
    model = create_model(args.arch, multi_level_dglgraph,
                         num_layers=args.level-1,
                         in_dim=in_feats,
                         num_hidden=args.num_hidden,
                         num_classes=n_classes,
                         heads=heads,
                         #  activation=F.elu,
                         feat_drop=args.in_drop,
                         attn_drop=args.attn_drop,
                         negative_slope=args.negative_slope,
                         residual=args.residual, log_softmax=args.coarse, projection_matrix=projections)

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
        loss = loss_fcn(logits[train_mask], labels[train_mask])

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if epoch >= 3:
            dur.append(time.time() - t0)

        # if not args.coarse:
        acc = evaluate(model, features, labels, val_mask)
        print("Epoch {:05d} | Time(s) {:.4f} | Loss {:.4f} | Accuracy {:.4f} | "
              "ETputs(KTEPS) {:.2f}". format(epoch, np.mean(dur), loss.item(),
                                             acc, n_edges / np.mean(dur) / 1000))
    print(f'training time: {time.time() - start}')
    # if not args.coarse:
    acc = evaluate(model, features, labels, test_mask)
    # print(h.shape)
    # np.save(f'embeddings/{(args.arch).upper()}_{dataset}_emb_level_1_mask',
    #         h.detach().cpu().numpy())
    # torch.save(model.state_dict(),
    #            f'embeddings/{(args.arch).upper()}_{dataset}_emb_level_1_params.pth.tar',)
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
    parser.add_argument("--num-layers", type=int, default=2,
                        help="number of hidden layers")
    parser.add_argument("--num-hidden", type=int, default=16,
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
    parser.add_argument("--coarse", action="store_true", default=False)
    parser.add_argument("--level", type=int, default=2)
    # * dataset
    parser.add_argument("--prefix", type=str,
                        default='graphzoom', help='dataset prefix')
    args = parser.parse_args()
    args = EasyDict(vars(args))
    print(args)

    main(args)
