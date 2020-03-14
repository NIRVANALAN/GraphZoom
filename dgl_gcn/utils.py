from easydict import EasyDict
import json
import torch
import torch.nn.functional as F
import numpy as np
from networkx.readwrite import json_graph, read_gpickle
import networkx as nx


def _sample_mask(idx, l):
    """Create mask."""
    mask = np.zeros(l)
    mask[idx] = 1
    return mask


# train_prefix = '../graphzoom/dataset/cora/cora'
# G, features, class_map = load_data(train_prefix)
def load_data(dataset_dir, dataset):
    feats = np.load(dataset_dir + f"/{dataset}-feats.npy")
    if dataset in ['cora', 'citeseer', 'pubmed']:
        G = json_graph.node_link_graph(
            json.load(open(dataset_dir + "/{}-G.json".format(dataset))))
        labels = json.load(
            open(dataset_dir + "/{}-class_map.json".format(dataset)))
        train_ids = [n for n in G.nodes() if not G.node[n]['val']
                     and not G.node[n]['test']]
        test_ids = [n for n in G.nodes() if G.node[n]['test']]
        train_labels = [labels[str(i)] for i in train_ids]
        test_labels = [labels[str(i)] for i in test_ids]
        labels = list(labels.values())
    elif dataset in ['reddit', 'Amazon2M']:
        G = read_gpickle(
            dataset_dir + f'/{dataset}.gpickle')
        train_ids = np.load(dataset_dir+f'/{dataset}_train_data.npy')
        test_ids = np.load(dataset_dir+f'/{dataset}_test_data.npy')
        labels = np.load(dataset_dir+f'/{dataset}_labels.npy')
        train_labels = labels[train_ids]
        test_labels = labels[test_ids]
    return G, labels, train_ids, test_ids, train_labels, test_labels, feats


class EarlyStopping:
    def __init__(self, patience=10):
        self.patience = patience
        self.counter = 0
        self.best_score = None
        self.early_stop = False

    def step(self, acc, model):
        score = acc
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(model)
        elif score < self.best_score:
            self.counter += 1
            print(
                f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(model)
            self.counter = 0
        return self.early_stop

    def save_checkpoint(self, model):
        '''Saves model when validation loss decrease.'''
        torch.save(model.state_dict(), 'es_checkpoint.pt')
