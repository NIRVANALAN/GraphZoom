from __future__ import print_function
import json
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from networkx.readwrite import json_graph


def run_regression(train_embeds, train_labels, test_embeds, test_labels):
    log = LogisticRegression(solver='liblinear', multi_class='ovr')
    log.fit(train_embeds, train_labels)
    pred_labels = (log.predict(test_embeds)).tolist()
    acc = accuracy_score(test_labels, pred_labels)
    import pdb
    pdb.set_trace()
    np.save('reddit_pred_labels.npy', pred_labels)
    print("Test Accuracy: {:.4f}".format(acc))


# def gcn_mlp_classifier(embeds, state_dict_path, model_arch, train_ids, test_ids):


def load_data(dataset_dir, dataset):
    if dataset in ['cora', 'citeseer', 'pubmed']:
        G = json_graph.node_link_graph(
            json.load(open(dataset_dir + "/{}-G.json".format(dataset))))
        labels = json.load(
            open(dataset_dir + "/{}-class_map.json".format(dataset)))
        train_ids = [n for n in G.nodes() if not G.node[n]['val']
                     and not G.node[n]['test']]
        test_ids = [n for n in G.nodes() if G.node[n]['test']]
        test_ids = test_ids[:1000]
        train_labels = [labels[str(i)] for i in train_ids]
        test_labels = [labels[str(i)] for i in test_ids]
    elif dataset in ['reddit', 'Amazon2M']:
        train_ids = np.load(dataset_dir+f'/{dataset}_train_data.npy')
        test_ids = np.load(dataset_dir+f'/{dataset}_test_data.npy')
        labels = np.load(dataset_dir+f'/{dataset}_labels.npy')
        # test_ids = test_ids[:1000]
        train_labels = labels[train_ids]
        test_labels = labels[test_ids]
    return labels, train_ids, test_ids, train_labels, test_labels


def lr(dataset_dir, data_dir, dataset):
    print("%%%%%% Starting Evaluation %%%%%%")
    print("Loading data...")
    labels, train_ids, test_ids, train_labels, test_labels = load_data(
        dataset_dir, dataset)
    embeds = np.load(data_dir)
    train_embeds = embeds[[id for id in train_ids]]
    test_embeds = embeds[[id for id in test_ids]]
    print("Running regression..")
    run_regression(train_embeds, train_labels, test_embeds, test_labels)
