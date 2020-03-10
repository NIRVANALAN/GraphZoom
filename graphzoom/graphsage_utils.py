from __future__ import print_function

from pathlib import Path
import numpy as np
import random
import json
import sys
import os

import networkx as nx
from networkx.readwrite import json_graph
# version_info = list(map(int, nx.__version__.split('.')))
# major = version_info[0]
# minor = version_info[1]
# assert (major <= 1) and (minor <= 11), "networkx major version > 1.11"

WALK_LEN = 5
N_WALKS = 50


def load_gsage_graph(prefix, normalize=True, load_walks=False):
    G_data = json.load(open(prefix + "-G.json"))
    G = json_graph.node_link_graph(G_data)
    if isinstance(G.nodes()[0], int):
        def conversion(n): return int(n)
    else:
        def conversion(n): return n
    id_map = json.load(open(prefix + "-id_map.json"))
    id_map = {conversion(k): int(v) for k, v in id_map.items()}
    class_map = json.load(open(prefix + "-class_map.json"))
    if isinstance(list(class_map.values())[0], list):
        def lab_conversion(n): return n
    else:
        def lab_conversion(n): return int(n)

    class_map = {conversion(k): lab_conversion(v)
                 for k, v in class_map.items()}

    # Remove all nodes that do not have val/test annotations
    # (necessary because of networkx weirdness with the Reddit data)
    broken_nodes = []
    for node in G.nodes():
        if not 'val' in G.node[node] or not 'test' in G.node[node]:
            broken_nodes.append(node)
    for node in broken_nodes:
        G.remove_node(node)
    print("Removed {:d} nodes that lacked proper annotations due to networkx versioning issues".format(
        len(broken_nodes)))

    # Make sure the graph has edge train_removed annotations
    # (some datasets might already have this..)
    print("Loaded data.. now preprocessing..")
    for edge in G.edges():
        if (G.node[edge[0]]['val'] or G.node[edge[1]]['val'] or
                G.node[edge[0]]['test'] or G.node[edge[1]]['test']):
            G[edge[0]][edge[1]]['train_removed'] = True
        else:
            G[edge[0]][edge[1]]['train_removed'] = False

    if normalize:
        norm_feats_path = Path(str(prefix)+'-norm-feats.npy')
        if not norm_feats_path.exists():
            feats = np.load(str(prefix)+'-feats.npy')
            from sklearn.preprocessing import StandardScaler
            train_ids = np.array([id_map[n] for n in G.nodes(
            ) if not G.node[n]['val'] and not G.node[n]['test']])
            train_feats = feats[train_ids]
            scaler = StandardScaler()
            scaler.fit(train_feats)
            feats = scaler.transform(feats)
            np.save(str(prefix)+'-norm-feats.npy', feats)
        else:
            feats = np.load(str(prefix)+'-norm-feats.npy')
    else:
        feats = np.load(str(prefix)+'-feats.npy')
    import pdb
    pdb.set_trace()
    return G, feats, class_map


def run_random_walks(G, nodes, num_walks=N_WALKS):
    pairs = []
    for count, node in enumerate(nodes):
        if G.degree(node) == 0:
            continue
        for i in range(num_walks):
            curr_node = node
            for j in range(WALK_LEN):
                next_node = random.choice(G.neighbors(curr_node))
                # self co-occurrences are useless
                if curr_node != node:
                    pairs.append((node, curr_node))
                curr_node = next_node
        if count % 1000 == 0:
            print("Done walks for", count, "nodes")
    return pairs


if __name__ == "__main__":
    """ Run random walks """
    graph_file = sys.argv[1]
    out_file = sys.argv[2]
    G_data = json.load(open(graph_file))
    G = json_graph.node_link_graph(G_data)
    nodes = [n for n in G.nodes() if not G.node[n]["val"]
             and not G.node[n]["test"]]
    G = G.subgraph(nodes)
    pairs = run_random_walks(G, nodes)
    with open(out_file, "w") as fp:
        fp.write("\n".join([str(p[0]) + "\t" + str(p[1]) for p in pairs]))