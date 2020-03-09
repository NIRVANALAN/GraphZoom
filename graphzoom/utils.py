import json
from pathlib import Path
import networkx as nx
import numpy as np
from networkx.readwrite import json_graph
from networkx.linalg.laplacianmatrix import laplacian_matrix
from scipy.io import mmwrite, mmread
from scipy.sparse import csr_matrix
from pathlib import Path

from graphsage_utils import load_gsage_data


def normalize_features(G, feats):
    from sklearn.preprocessing import StandardScaler
    train_ids = np.array([n for n in G.nodes(
    ) if not G.node[n]['val'] and not G.node[n]['test']])
    train_feats = feats[train_ids]
    scaler = StandardScaler()
    scaler.fit(train_feats)
    feats = scaler.transform(feats)
    return feats


def load_dataset(dataset, prefix, normalize_features=True):
    mtx_path = Path("dataset/{}/{}.mtx".format(dataset, dataset))
    prefix = Path(prefix, 'dataset', dataset, f'{dataset}')
    if mtx_path.exists():
        laplacian = mmread(str(mtx_path))
        pass
    else:
        if dataset in ['citeseer', 'cora', 'pubmed', ]:
            dataset_path = str(prefix) + '-G.json'
            G_data = json.load(
                open(dataset_path))
            G = json_graph.node_link_graph(G_data)
        elif dataset in ['Amazon2M', 'reddit', 'ppi', 'Amazon2M']:
            G, class_map = load_gsage_data()
            pass
        else:
            raise ValueError('dataset not known')
        laplacian = laplacian_matrix(G)
        file = open(mtx_path, "wb")
        mmwrite(mtx_path, laplacian)
        file.close()
        if normalize_features:
            feats = np.load(str(prefix)+'-feats.npy')
            feats = normalize_features(G, feats)
            np.save(str(prefix)+'-norm_feats.npy')
    if normalize_features:
        feats = np.load(str(prefix)+'-norm_feats.npy')
    else:
        feats = np.load(str(prefix)+'-feats.npy')
    return laplacian, feats


def json2mtx(dataset):
    G_data = json.load(open("dataset/{}/{}-G.json".format(dataset, dataset)))
    G = json_graph.node_link_graph(G_data)
    laplacian = laplacian_matrix(G)
    file = open("dataset/{}/{}.mtx".format(dataset, dataset), "wb")
    mmwrite("dataset/{}/{}.mtx".format(dataset, dataset), laplacian)
    file.close()

    return laplacian


def mtx2matrix(proj_name):
    data = []
    row = []
    col = []
    with open(proj_name) as ff:
        for i, line in enumerate(ff):
            info = line.split()
            if i == 0:
                NumReducedNodes = int(info[0])
                NumOriginNodes = int(info[1])
            else:
                row.append(int(info[0])-1)
                col.append(int(info[1])-1)
                data.append(1)
    matrix = csr_matrix((data, (row, col)), shape=(
        NumReducedNodes, NumOriginNodes))
    return matrix


def mtx2graph(mtx_path):
    G = nx.Graph()
    with open(mtx_path) as ff:
        for i, line in enumerate(ff):
            info = line.split()
            if i == 0:
                num_nodes = int(info[0])
            elif int(info[0]) < int(info[1]):
                G.add_edge(int(info[0])-1, int(info[1]) -
                           1, wgt=abs(float(info[2])))
    # add isolated nodes
    for i in range(num_nodes):
        G.add_node(i)
    return G


def read_levels(level_path):
    with open(level_path) as ff:
        levels = int(ff.readline()) - 1
    return levels


def read_time(cputime_path):
    with open(cputime_path) as ff:
        cpu_time = float(ff.readline())
    return cpu_time


def construct_proj_laplacian(laplacian, levels, proj_dir):
    coarse_laplacian = []
    projections = []
    for i in range(levels):
        projection_name = "{}/Projection_{}.mtx".format(proj_dir, i+1)
        projection = mtx2matrix(projection_name)
        projections.append(projection)
        coarse_laplacian.append(laplacian)
        if i != (levels-1):
            laplacian = projection @ laplacian @ (projection.transpose())
    return projections, coarse_laplacian
