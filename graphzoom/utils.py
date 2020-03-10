import json
import time
from pathlib import Path
import networkx as nx
from pathlib import Path
import numpy as np
import networkx as nx
from networkx.readwrite import json_graph, read_gpickle
from networkx.linalg.laplacianmatrix import laplacian_matrix
from scipy.io import mmwrite, mmread
from scipy.sparse import csr_matrix
import sklearn

from dgl.data import RedditDataset


def load_dgl_graph(name, normalize=False, self_loop=False):
    if name == 'reddit':
        data = RedditDataset(self_loop=self_loop)
        if normalize:
            train_nid = np.nonzero(data.train_mask)[0].astype(np.int64)
            train_feats = data.features[train_nid]
            scaler = sklearn.preprocessing.StandardScaler()
            scaler.fit(train_feats)
            features = scaler.transform(data.features)
        else:
            features = data.features
        print('transforming DGLGraph to NXGraph...')
        return data.graph.to_networkx().to_undirected(), features
    pass


def load_nx_graph(dataset, prefix):
    pass


def load_dataset(dataset, prefix='',):
    mtx_path = Path("dataset/{}/{}.mtx".format(dataset, dataset))
    feats_path = Path(prefix, 'dataset', dataset, f'{dataset}-feats.npy')
    feats = np.load(str(feats_path))
    if mtx_path.exists():
        laplacian = mmread(str(mtx_path))
    else:
        if dataset in ['citeseer', 'cora', 'pubmed', ]:
            dataset_path = str(prefix) + '-G.json'
            G_data = json.load(
                open(dataset_path))
            G = json_graph.node_link_graph(G_data)
        elif dataset in ['Amazon2M', 'reddit', 'ppi']:
            start = time.time()
            G = read_gpickle(
                str(Path(prefix, 'dataset', dataset, f'{dataset}.gpickle')))
            print(f'gpickle load finish: {time.time() - start}')
        else:
            raise ValueError('dataset not known')
        print('calculating laplacian')
        start = time.time()
        laplacian = laplacian_matrix(G)
        print(f'calculating finished: {time.time() - start}')
        file = open(mtx_path, "wb")
        mmwrite(str(mtx_path), laplacian)
        file.close()
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
