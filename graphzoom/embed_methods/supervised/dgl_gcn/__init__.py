from .gat import GAT
from .gcn import GCN
from .multi_level_gcn import MultiLevelGCN
from .utils import _sample_mask,load_data
# from .gcn import GCN
# from .g_centroid import GATCentroid
# from . import utils
# from .gat import GAT

# # TODO: add centroid-driven message passing graph net
# FACTORY = {
#     'gcn': GCN,
#     'gat': GAT,
#     'centroid': GATCentroid}


# def create_model(name, g, **kwargs):
#     if name not in FACTORY:
#         raise NotImplementedError(f'{name} not in arch FACTORY')
#     return FACTORY[name](g, **kwargs)
