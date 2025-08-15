import torch
import torch.nn as nn
import pandas as pd
from torch_geometric.nn import knn_graph
from torch_geometric.utils import to_undirected

from torch_geometric.nn.conv import MessagePassing
from torch_scatter import scatter_add
from torch_geometric.utils import add_remaining_self_loops


def full_block(in_dim, out_dim, p_drop=0.1):
    return nn.Sequential(
        nn.Linear(in_dim, out_dim),
        nn.BatchNorm1d(out_dim),
        nn.ReLU(),
        nn.Dropout(p=p_drop),
    )


def build_spatial_graph(adata, n_neighbors, spatial_key):
    """
    Build the spatial neighbor graphs.
    adata: AnnData object of scanpy package.
    n_neighbors: The number of nearest neighbors when model='KNN'
    """

    edge_index = to_undirected(
        knn_graph(
            x=torch.tensor(adata.obsm[spatial_key]),
            flow="target_to_source",
            k=n_neighbors,
            loop=True,
            num_workers=8,
        ),
        num_nodes=adata.shape[0],
    )

    graph_df = pd.DataFrame(edge_index.numpy().T, columns=["Cell1", "Cell2"])
    id_cell_trans = dict(zip(range(adata.n_obs), adata.obs_names))
    graph_df["Cell1"] = graph_df["Cell1"].map(id_cell_trans)
    graph_df["Cell2"] = graph_df["Cell2"].map(id_cell_trans)

    return edge_index, graph_df


# Define the GCN feature extractor
def sym_norm(edge_index, num_nodes, edge_weight=None, improved=False, dtype=None):
    if edge_weight is None:
        edge_weight = torch.ones(
            (edge_index.size(1),), dtype=dtype, device=edge_index.device
        )

    fill_value = 1 if not improved else 2
    edge_index, edge_weight = add_remaining_self_loops(
        edge_index, edge_weight, fill_value, num_nodes
    )

    # compute normalization
    row, col = edge_index
    deg = scatter_add(edge_weight, row, dim=0, dim_size=num_nodes)
    deg_inv_sqrt = deg.pow(-0.5)
    deg_inv_sqrt[deg_inv_sqrt == float("inf")] = 0

    return edge_index, deg_inv_sqrt[row] * edge_weight * deg_inv_sqrt[col]


class GCN(MessagePassing):
    """
    LGCN (GCN without learnable and concat)
    K: K-hop neighbor to propagate
    """

    def __init__(self, K=1, cached=False, bias=True, **kwargs):
        super().__init__(aggr="add", **kwargs)
        self.K = K

    def forward(self, x, edge_index, edge_weight=None):
        edge_index, norm = sym_norm(
            edge_index, x.size(0), edge_weight, dtype=x.dtype)

        xs = [x]
        if self.K == 0:
            return x
        for k in range(self.K):
            xs.append(self.propagate(edge_index, x=xs[-1], norm=norm))
        return torch.cat(xs[1:], dim=1)

    def message(self, x_j, norm):
        return norm.view(-1, 1) * x_j