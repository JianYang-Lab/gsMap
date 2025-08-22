import torch
import torch.nn as nn
import pandas as pd
from torch_geometric.nn import knn_graph
from torch_geometric.utils import to_undirected
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.utils import add_remaining_self_loops, degree


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


class GCN(MessagePassing):
    """
    GCN for unweighted graphs.
    """

    def __init__(self, K=1):
        super().__init__(aggr="add")
        self.K = K

    def forward(self, x, edge_index):
        # Add self-loops
        edge_index, _ = add_remaining_self_loops(edge_index, num_nodes=x.size(0))

        # Compute normalization: 1/sqrt(deg_i * deg_j)
        row, col = edge_index
        deg = degree(row, x.size(0), dtype=x.dtype)
        norm = (deg[row] * deg[col]).pow(-0.5)
        norm[norm == float("inf")] = 0

        # K-hop propagation
        xs = [x]
        for _ in range(self.K):
            xs.append(self.propagate(edge_index, x=xs[-1], norm=norm))

        return torch.cat(xs[1:], dim=1)

    def message(self, x_j, norm):
        return norm.view(-1, 1) * x_j
