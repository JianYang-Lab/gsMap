import numpy as np
import torch
from scipy.spatial import cKDTree
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.utils import add_remaining_self_loops, degree


def build_spatial_graph(
        coords: np.ndarray,
        n_neighbors: int,
        undirected: bool = False
) -> np.ndarray:
    """

    Parameters:
    -----------
    coords : np.ndarray
        Spatial coordinates of shape (n_cells, n_dims)
    n_neighbors : int
        Number of nearest neighbors
    undirected : bool, default=True
        Whether to make graph undirected

    Returns:
    --------
    edge_array : np.ndarray
        Edge array of shape (n_edges, 2)
    """

    coords = np.ascontiguousarray(coords, dtype=np.float32)

    # Query k-NN
    tree = cKDTree(coords, balanced_tree=True, compact_nodes=True)
    _, indices = tree.query(coords, k=n_neighbors, workers=-1)

    n_nodes = coords.shape[0]

    if undirected:
        # Create bidirectional edges
        source = np.repeat(np.arange(n_nodes), n_neighbors)
        target = indices.flatten()

        # Combine forward and reverse edges
        all_edges = np.column_stack([
            np.concatenate([source, target]),
            np.concatenate([target, source])
        ])

        # Remove duplicates using set
        edge_set = {tuple(sorted([i, j])) for i, j in all_edges}
        return np.array(list(edge_set), dtype=np.int32)
    else:
        # Directed graph - just flatten the indices
        source = np.repeat(np.arange(n_nodes), n_neighbors)
        target = indices.flatten()
        return np.column_stack([source, target]).astype(np.int32)

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
