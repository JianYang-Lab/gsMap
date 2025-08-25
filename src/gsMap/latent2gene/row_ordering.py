"""
Row ordering optimization for cache efficiency
"""

import numpy as np
from typing import Optional
from numba import njit
import logging
import networkx as nx

logger = logging.getLogger(__name__)


@njit
def compute_jaccard_similarity(neighbors_a: np.ndarray, neighbors_b: np.ndarray) -> float:
    """Compute Jaccard similarity between two neighbor sets"""
    set_a = set(neighbors_a)
    set_b = set(neighbors_b)
    intersection = len(set_a & set_b)
    union = len(set_a | set_b)
    return intersection / union if union > 0 else 0.0


def optimize_row_order(
    neighbor_indices: np.ndarray,
    cell_indices: np.ndarray,
    method: Optional[str] = None,
    neighbor_weights: Optional[np.ndarray] = None
) -> np.ndarray:
    """
    Sort rows by shared neighbors to improve cache locality
    
    Args:
        neighbor_indices: (n_cells, k) array of neighbor indices (global indices)
        cell_indices: (n_cells,) array of global indices for each cell
        method: None (auto), 'weighted', 'greedy', 'graph', or 'none'
        neighbor_weights: Optional (n_cells, k) array of weights for each neighbor
    
    Returns:
        Reordered row indices (local indices 0 to n_cells-1)
    
    Complexity:
        - graph: O(n*k + n) - graph traversal approach
        - weighted: O(n*k) where k is number of neighbors - very efficient!
        - greedy: O(n²) - only for very small datasets
        - none: O(1) - returns original order
    """
    n_cells = len(neighbor_indices)
    
    # Auto-select method if None
    if method is None:
        # Use graph method by default for better handling of global indices
        method = 'graph'
    
    if method == 'graph':
        # Build directed graph from neighbor relationships
        G = nx.DiGraph()
        
        # Create mapping from global to local indices
        global_to_local = {global_idx: local_idx for local_idx, global_idx in enumerate(cell_indices)}
        
        # Add all cells as nodes (using local indices)
        G.add_nodes_from(range(n_cells))
        
        # Add edges based on neighbor relationships
        for i in range(n_cells):
            for j, neighbor_global_idx in enumerate(neighbor_indices[i]):
                # Check if neighbor is in our cell set
                if neighbor_global_idx in global_to_local:
                    neighbor_local_idx = global_to_local[neighbor_global_idx]
                    weight = neighbor_weights[i, j] if neighbor_weights is not None else 1.0
                    G.add_edge(i, neighbor_local_idx, weight=weight)
        
        # Try to find a good traversal order
        visited = set()
        ordered = []
        
        # Start with node with highest weighted degree
        if neighbor_weights is not None:
            weighted_degrees = dict(G.degree(weight='weight'))
            if weighted_degrees:
                start_node = max(weighted_degrees, key=weighted_degrees.get)
            else:
                start_node = 0
        else:
            degrees = dict(G.degree())
            start_node = max(degrees, key=degrees.get) if degrees else 0
        
        # BFS/DFS hybrid traversal prioritizing high-weight connections
        stack = [start_node]
        visited.add(start_node)
        ordered.append(start_node)
        
        while len(ordered) < n_cells:
            if stack:
                current = stack[-1]
                
                # Get unvisited neighbors sorted by edge weight
                neighbors = []
                for neighbor in G.neighbors(current):
                    if neighbor not in visited:
                        weight = G[current][neighbor].get('weight', 1.0)
                        neighbors.append((neighbor, weight))
                
                if neighbors:
                    # Sort by weight and pick highest
                    neighbors.sort(key=lambda x: x[1], reverse=True)
                    next_node = neighbors[0][0]
                    visited.add(next_node)
                    ordered.append(next_node)
                    stack.append(next_node)
                else:
                    # No unvisited neighbors, backtrack
                    stack.pop()
            else:
                # Stack empty, find next unvisited node with highest connectivity
                unvisited = set(range(n_cells)) - visited
                if unvisited:
                    # Pick unvisited node with most connections to visited nodes
                    best_node = None
                    best_score = -1
                    
                    for node in unvisited:
                        score = 0
                        # Count connections to visited nodes
                        for visited_node in visited:
                            if G.has_edge(node, visited_node):
                                score += G[node][visited_node].get('weight', 1.0)
                            if G.has_edge(visited_node, node):
                                score += G[visited_node][node].get('weight', 1.0)
                        
                        if score > best_score:
                            best_score = score
                            best_node = node
                    
                    if best_node is None:
                        # No connections, just pick any unvisited
                        best_node = next(iter(unvisited))
                    
                    visited.add(best_node)
                    ordered.append(best_node)
                    stack.append(best_node)
        
        return np.array(ordered)
    
    elif method == 'weighted' and neighbor_weights is not None:
        # Efficient weighted heuristic: follow highest-weight neighbors
        visited = np.zeros(n_cells, dtype=bool)
        ordered = []
        
        # Create mapping from global to local indices
        global_to_local = {global_idx: local_idx for local_idx, global_idx in enumerate(cell_indices)}
        
        # Start with cell that has highest max weight to any single neighbor
        max_weights = neighbor_weights.max(axis=1)
        current = np.argmax(max_weights)
        
        ordered.append(current)
        visited[current] = True
        
        # Build reverse lookup (using local indices)
        reverse_neighbors = [[] for _ in range(n_cells)]
        for i in range(n_cells):
            for j, neighbor_global_idx in enumerate(neighbor_indices[i]):
                if neighbor_global_idx in global_to_local:
                    neighbor_local_idx = global_to_local[neighbor_global_idx]
                    reverse_neighbors[neighbor_local_idx].append((i, neighbor_weights[i, j]))
        
        # Process all cells
        for _ in range(n_cells - 1):
            neighbors = neighbor_indices[current]
            weights = neighbor_weights[current]
            
            # Sort neighbors by weight (highest first)
            sorted_idx = np.argsort(weights)[::-1]
            
            # Find the unvisited neighbor with highest weight
            next_cell = None
            best_weight = -1
            
            for idx in sorted_idx:
                neighbor_global_idx = neighbors[idx]
                if neighbor_global_idx in global_to_local:
                    neighbor_local_idx = global_to_local[neighbor_global_idx]
                    if not visited[neighbor_local_idx]:
                        if weights[idx] > best_weight:
                            best_weight = weights[idx]
                            next_cell = neighbor_local_idx
            
            # If no unvisited direct neighbors, find closest unvisited cell
            if next_cell is None:
                connection_scores = np.zeros(n_cells)
                
                # Check connections from last few visited cells
                for cell_idx in ordered[-min(10, len(ordered)):]:
                    # Add forward connections
                    for j, neighbor_global_idx in enumerate(neighbor_indices[cell_idx]):
                        if neighbor_global_idx in global_to_local:
                            neighbor_local_idx = global_to_local[neighbor_global_idx]
                            if not visited[neighbor_local_idx]:
                                connection_scores[neighbor_local_idx] += neighbor_weights[cell_idx, j]
                    
                    # Add reverse connections
                    for reverse_idx, reverse_weight in reverse_neighbors[cell_idx]:
                        if not visited[reverse_idx]:
                            connection_scores[reverse_idx] += reverse_weight
                
                # Pick the unvisited cell with highest connection score
                if connection_scores.max() > 0:
                    unvisited_scores = connection_scores.copy()
                    unvisited_scores[visited] = -1
                    next_cell = np.argmax(unvisited_scores)
                else:
                    # No connections found, pick cell with highest max weight
                    unvisited_max_weights = max_weights.copy()
                    unvisited_max_weights[visited] = -1
                    next_cell = np.argmax(unvisited_max_weights)
            
            ordered.append(next_cell)
            visited[next_cell] = True
            current = next_cell
        
        return np.array(ordered)
    
    elif method == 'greedy':
        # Original greedy approach - only for small datasets
        if n_cells > 1000:
            logger.warning(f"Greedy method is O(n²) - not recommended for {n_cells} cells")
        
        # Create mapping from global to local indices
        global_to_local = {global_idx: local_idx for local_idx, global_idx in enumerate(cell_indices)}
        
        # Convert neighbor indices to sets of local indices for Jaccard computation
        neighbor_sets = []
        for i in range(n_cells):
            local_neighbors = set()
            for neighbor_global_idx in neighbor_indices[i]:
                if neighbor_global_idx in global_to_local:
                    local_neighbors.add(global_to_local[neighbor_global_idx])
            neighbor_sets.append(local_neighbors)
        
        remaining = set(range(n_cells))
        ordered = []
        
        current = np.random.choice(list(remaining))
        ordered.append(current)
        remaining.remove(current)
        
        while remaining:
            max_sim = -1
            next_row = None
            
            # Sample candidates for large datasets
            candidates = list(remaining)
            if len(candidates) > 100:
                candidates = np.random.choice(candidates, min(100, len(candidates)), replace=False)
            
            for candidate in candidates:
                # Use precomputed sets for Jaccard similarity
                intersection = len(neighbor_sets[current] & neighbor_sets[candidate])
                union = len(neighbor_sets[current] | neighbor_sets[candidate])
                sim = intersection / union if union > 0 else 0.0
                
                if sim > max_sim:
                    max_sim = sim
                    next_row = candidate
            
            ordered.append(next_row)
            remaining.remove(next_row)
            current = next_row
        
        return np.array(ordered)
    
    else:
        # Simple sequential order as fallback
        return np.arange(n_cells)