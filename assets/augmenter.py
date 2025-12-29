from __future__ import annotations

import random
from typing import Optional

import networkx as nx


class GraphAugmenter:
    def __init__(self, seed: Optional[int] = None) -> None:
        self.rng = random.Random(seed)

    def marginalize(self, base_graph: nx.DiGraph, drop_rate: float = 0.3) -> nx.DiGraph:
        """
        Remove a proportion of nodes and apply latent projection (wire-through of chains).
        Forks/colliders are left untouched because they lack both parents and children simultaneously.
        """
        G = base_graph.copy()
        nodes = list(G.nodes())
        if len(nodes) <= 1:
            return G

        # Clamp drop_rate and ensure at least one node remains.
        drop_rate = max(0.0, min(0.95, drop_rate))
        num_to_remove = max(0, min(int(len(nodes) * drop_rate), len(nodes) - 1))
        if num_to_remove == 0:
            return G

        targets = self.rng.sample(nodes, num_to_remove)
        for node in targets:
            if node in G:
                self._remove_and_project(G, node)
        return G

    def _remove_and_project(self, G: nx.DiGraph, node) -> None:
        parents = list(G.predecessors(node))
        children = list(G.successors(node))

        for p in parents:
            for c in children:
                if p == c:
                    continue
                if not G.has_edge(p, c):
                    G.add_edge(p, c)

        G.remove_node(node)

    def snowball_sample(self, base_graph: nx.DiGraph, min_nodes: int = 5, max_nodes: int = 15) -> nx.DiGraph:
        """
        Extract a connected-ish neighborhood by expanding from a random seed.
        """
        nodes = list(base_graph.nodes())
        if len(nodes) <= min_nodes:
            return base_graph.copy()

        seed_node = self.rng.choice(nodes)
        selected = {seed_node}
        frontier = set(base_graph.successors(seed_node)) | set(base_graph.predecessors(seed_node))

        target_size = self.rng.randint(min_nodes, min(len(nodes), max_nodes))

        while len(selected) < target_size and frontier:
            new_node = self.rng.choice(list(frontier))
            frontier.remove(new_node)
            selected.add(new_node)

            neighbors = set(base_graph.successors(new_node)) | set(base_graph.predecessors(new_node))
            frontier.update(neighbors - selected)

        return base_graph.subgraph(list(selected)).copy()
