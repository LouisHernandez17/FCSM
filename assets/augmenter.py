from __future__ import annotations

import random
from typing import Optional

import networkx as nx


class GraphAugmenter:
    def __init__(self, seed: Optional[int] = None) -> None:
        self.rng = random.Random(seed)

    def marginalize(self, base_graph: nx.DiGraph, drop_rate: float = 0.3) -> nx.DiGraph:
        """
        Remove a proportion of nodes and apply latent projection.
        Ensures the resulting graph remains connected (keeps largest component).
        """
        G = base_graph.copy()
        nodes = list(G.nodes())
        if len(nodes) <= 1:
            return G

        drop_rate = max(0.0, min(0.95, drop_rate))
        num_to_remove = max(0, min(int(len(nodes) * drop_rate), len(nodes) - 1))

        if num_to_remove == 0:
            return G

        targets = self.rng.sample(nodes, num_to_remove)
        for node in targets:
            if node in G:
                self._remove_and_project(G, node)

        return self._keep_largest_component(G)

    def _remove_and_project(self, G: nx.DiGraph, node) -> None:
        if node not in G:
            return
        parents = list(G.predecessors(node))
        children = list(G.successors(node))

        for p in parents:
            for c in children:
                if p == c:
                    continue
                if not G.has_edge(p, c):
                    G.add_edge(p, c)

        G.remove_node(node)

    def _keep_largest_component(self, G: nx.DiGraph) -> nx.DiGraph:
        if not G.nodes():
            return G
        components = sorted(nx.weakly_connected_components(G), key=len, reverse=True)
        if not components:
            return G
        largest = components[0]
        if len(largest) < len(G):
            return G.subgraph(largest).copy()
        return G

    def random_walk_sample(self, base_graph: nx.DiGraph, min_nodes: int = 5, max_nodes: int = 15) -> nx.DiGraph:
        """
        Random-walk-biased sampler to favor chains over stars. Keeps largest component.
        """
        nodes = list(base_graph.nodes())
        if not nodes:
            return base_graph

        current_node = self.rng.choice(nodes)
        selected_nodes = {current_node}

        attempts = 0
        while len(selected_nodes) < max_nodes and attempts < max_nodes * 2:
            attempts += 1
            neighbors = list(base_graph.successors(current_node))
            if not neighbors and self.rng.random() < 0.5:
                neighbors = list(base_graph.predecessors(current_node))

            if neighbors:
                next_node = self.rng.choice(neighbors)
                selected_nodes.add(next_node)
                current_node = next_node
            else:
                current_node = self.rng.choice(list(selected_nodes))

        if len(selected_nodes) < min_nodes:
            frontier = set()
            for n in selected_nodes:
                frontier.update(base_graph.successors(n))
                frontier.update(base_graph.predecessors(n))
            frontier -= selected_nodes
            needed = min_nodes - len(selected_nodes)
            if frontier:
                extra = list(frontier)
                self.rng.shuffle(extra)
                selected_nodes.update(extra[:needed])

        subgraph = base_graph.subgraph(selected_nodes).copy()
        return self._keep_largest_component(subgraph)

    snowball_sample = random_walk_sample
