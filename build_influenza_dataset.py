from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Iterable

import networkx as nx

from assets.augmenter import GraphAugmenter

DEFAULT_INPUT = Path("A network leading to influenza onset obtained through our network structure estimation..cx")
DEFAULT_OUTPUT_DIR = Path("dataset/influenza")
DICT_PATH = Path("dataset/node_dictionary.json")


def load_concept_dictionary() -> dict[str, str]:
    if not DICT_PATH.exists():
        return {}
    try:
        return json.loads(DICT_PATH.read_text())
    except json.JSONDecodeError:
        print(f"Warning: could not parse {DICT_PATH}; ignoring dictionary.")
        return {}


CONCEPT_LIB = load_concept_dictionary()


def describe_node(name: str) -> str:
    if CONCEPT_LIB:
        desc = CONCEPT_LIB.get(name)
        if desc:
            return desc
    return f"The variable {name}"


def load_cx_data(path: Path) -> list[dict]:
    return json.loads(path.read_text(encoding="utf-8"))


def get_cx_section(data: list[dict], key: str):
    for element in data:
        if key in element:
            return element[key]
    raise KeyError(f"CX data missing '{key}' section")


def load_influenza_graph(path: Path) -> nx.DiGraph:
    data = load_cx_data(path)
    nodes = get_cx_section(data, "nodes")
    edges = get_cx_section(data, "edges")

    id_to_name: dict[int, str] = {}
    graph = nx.DiGraph()

    for node in nodes:
        node_id = node.get("@id")
        name = node.get("n") or str(node_id)
        id_to_name[node_id] = name
        graph.add_node(name, original_id=node_id)

    missing = 0
    for edge in edges:
        source_id = edge.get("s")
        target_id = edge.get("t")
        source = id_to_name.get(source_id)
        target = id_to_name.get(target_id)
        if source is None or target is None:
            missing += 1
            continue
        graph.add_edge(source, target)

    if missing:
        print(f"Warning: skipped {missing} edges with missing node references.")

    print(f"Loaded influenza graph with {graph.number_of_nodes()} nodes and {graph.number_of_edges()} edges.")
    return graph


def extract_subgraphs(
    graph: nx.DiGraph,
    output_dir: Path,
    num_graphs: int,
    min_nodes: int,
    max_nodes: int,
    seed: int | None = None,
    require_dag: bool = True,
    min_path_len: int = 3,
) -> None:
    augmenter = GraphAugmenter(seed=seed)
    output_dir.mkdir(parents=True, exist_ok=True)

    nodes_list = list(graph.nodes())
    if not nodes_list:
        raise ValueError("Influenza graph is empty; check the input file.")

    generated = 0
    attempts = 0
    max_attempts = max(num_graphs * 20, num_graphs)
    print(f"Sampling {num_graphs} graphs using depth-biased random walks on {len(nodes_list)} nodes...")

    while generated < num_graphs and attempts < max_attempts:
        attempts += 1

        subgraph = augmenter.random_walk_sample(
            graph,
            min_nodes=min_nodes,
            max_nodes=max_nodes,
        )

        if subgraph.number_of_nodes() < min_nodes:
            continue
        if not nx.is_weakly_connected(subgraph):
            continue

        if require_dag:
            if not nx.is_directed_acyclic_graph(subgraph):
                continue
            if min_path_len > 0:
                try:
                    longest_path = nx.dag_longest_path(subgraph)
                    if len(longest_path) < min_path_len:
                        continue
                except nx.NetworkXError:
                    continue

        nodes_data = []
        for node in subgraph.nodes:
            nodes_data.append({
                "id": node,
                "name": node,
                "description": describe_node(node),
            })

        edges_data = [{"source": u, "target": v} for u, v in subgraph.edges]

        output = {
            "graph_id": f"influenza_{generated}",
            "domain": "Biomedicine",
            "source": "InfluenzaNetwork",
            "nodes": nodes_data,
            "edges": edges_data,
        }

        output_path = output_dir / f"graph_{generated}.json"
        output_path.write_text(json.dumps(output, indent=2))

        generated += 1
        if generated % 100 == 0:
            print(f"Saved {generated} graphs to {output_dir}.")

    if generated < num_graphs:
        print(f"Warning: requested {num_graphs}, generated {generated} after {attempts} attempts.")
    print(f"Completed. Generated {generated} graphs in {output_dir}.")


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build influenza subgraph dataset from the CX network.")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT, help="Path to influenza CX file")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR, help="Directory to write JSON graphs")
    parser.add_argument("--num-graphs", type=int, default=500, help="Number of subgraphs to sample")
    parser.add_argument("--min-nodes", type=int, default=5, help="Minimum number of nodes per subgraph")
    parser.add_argument("--max-nodes", type=int, default=15, help="Maximum number of nodes per subgraph")
    parser.add_argument("--seed", type=int, default=17, help="Random seed for reproducibility")
    parser.add_argument("--allow-cycles", action="store_true", help="Allow cyclic subgraphs (skip DAG filter)")
    parser.add_argument("--min-path-len", type=int, default=3, help="Minimum longest path length when requiring DAGs")
    return parser.parse_args(argv)


def main(argv: Iterable[str] | None = None) -> None:
    args = parse_args(argv)

    if not args.input.exists():
        raise SystemExit(f"Input file not found: {args.input}")

    graph = load_influenza_graph(args.input)
    extract_subgraphs(
        graph=graph,
        output_dir=args.output_dir,
        num_graphs=args.num_graphs,
        min_nodes=args.min_nodes,
        max_nodes=args.max_nodes,
        seed=args.seed,
        require_dag=not args.allow_cycles,
        min_path_len=args.min_path_len,
    )


if __name__ == "__main__":
    main(sys.argv[1:])
