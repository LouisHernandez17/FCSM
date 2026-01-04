from __future__ import annotations

import argparse
import csv
import gzip
import json
import sys
from pathlib import Path
from typing import Iterable

import networkx as nx
import nltk
from nltk.corpus import wordnet

from assets.augmenter import GraphAugmenter

TARGET_RELATIONS = {
                    "/r/Causes",
                    # "/r/HasPrerequisite",
                    # "/r/HasSubevent",
                    # "/r/Entails",
                     }
DEFAULT_INPUT = Path("conceptnet-assertions-5.7.0.csv/assertions.csv")
DEFAULT_OUTPUT_DIR = Path("dataset/tier2_conceptnet")
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


def ensure_wordnet() -> None:
    try:
        nltk.data.find("corpora/wordnet")
    except LookupError:
        nltk.download("wordnet")


def get_wordnet_description(concept_name: str) -> str:
    clean_name = concept_name.replace("_", " ")
    synsets = wordnet.synsets(clean_name)
    if synsets:
        return synsets[0].definition()
    return f"The concept of {clean_name}"


def describe_concept(clean_name: str) -> str:
    if CONCEPT_LIB:
        desc = CONCEPT_LIB.get(clean_name)
        if desc:
            return desc
    return get_wordnet_description(clean_name)


def open_conceptnet_file(path: Path):
    if path.suffix == ".gz":
        return gzip.open(path, "rt", encoding="utf-8", errors="replace")
    return path.open("r", encoding="utf-8", errors="replace")


def load_conceptnet_graph(filepath: Path, target_relations: set[str]) -> nx.DiGraph:
    graph = nx.DiGraph()
    print("Streaming ConceptNet... (filtering for causal edges)")

    with open_conceptnet_file(filepath) as handle:
        reader = csv.reader(handle, delimiter="\t")
        for idx, row in enumerate(reader):
            if len(row) < 4:
                continue

            relation = row[1]
            start_node = row[2]
            end_node = row[3]

            if relation not in target_relations:
                continue
            if not (start_node.startswith("/c/en/") and end_node.startswith("/c/en/")):
                continue

            u = start_node.split("/")[3]
            v = end_node.split("/")[3]

            if len(u) <= 2 or len(v) <= 2:
                continue

            graph.add_edge(u, v, relation=relation)

            if idx > 0 and idx % 2_000_000 == 0:
                print(f"Scanned {idx} lines... graph has {graph.number_of_nodes()} nodes.")

    print(f"Finished streaming. Graph has {graph.number_of_nodes()} nodes and {graph.number_of_edges()} edges.")
    return graph


def extract_subgraphs(
    graph: nx.DiGraph,
    output_dir: Path,
    num_graphs: int,
    min_nodes: int,
    max_nodes: int,
    seed: int | None = None,
) -> None:
    augmenter = GraphAugmenter(seed=seed)
    output_dir.mkdir(parents=True, exist_ok=True)

    nodes_list = list(graph.nodes())
    if not nodes_list:
        raise ValueError("ConceptNet graph is empty; check the input file and filters.")

    generated = 0
    attempts = 0
    max_attempts = num_graphs * 10
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
        if not nx.is_directed_acyclic_graph(subgraph):
            continue
        if not nx.is_weakly_connected(subgraph):
            continue

        try:
            longest_path = nx.dag_longest_path(subgraph)
            if len(longest_path) < 3:
                continue
        except nx.NetworkXError:
            continue

        nodes_data = []
        for node in subgraph.nodes:
            clean_name = node.replace("_", " ")
            nodes_data.append({
                "id": node,
                "name": clean_name,
                "description": describe_concept(clean_name),
            })

        edges_data = [{"source": u, "target": v} for u, v in subgraph.edges]

        output = {
            "graph_id": f"cn_{generated}",
            "domain": "CommonSense",
            "source": "ConceptNet",
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
    parser = argparse.ArgumentParser(description="Build ConceptNet-based causal graph dataset using WordNet descriptions.")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT, help="Path to ConceptNet assertions file (.csv or .csv.gz)")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR, help="Directory to write JSON graphs")
    parser.add_argument("--num-graphs", type=int, default=5000, help="Number of subgraphs to sample")
    parser.add_argument("--min-nodes", type=int, default=5, help="Minimum number of nodes per subgraph")
    parser.add_argument("--max-nodes", type=int, default=15, help="Maximum number of nodes per subgraph")
    parser.add_argument("--seed", type=int, default=17, help="Random seed for reproducibility")
    parser.add_argument("--relations", nargs="*", default=sorted(TARGET_RELATIONS), help="Relations to keep from ConceptNet")
    return parser.parse_args(argv)


def main(argv: Iterable[str] | None = None) -> None:
    args = parse_args(argv)

    if not args.input.exists():
        raise SystemExit(f"Input file not found: {args.input}")

    ensure_wordnet()

    graph = load_conceptnet_graph(args.input, set(args.relations))
    extract_subgraphs(
        graph=graph,
        output_dir=args.output_dir,
        num_graphs=args.num_graphs,
        min_nodes=args.min_nodes,
        max_nodes=args.max_nodes,
        seed=args.seed,
    )


if __name__ == "__main__":
    main(sys.argv[1:])
