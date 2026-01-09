from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Iterable

import networkx as nx

from assets.augmenter import GraphAugmenter
from build_conceptnet_dataset import describe_concept, ensure_wordnet

DEFAULT_INPUT = Path("causenet-full.jsonl")
DEFAULT_OUTPUT_DIR = Path("dataset/causenet")


def normalize_concept(concept: str) -> str:
    return concept.strip().replace(" ", "_")


def load_causenet_graph(filepath: Path, min_support: int = 1) -> nx.DiGraph:
    graph = nx.DiGraph()
    print("Streaming CauseNet... (loading causal edges)")

    bad_lines = 0
    kept = 0
    with filepath.open("r", encoding="utf-8", errors="replace") as handle:
        for idx, line in enumerate(handle):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                bad_lines += 1
                continue

            relation = record.get("causal_relation", {})
            cause = relation.get("cause", {}).get("concept")
            effect = relation.get("effect", {}).get("concept")
            if not cause or not effect:
                continue

            cause_id = normalize_concept(str(cause))
            effect_id = normalize_concept(str(effect))
            if len(cause_id) <= 2 or len(effect_id) <= 2:
                continue
            if cause_id == effect_id:
                continue

            support = record.get("support")
            if support is None:
                support = len(record.get("sources", []))
            try:
                support_val = int(support)
            except (TypeError, ValueError):
                support_val = 1
            if support_val < min_support:
                continue

            if graph.has_edge(cause_id, effect_id):
                graph[cause_id][effect_id]["support"] += support_val
            else:
                graph.add_edge(cause_id, effect_id, support=support_val)
            kept += 1

            if idx > 0 and idx % 2_000_000 == 0:
                print(f"Scanned {idx} lines... graph has {graph.number_of_nodes()} nodes.")

    if bad_lines:
        print(f"Warning: skipped {bad_lines} malformed lines.")
    print(f"Finished streaming. Kept {kept} edges across {graph.number_of_nodes()} nodes.")
    return graph


def break_cycles(graph: nx.DiGraph, rng) -> nx.DiGraph:
    if graph.number_of_edges() == 0:
        return graph
    while True:
        try:
            cycle = nx.find_cycle(graph, orientation="original")
        except nx.NetworkXNoCycle:
            break
        u, v, _ = rng.choice(cycle)
        if graph.has_edge(u, v):
            graph.remove_edge(u, v)
        if graph.number_of_edges() == 0:
            break
    return graph


def extract_subgraphs(
    graph: nx.DiGraph,
    output_dir: Path,
    num_graphs: int,
    min_nodes: int,
    max_nodes: int,
    seed: int | None = None,
    allow_cycles: bool = False,
    break_cycles_enabled: bool = True,
    min_path_len: int = 3,
) -> None:
    augmenter = GraphAugmenter(seed=seed)
    output_dir.mkdir(parents=True, exist_ok=True)

    nodes_list = list(graph.nodes())
    if not nodes_list:
        raise ValueError("CauseNet graph is empty; check the input file and filters.")

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
        if break_cycles_enabled and not allow_cycles:
            subgraph = break_cycles(subgraph, augmenter.rng)
        if not nx.is_weakly_connected(subgraph):
            continue

        if not allow_cycles:
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
            clean_name = node.replace("_", " ")
            nodes_data.append({
                "id": node,
                "name": clean_name,
                "description": describe_concept(clean_name),
            })

        edges_data = [{"source": u, "target": v} for u, v in subgraph.edges]

        output = {
            "graph_id": f"causenet_{generated}",
            "domain": "CausalText",
            "source": "CauseNet",
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
    parser = argparse.ArgumentParser(description="Build CauseNet-based causal graph dataset using WordNet descriptions.")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT, help="Path to CauseNet JSONL file")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR, help="Directory to write JSON graphs")
    parser.add_argument("--num-graphs", type=int, default=5000, help="Number of subgraphs to sample")
    parser.add_argument("--min-nodes", type=int, default=5, help="Minimum number of nodes per subgraph")
    parser.add_argument("--max-nodes", type=int, default=15, help="Maximum number of nodes per subgraph")
    parser.add_argument("--seed", type=int, default=17, help="Random seed for reproducibility")
    parser.add_argument("--min-support", type=int, default=1, help="Minimum support value to keep a causal relation")
    parser.add_argument("--allow-cycles", action="store_true", help="Allow cyclic subgraphs (skip DAG filter)")
    parser.add_argument("--break-cycles", action=argparse.BooleanOptionalAction, default=True, help="Break cycles instead of discarding samples")
    parser.add_argument("--min-path-len", type=int, default=3, help="Minimum longest path length when requiring DAGs")
    return parser.parse_args(argv)


def main(argv: Iterable[str] | None = None) -> None:
    args = parse_args(argv)

    if not args.input.exists():
        raise SystemExit(f"Input file not found: {args.input}")

    ensure_wordnet()

    graph = load_causenet_graph(args.input, min_support=args.min_support)
    extract_subgraphs(
        graph=graph,
        output_dir=args.output_dir,
        num_graphs=args.num_graphs,
        min_nodes=args.min_nodes,
        max_nodes=args.max_nodes,
        seed=args.seed,
        allow_cycles=args.allow_cycles,
        break_cycles_enabled=args.break_cycles,
        min_path_len=args.min_path_len,
    )


if __name__ == "__main__":
    main(sys.argv[1:])
