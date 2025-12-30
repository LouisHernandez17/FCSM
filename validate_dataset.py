from __future__ import annotations

import argparse
import gzip
import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, List, Sequence

import networkx as nx
import numpy as np


@dataclass
class GraphStats:
    path: Path
    nodes: int
    edges: int
    density: float
    edges_per_node: float
    is_dag: bool
    is_weakly_connected: bool
    isolates: int


@dataclass
class Aggregate:
    valid_dags: int = 0
    cycles_found: int = 0
    disconnected: int = 0
    total_files: int = 0
    node_counts: List[int] = field(default_factory=list)
    edge_counts: List[int] = field(default_factory=list)
    densities: List[float] = field(default_factory=list)
    edges_per_node: List[float] = field(default_factory=list)

    def update(self, stats: GraphStats) -> None:
        self.total_files += 1
        if stats.is_dag:
            self.valid_dags += 1
        else:
            self.cycles_found += 1
        if not stats.is_weakly_connected:
            self.disconnected += 1
        self.node_counts.append(stats.nodes)
        self.edge_counts.append(stats.edges)
        self.densities.append(stats.density)
        self.edges_per_node.append(stats.edges_per_node)

    def summary(self) -> dict:
        if self.total_files == 0:
            return {}
        arr_nodes = np.array(self.node_counts)
        arr_edges = np.array(self.edge_counts)
        arr_density = np.array(self.densities)
        arr_edges_per_node = np.array(self.edges_per_node)
        return {
            "total_files": int(self.total_files),
            "valid_dags": int(self.valid_dags),
            "cycles_found": int(self.cycles_found),
            "disconnected": int(self.disconnected),
            "avg_nodes": float(arr_nodes.mean()),
            "avg_edges": float(arr_edges.mean()),
            "avg_density": float(arr_density.mean()),
            "p10_density": float(np.percentile(arr_density, 10)),
            "p90_density": float(np.percentile(arr_density, 90)),
            "avg_edges_per_node": float(arr_edges_per_node.mean()),
            "p10_edges_per_node": float(np.percentile(arr_edges_per_node, 10)),
            "p90_edges_per_node": float(np.percentile(arr_edges_per_node, 90)),
            "min_nodes": int(arr_nodes.min()),
            "max_nodes": int(arr_nodes.max()),
            "min_edges": int(arr_edges.min()),
            "max_edges": int(arr_edges.max()),
        }


def read_json(path: Path) -> dict:
    if path.suffix == ".gz":
        with gzip.open(path, "rt", encoding="utf-8") as f:
            return json.load(f)
    return json.loads(path.read_text())


def to_digraph(data: dict) -> nx.DiGraph:
    G = nx.DiGraph()
    nodes = data.get("nodes", [])
    edges = data.get("edges", [])

    # nodes can be list of dicts with id/name
    node_ids = []
    for n in nodes:
        if isinstance(n, dict) and "id" in n:
            node_ids.append(n["id"])
        else:
            node_ids.append(n)
    G.add_nodes_from(node_ids)

    # edges can be list of dicts {source,target} or pairs
    for e in edges:
        if isinstance(e, dict) and "source" in e and "target" in e:
            G.add_edge(e["source"], e["target"])
        elif isinstance(e, (list, tuple)) and len(e) == 2:
            G.add_edge(e[0], e[1])
    return G


def graph_stats(path: Path) -> GraphStats:
    data = read_json(path)
    G = to_digraph(data)

    n = G.number_of_nodes()
    e = G.number_of_edges()
    density = e / (n * (n - 1)) if n > 1 else 0.0
    edges_per_node = e / n if n > 0 else 0.0
    is_dag = nx.is_directed_acyclic_graph(G)
    is_weak = nx.is_weakly_connected(G) if n > 0 else True
    isolates = len(list(nx.isolates(G)))

    return GraphStats(path=path, nodes=n, edges=e, density=density, edges_per_node=edges_per_node, is_dag=is_dag, is_weakly_connected=is_weak, isolates=isolates)


def validate(paths: Sequence[Path]) -> Aggregate:
    agg = Aggregate()
    for p in paths:
        try:
            stats = graph_stats(p)
            agg.update(stats)
            if not stats.is_dag:
                print(f"⚠️ Cycle detected: {p}")
            elif stats.density > 0.5:
                print(f"⚠️ High density ({stats.density:.2f}) in {p}")
        except Exception as exc:  # noqa: BLE001
            print(f"❌ Error on {p}: {exc}")
    return agg


def collect_files(root: Path, include_glob: str) -> List[Path]:
    return sorted(root.glob(include_glob))


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate dataset graphs for structural integrity and distribution heuristics.")
    parser.add_argument("--root", type=Path, default=Path("dataset"), help="Root directory containing graph JSONs")
    parser.add_argument("--pattern", type=str, default="**/*.json", help="Glob pattern under root to include")
    parser.add_argument("--limit", type=int, default=0, help="Optional limit on number of files to check (0 = all)")
    parser.add_argument("--report", type=Path, help="Optional path to write JSON summary report")
    return parser.parse_args(argv)


def main(argv: Iterable[str] | None = None) -> None:
    args = parse_args(argv)
    files = collect_files(args.root, args.pattern)
    if args.limit and args.limit > 0:
        files = files[: args.limit]
    print(f"🔍 Scanning {len(files)} files under {args.root}...")
    agg = validate(files)

    summary = agg.summary()
    if not summary:
        print("No files processed.")
        return

    print("\n" + "=" * 40)
    print("📊 DATASET HEALTH REPORT")
    print("=" * 40)
    print(f"Total Files Checked: {summary['total_files']}")
    print(f"✅ Valid DAGs:       {summary['valid_dags']}")
    print(f"❌ Cycles Found:     {summary['cycles_found']} (critical)")
    print(f"⚠️ Disconnected:     {summary['disconnected']} (warning)")
    print("-" * 20)
    print(f"Avg Nodes:   {summary['avg_nodes']:.2f} (min {summary['min_nodes']}, max {summary['max_nodes']})")
    print(f"Avg Edges:   {summary['avg_edges']:.2f} (min {summary['min_edges']}, max {summary['max_edges']})")
    print(f"Avg Density: {summary['avg_density']:.2f} (p10 {summary['p10_density']:.2f}, p90 {summary['p90_density']:.2f})")
    print(f"Edges/Node:  {summary['avg_edges_per_node']:.2f} (p10 {summary['p10_edges_per_node']:.2f}, p90 {summary['p90_edges_per_node']:.2f})")
    if summary['avg_density'] > 0.5:
        print("⚠️ WARNING: High mean density (>0.5). Marginalization may be too aggressive.")
    elif summary['avg_density'] < 0.05:
        print("⚠️ WARNING: Very sparse mean density (<0.05). Graphs may be too simple.")
    elif summary['avg_edges_per_node'] < 0.8:
        print("⚠️ WARNING: Low edges-per-node (<0.8). Graphs may be too chain-like.")

    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(summary, indent=2))
        print(f"Report written to {args.report}")


if __name__ == "__main__":
    main()
