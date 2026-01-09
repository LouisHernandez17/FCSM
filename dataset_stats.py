from __future__ import annotations

import argparse
import csv
import gzip
import json
import statistics
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Iterator, Sequence

from build_conceptnet_dataset import TARGET_RELATIONS


@dataclass
class DatasetStats:
    name: str
    path: Path
    graph_count: int = 0
    nodes_per_graph: list[int] = field(default_factory=list)
    edges_per_graph: list[int] = field(default_factory=list)
    edges_per_node: list[float] = field(default_factory=list)
    unique_nodes: set[str] = field(default_factory=set)

    def total_nodes(self) -> int:
        return int(sum(self.nodes_per_graph))

    def total_edges(self) -> int:
        return int(sum(self.edges_per_graph))


def read_json(path: Path) -> dict:
    if path.suffix == ".gz":
        with gzip.open(path, "rt", encoding="utf-8") as f:
            return json.load(f)
    return json.loads(path.read_text(encoding="utf-8"))


def extract_node_ids(nodes: Sequence) -> list[str]:
    node_ids: list[str] = []
    for n in nodes:
        node_id = None
        if isinstance(n, dict):
            node_id = n.get("id")
            if node_id is None:
                node_id = n.get("name")
            if node_id is None:
                node_id = n.get("n")
        else:
            node_id = n
        if node_id is None:
            continue
        node_ids.append(str(node_id))
    return node_ids


def percentile(sorted_vals: Sequence[float], pct: float) -> float:
    if not sorted_vals:
        return 0.0
    if pct <= 0:
        return float(sorted_vals[0])
    if pct >= 1:
        return float(sorted_vals[-1])
    idx = (len(sorted_vals) - 1) * pct
    lo = int(idx)
    hi = min(lo + 1, len(sorted_vals) - 1)
    if lo == hi:
        return float(sorted_vals[lo])
    frac = idx - lo
    return float(sorted_vals[lo] * (1 - frac) + sorted_vals[hi] * frac)


def summarize(values: Sequence[float]) -> dict:
    if not values:
        return {}
    sorted_vals = sorted(values)
    return {
        "count": len(sorted_vals),
        "min": float(sorted_vals[0]),
        "max": float(sorted_vals[-1]),
        "mean": float(statistics.fmean(sorted_vals)),
        "median": float(statistics.median(sorted_vals)),
        "p10": percentile(sorted_vals, 0.10),
        "p90": percentile(sorted_vals, 0.90),
    }


def collect_graph_paths(root: Path, pattern: str, recursive: bool) -> list[Path]:
    if root.is_file():
        return [root]
    glob_pattern = f"**/{pattern}" if recursive else pattern
    return sorted(root.glob(glob_pattern))


def load_dataset_stats(name: str, path: Path, pattern: str, recursive: bool, limit: int) -> DatasetStats:
    stats = DatasetStats(name=name, path=path)
    graph_paths = collect_graph_paths(path, pattern, recursive)
    if limit > 0:
        graph_paths = graph_paths[:limit]

    for graph_path in graph_paths:
        data = read_json(graph_path)
        nodes = data.get("nodes", [])
        edges = data.get("edges", [])
        node_ids = extract_node_ids(nodes)
        stats.graph_count += 1
        stats.nodes_per_graph.append(len(node_ids))
        stats.edges_per_graph.append(len(edges) if isinstance(edges, list) else 0)
        stats.unique_nodes.update(node_ids)
        if node_ids:
            stats.edges_per_node.append((len(edges) if isinstance(edges, list) else 0) / len(node_ids))
        else:
            stats.edges_per_node.append(0.0)

    return stats


def open_text(path: Path):
    if path.suffix == ".gz":
        return gzip.open(path, "rt", encoding="utf-8", errors="replace")
    return path.open("r", encoding="utf-8", errors="replace")


def iter_conceptnet_nodes(path: Path, relations: set[str]) -> Iterator[str]:
    with open_text(path) as handle:
        reader = csv.reader(handle, delimiter="\t")
        for row in reader:
            if len(row) < 4:
                continue
            relation = row[1]
            if relation not in relations:
                continue
            start_node = row[2]
            end_node = row[3]
            if not (start_node.startswith("/c/en/") and end_node.startswith("/c/en/")):
                continue
            u = start_node.split("/")[3]
            v = end_node.split("/")[3]
            if len(u) <= 2 or len(v) <= 2:
                continue
            yield u
            yield v


def load_conceptnet_node_set(path: Path, relations: set[str]) -> set[str]:
    if path.is_dir():
        candidate = path / "assertions.csv"
        if not candidate.exists():
            candidate = path / "assertions.csv.gz"
        if not candidate.exists():
            raise FileNotFoundError(f"ConceptNet assertions not found under {path}")
        path = candidate
    nodes: set[str] = set()
    for node in iter_conceptnet_nodes(path, relations):
        nodes.add(node)
    return nodes


def load_cx_node_set(path: Path) -> set[str]:
    data = json.loads(path.read_text(encoding="utf-8"))
    nodes = []
    for element in data:
        if "nodes" in element:
            nodes = element["nodes"]
            break
    node_set: set[str] = set()
    for node in nodes:
        name = node.get("n")
        if name:
            node_set.add(str(name))
    return node_set


def load_json_node_set(path: Path) -> set[str]:
    data = read_json(path)
    nodes = data.get("nodes", [])
    return set(extract_node_ids(nodes))


def normalize_concept(concept: str) -> str:
    return concept.strip().replace(" ", "_")


def load_causenet_node_set(path: Path) -> set[str]:
    nodes: set[str] = set()
    with path.open("r", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            relation = record.get("causal_relation", {})
            cause = relation.get("cause", {}).get("concept")
            effect = relation.get("effect", {}).get("concept")
            if cause:
                nodes.add(normalize_concept(str(cause)))
            if effect:
                nodes.add(normalize_concept(str(effect)))
    return nodes


def load_source_nodes(path: Path, relations: set[str]) -> set[str]:
    if path.is_dir():
        return load_conceptnet_node_set(path, relations)
    if path.suffix == ".cx":
        return load_cx_node_set(path)
    if path.suffix in {".csv", ".gz"}:
        return load_conceptnet_node_set(path, relations)
    if path.suffix == ".jsonl":
        return load_causenet_node_set(path)
    if path.suffix == ".json":
        return load_json_node_set(path)
    raise ValueError(f"Unsupported source format for {path}")


def parse_kv(arg: str) -> tuple[str, Path]:
    if "=" not in arg:
        raise ValueError(f"Expected name=path, got {arg}")
    name, raw_path = arg.split("=", 1)
    return name.strip(), Path(raw_path)


def default_datasets() -> list[tuple[str, Path]]:
    defaults = [
        ("conceptnet", Path("dataset/conceptnet")),
        ("causenet", Path("dataset/causenet")),
        ("influenza", Path("dataset/influenza")),
        ("bnlearn", Path("dataset/bnlearn")),
    ]
    return [(name, path) for name, path in defaults if path.exists()]


DEFAULT_SOURCE_BY_NAME = {
    "conceptnet": Path("conceptnet-assertions-5.7.0.csv"),
    "causenet": Path("causenet-full.jsonl"),
    "influenza": Path("A network leading to influenza onset obtained through our network structure estimation..cx"),
}

DEFAULT_SOURCE_BY_PATH = {
    Path("dataset/conceptnet"): Path("conceptnet-assertions-5.7.0.csv"),
    Path("dataset/causenet"): Path("causenet-full.jsonl"),
    Path("dataset/influenza"): Path("A network leading to influenza onset obtained through our network structure estimation..cx"),
}


def default_sources(datasets: Sequence[tuple[str, Path]]) -> dict[str, Path]:
    sources: dict[str, Path] = {}
    for name, path in datasets:
        candidate = DEFAULT_SOURCE_BY_NAME.get(name)
        if candidate is None:
            for default_path, default_source in DEFAULT_SOURCE_BY_PATH.items():
                if path.resolve() == default_path.resolve():
                    candidate = default_source
                    break
        if candidate is None:
            continue
        if candidate.exists():
            sources[name] = candidate
        else:
            print(f"Warning: default source not found for {name}: {candidate}")
    return sources


def summarize_dataset(stats: DatasetStats, source_nodes: set[str] | None = None) -> dict:
    nodes_summary = summarize(stats.nodes_per_graph)
    edges_summary = summarize(stats.edges_per_graph)
    epn_summary = summarize(stats.edges_per_node)
    unique_nodes = len(stats.unique_nodes)
    total_nodes = stats.total_nodes()
    reuse_ratio = (total_nodes / unique_nodes) if unique_nodes else 0.0

    summary = {
        "name": stats.name,
        "path": str(stats.path),
        "graphs": stats.graph_count,
        "unique_nodes": unique_nodes,
        "total_nodes": total_nodes,
        "node_reuse_ratio": reuse_ratio,
        "nodes_per_graph": nodes_summary,
        "edges_per_graph": edges_summary,
        "edges_per_node": epn_summary,
    }
    if source_nodes is not None:
        source_total = len(source_nodes)
        covered = len(stats.unique_nodes & source_nodes)
        coverage = (covered / source_total) if source_total else 0.0
        summary.update({
            "source_nodes": source_total,
            "source_nodes_covered": covered,
            "coverage_ratio": coverage,
        })
    return summary


def format_summary(summary: dict) -> str:
    lines = [
        f"DATASET: {summary['name']} ({summary['path']})",
        f"  graphs: {summary['graphs']}",
        f"  unique nodes: {summary['unique_nodes']}",
        f"  total nodes (sum): {summary['total_nodes']}",
        f"  node reuse ratio (sum/unique): {summary['node_reuse_ratio']:.2f}",
    ]
    if "source_nodes" in summary:
        lines.append(f"  source nodes: {summary['source_nodes']}")
        lines.append(f"  coverage: {summary['source_nodes_covered']} ({summary['coverage_ratio']:.2%})")

    def render_stats(label: str, stats: dict) -> None:
        if not stats:
            lines.append(f"  {label}: n/a")
            return
        lines.append(
            f"  {label}: mean {stats['mean']:.2f}, median {stats['median']:.2f}, "
            f"min {stats['min']:.0f}, max {stats['max']:.0f}, "
            f"p10 {stats['p10']:.2f}, p90 {stats['p90']:.2f}"
        )

    render_stats("nodes per graph", summary.get("nodes_per_graph", {}))
    render_stats("edges per graph", summary.get("edges_per_graph", {}))
    render_stats("edges per node", summary.get("edges_per_node", {}))
    return "\n".join(lines)


def overlap_report(stats_by_name: dict[str, DatasetStats]) -> list[str]:
    names = sorted(stats_by_name.keys())
    lines = []
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            left = stats_by_name[names[i]].unique_nodes
            right = stats_by_name[names[j]].unique_nodes
            if not left and not right:
                jaccard = 0.0
                intersection = 0
                union = 0
            else:
                intersection = len(left & right)
                union = len(left | right)
                jaccard = (intersection / union) if union else 0.0
            lines.append(
                f"{names[i]} vs {names[j]}: intersection {intersection}, union {union}, jaccard {jaccard:.4f}"
            )
    return lines


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Report dataset-level stats and overlap for graph JSON datasets.")
    parser.add_argument("--dataset", action="append", help="Dataset mapping name=path (repeatable)")
    parser.add_argument("--source", action="append", help="Source graph mapping name=path (repeatable, overrides defaults)")
    parser.add_argument("--pattern", type=str, default="*.json", help="Glob pattern under dataset path")
    parser.add_argument("--recursive", action="store_true", help="Enable recursive globbing (**/pattern)")
    parser.add_argument("--limit", type=int, default=0, help="Optional limit on number of graphs per dataset (0 = all)")
    parser.add_argument("--json", action="store_true", help="Emit machine-readable JSON summary")
    parser.add_argument("--conceptnet-relations", nargs="*", default=sorted(TARGET_RELATIONS))
    return parser.parse_args(argv)


def main(argv: Iterable[str] | None = None) -> None:
    args = parse_args(argv)

    dataset_args = args.dataset or []
    datasets: list[tuple[str, Path]] = []
    if dataset_args:
        for item in dataset_args:
            name, path = parse_kv(item)
            if not path.exists():
                raise SystemExit(f"Dataset path not found: {path}")
            datasets.append((name, path))
    else:
        datasets = default_datasets()
        if not datasets:
            raise SystemExit("No datasets provided and no default dataset paths found.")

    sources = default_sources(datasets)
    for item in args.source or []:
        name, path = parse_kv(item)
        if not path.exists():
            raise SystemExit(f"Source path not found: {path}")
        sources[name] = path

    stats_by_name: dict[str, DatasetStats] = {}
    summary_by_name: dict[str, dict] = {}
    for name, path in datasets:
        stats = load_dataset_stats(name, path, args.pattern, args.recursive, args.limit)
        stats_by_name[name] = stats
        source_nodes = None
        if name in sources:
            source_nodes = load_source_nodes(sources[name], set(args.conceptnet_relations))
        summary_by_name[name] = summarize_dataset(stats, source_nodes=source_nodes)

    if args.json:
        output = {
            "datasets": summary_by_name,
            "overlap": overlap_report(stats_by_name),
        }
        print(json.dumps(output, indent=2))
        return

    for name in summary_by_name:
        print(format_summary(summary_by_name[name]))
        print("")

    overlap_lines = overlap_report(stats_by_name)
    if overlap_lines:
        print("OVERLAP (unique node sets)")
        for line in overlap_lines:
            print(f"  {line}")


if __name__ == "__main__":
    main()
