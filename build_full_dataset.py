from __future__ import annotations

import argparse
import gzip
import json
import shutil
from pathlib import Path
from typing import Iterable, List

from build_conceptnet_dataset import (
    DEFAULT_INPUT as CN_DEFAULT_INPUT,
    DEFAULT_OUTPUT_DIR as CN_DEFAULT_OUTPUT_DIR,
    TARGET_RELATIONS,
    ensure_wordnet,
    extract_subgraphs,
    load_conceptnet_graph,
)
from build_causenet_dataset import (
    DEFAULT_INPUT as CAUSENET_DEFAULT_INPUT,
    DEFAULT_OUTPUT_DIR as CAUSENET_DEFAULT_OUTPUT_DIR,
    extract_subgraphs as extract_causenet_subgraphs,
    load_causenet_graph,
)
from build_bnlearn_dataset import (
    BNLEARN_PRESETS,
    DEFAULT_OUTPUT_DIR as BNLEARN_DEFAULT_OUTPUT_DIR,
    parse_args as bnlearn_parse_args,  # unused but keeps compatibility
    model_to_digraph,
    graph_to_payload,
    load_model,
    BNLEARN_NETWORKS,
    GraphAugmenter,
)


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build full dataset: ConceptNet + CauseNet + BNLearn with augmentations, and emit a JSONL manifest.")

    # ConceptNet
    parser.add_argument("--conceptnet-input", type=Path, default=CN_DEFAULT_INPUT, help="ConceptNet assertions file (.csv or .csv.gz)")
    parser.add_argument("--conceptnet-output", type=Path, default=CN_DEFAULT_OUTPUT_DIR, help="Output directory for ConceptNet graphs")
    parser.add_argument("--conceptnet-num-graphs", type=int, default=5000, help="Number of ConceptNet graphs to sample")
    parser.add_argument("--conceptnet-min-nodes", "--min-nodes", dest="conceptnet_min_nodes", type=int, default=5)
    parser.add_argument("--conceptnet-max-nodes", "--max-nodes", dest="conceptnet_max_nodes", type=int, default=15)
    parser.add_argument("--conceptnet-relations", nargs="*", default=sorted(TARGET_RELATIONS))
    parser.add_argument("--conceptnet-seed", type=int, default=17)

    # CauseNet
    parser.add_argument("--causenet-input", type=Path, default=CAUSENET_DEFAULT_INPUT, help="CauseNet JSONL file")
    parser.add_argument("--causenet-output", type=Path, default=CAUSENET_DEFAULT_OUTPUT_DIR, help="Output directory for CauseNet graphs")
    parser.add_argument("--causenet-num-graphs", type=int, default=5000, help="Number of CauseNet graphs to sample")
    parser.add_argument("--causenet-min-nodes", type=int, default=5)
    parser.add_argument("--causenet-max-nodes", type=int, default=15)
    parser.add_argument("--causenet-seed", type=int, default=17)
    parser.add_argument("--causenet-min-support", type=int, default=1, help="Minimum support value to keep a causal relation")
    parser.add_argument("--causenet-allow-cycles", action="store_true", help="Allow cyclic CauseNet subgraphs (skip DAG filter)")
    parser.add_argument(
        "--causenet-break-cycles",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Break cycles in CauseNet samples instead of discarding",
    )
    parser.add_argument("--causenet-min-path-len", type=int, default=3, help="Minimum longest path length for CauseNet DAGs")

    # BNLearn
    parser.add_argument("--bnlearn-all", action="store_true", help="Export bnlearn networks (uses preset)")
    parser.add_argument("--bnlearn-preset", choices=sorted(BNLEARN_PRESETS.keys()), default="expanded")
    parser.add_argument("--networks", nargs="+", default=["asia"], help="Specific networks to export when --bnlearn-all is not set")
    parser.add_argument("--bnlearn-output", type=Path, default=BNLEARN_DEFAULT_OUTPUT_DIR)
    parser.add_argument("--bif", type=Path)
    parser.add_argument("--marginalize-copies", type=int, default=5)
    parser.add_argument("--marginalize-drop-min", type=float, default=0.2)
    parser.add_argument("--marginalize-drop-max", type=float, default=0.4)
    parser.add_argument("--snowball-copies", type=int, default=5)
    parser.add_argument("--snowball-min-nodes", type=int, default=8)
    parser.add_argument("--snowball-max-nodes", type=int, default=15)
    parser.add_argument("--seed", type=int, default=42)

    # Combined output
    parser.add_argument("--manifest", type=Path, default=Path("dataset/full_dataset.jsonl"), help="Path to JSONL manifest of all graphs")
    parser.add_argument("--manifest-gzip", action="store_true", help="Also emit gzip-compressed JSONL (.gz)")

    return parser.parse_args(argv)


def build_conceptnet(args: argparse.Namespace) -> List[Path]:
    ensure_wordnet()
    g = load_conceptnet_graph(args.conceptnet_input, set(args.conceptnet_relations))
    extract_subgraphs(
        graph=g,
        output_dir=args.conceptnet_output,
        num_graphs=args.conceptnet_num_graphs,
        min_nodes=args.conceptnet_min_nodes,
        max_nodes=args.conceptnet_max_nodes,
        seed=args.conceptnet_seed,
    )
    return sorted(args.conceptnet_output.glob("graph_*.json"))


def build_causenet(args: argparse.Namespace) -> List[Path]:
    ensure_wordnet()
    g = load_causenet_graph(args.causenet_input, min_support=args.causenet_min_support)
    extract_causenet_subgraphs(
        graph=g,
        output_dir=args.causenet_output,
        num_graphs=args.causenet_num_graphs,
        min_nodes=args.causenet_min_nodes,
        max_nodes=args.causenet_max_nodes,
        seed=args.causenet_seed,
        allow_cycles=args.causenet_allow_cycles,
        break_cycles_enabled=args.causenet_break_cycles,
        min_path_len=args.causenet_min_path_len,
    )
    return sorted(args.causenet_output.glob("graph_*.json"))


def build_bnlearn(args: argparse.Namespace) -> List[Path]:
    workdir = args.bnlearn_output / "_tmp"
    workdir.mkdir(parents=True, exist_ok=True)
    augmenter = GraphAugmenter(seed=args.seed)

    drop_min = min(args.marginalize_drop_min, args.marginalize_drop_max)
    drop_max = max(args.marginalize_drop_min, args.marginalize_drop_max)

    if args.bnlearn_all:
        names = BNLEARN_PRESETS[args.bnlearn_preset]
        networks = [(n, BNLEARN_NETWORKS.get(n)) for n in names]
    else:
        networks = [(n, BNLEARN_NETWORKS.get(n)) for n in args.networks]

    written: List[Path] = []

    for name, url in networks:
        if url:
            bif_path = download_bif(name, url, workdir)  # type: ignore[name-defined]
            if bif_path is None:
                continue
        else:
            bif_path = args.bif

        network_name, model = load_model(name, bif_path)
        base_graph = model_to_digraph(model)

        # orig
        payload = graph_to_payload(network_name, base_graph, suffix="orig")
        written.append(write_graph(args.bnlearn_output, payload))

        # marginals
        for i in range(max(0, args.marginalize_copies)):
            drop = augmenter.rng.uniform(drop_min, drop_max)
            marg_graph = augmenter.marginalize(base_graph, drop_rate=drop)
            payload = graph_to_payload(network_name, marg_graph, suffix=f"marg_{i}")
            written.append(write_graph(args.bnlearn_output, payload))

        # snowball
        if base_graph.number_of_nodes() > args.snowball_min_nodes:
            for i in range(max(0, args.snowball_copies)):
                snow_graph = augmenter.snowball_sample(
                    base_graph,
                    min_nodes=args.snowball_min_nodes,
                    max_nodes=args.snowball_max_nodes,
                )
                payload = graph_to_payload(network_name, snow_graph, suffix=f"sub_{i}")
                written.append(write_graph(args.bnlearn_output, payload))

    shutil.rmtree(workdir, ignore_errors=True)  # type: ignore[name-defined]
    return written


# Bring needed functions locally to avoid circular import issues
from build_bnlearn_dataset import download_bif, write_graph  # noqa: E402


def write_manifest(json_paths: List[Path], manifest_path: Path, gzip_out: bool = False) -> None:
    manifest_path.parent.mkdir(parents=True, exist_ok=True)

    def emit(handle):
        for path in json_paths:
            data = json.loads(path.read_text())
            data.setdefault("path", str(path))
            handle.write(json.dumps(data))
            handle.write("\n")

    with manifest_path.open("w", encoding="utf-8") as f:
        emit(f)

    if gzip_out:
        gz_path = manifest_path.with_suffix(manifest_path.suffix + ".gz")
        with gzip.open(gz_path, "wt", encoding="utf-8") as gf:
            emit(gf)



def main(argv: Iterable[str] | None = None) -> None:
    args = parse_args(argv)

    conceptnet_paths = build_conceptnet(args)
    causenet_paths = build_causenet(args)
    bnlearn_paths = build_bnlearn(args)

    all_paths = conceptnet_paths + causenet_paths + bnlearn_paths
    write_manifest(all_paths, args.manifest, gzip_out=args.manifest_gzip)
    print(f"Wrote {len(all_paths)} graphs to manifest {args.manifest}")


if __name__ == "__main__":
    main()
