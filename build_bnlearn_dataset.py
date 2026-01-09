from __future__ import annotations

import argparse
import gzip
import json
import shutil
import tempfile
from pathlib import Path
from typing import Iterable, List, Tuple

import networkx as nx
import requests
from pgmpy.readwrite import BIFReader
from pgmpy.utils import get_example_model

from assets.augmenter import GraphAugmenter

DEFAULT_OUTPUT_DIR = Path("dataset/bnlearn")
DEFAULT_NETWORKS = ("asia",)

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

BNLEARN_BASE = "https://www.bnlearn.com/bnrepository"

BNLEARN_NETWORKS = {
    # Small
    "asia": f"{BNLEARN_BASE}/asia/asia.bif.gz",
    "cancer": f"{BNLEARN_BASE}/cancer/cancer.bif.gz",
    "earthquake": f"{BNLEARN_BASE}/earthquake/earthquake.bif.gz",
    "sachs": f"{BNLEARN_BASE}/sachs/sachs.bif.gz",
    "survey": f"{BNLEARN_BASE}/survey/survey.bif.gz",
    # Medium
    "alarm": f"{BNLEARN_BASE}/alarm/alarm.bif.gz",
    "barley": f"{BNLEARN_BASE}/barley/barley.bif.gz",
    "child": f"{BNLEARN_BASE}/child/child.bif.gz",
    "insurance": f"{BNLEARN_BASE}/insurance/insurance.bif.gz",
    "mildew": f"{BNLEARN_BASE}/mildew/mildew.bif.gz",
    "water": f"{BNLEARN_BASE}/water/water.bif.gz",
    # Large
    "hailfinder": f"{BNLEARN_BASE}/hailfinder/hailfinder.bif.gz",
    "hepar2": f"{BNLEARN_BASE}/hepar2/hepar2.bif.gz",
    "win95pts": f"{BNLEARN_BASE}/win95pts/win95pts.bif.gz",
    # Very Large
    "andes": f"{BNLEARN_BASE}/andes/andes.bif.gz",
    "diabetes": f"{BNLEARN_BASE}/diabetes/diabetes.bif.gz",
    "link": f"{BNLEARN_BASE}/link/link.bif.gz",
    "pathfinder": f"{BNLEARN_BASE}/pathfinder/pathfinder.bif.gz",
    "pigs": f"{BNLEARN_BASE}/pigs/pigs.bif.gz",
    # Massive
    "munin": f"{BNLEARN_BASE}/munin/munin.bif.gz",
    # Only munin4 is published with this path; munin2/munin3 appear unavailable.
    "munin4": f"{BNLEARN_BASE}/munin4/munin4.bif.gz",
}

# Interpretability-based presets
BNLEARN_PRESETS = {
    "recommended": [
        "asia",
        "earthquake",
        "survey",
        "insurance",
        "child",
    ],
    "expanded": [
        # Recommended plus clear larger graphs and maybe-alarm
        "asia",
        "earthquake",
        "survey",
        "insurance",
        "child",
        "hailfinder",
        "win95pts",
        # "alarm",
    ],
    "all": sorted(BNLEARN_NETWORKS.keys()),
}

MANUAL_DESCRIPTIONS = {
    # Survey network (abbreviated nodes)
    "A": "Age of the individual",
    "S": "Sex or gender",
    "E": "Education level",
    "O": "Occupation category",
    "R": "Residence size (urban vs rural)",
    "T": "Primary transport or travel frequency",
    # Insurance network (abbreviated nodes)
    "GoodStudent": "Whether the driver qualifies as a good student",
    "Age": "Age of the policy holder",
    "SocioEcon": "Socio-economic status of the policy holder",
    "RiskAversion": "Driver's aversion to risk while driving",
    "VehicleYear": "Manufacture year of the insured vehicle",
    "ThisCarDam": "Past damage history of this car",
    "RuggedAuto": "How rugged or durable the vehicle is",
    "Accident": "Whether the driver has been in an accident",
    "MakeModel": "Make and model of the vehicle",
    "DrivQuality": "Overall driving quality of the driver",
    "Mileage": "Typical mileage driven",
    "Antilock": "Whether the vehicle has anti-lock brakes",
    "DrivingSkill": "Driver's skill level",
    "SeniorTrain": "Whether driver took senior training",
    "ThisCarCost": "Replacement or repair cost for this car",
    "Theft": "Likelihood of theft for this vehicle",
    "CarValue": "Current value of the car",
    "HomeBase": "Typical parking location of the car",
    "AntiTheft": "Presence of anti-theft devices",
    "PropCost": "Property damage cost in accidents",
    "OtherCarCost": "Cost of damage to other cars",
    "OtherCar": "Involvement of another car",
    "MedCost": "Medical cost resulting from accidents",
    "Cushioning": "Quality of car cushioning for safety",
    "Airbag": "Presence and quality of airbags",
    "ILiCost": "Injury liability cost",
    "DrivHist": "Past driving history of the driver",
}


def describe_node(node: str) -> str:
    if node in MANUAL_DESCRIPTIONS:
        return MANUAL_DESCRIPTIONS[node]
    if CONCEPT_LIB:
        desc = CONCEPT_LIB.get(node)
        if desc:
            return desc
    return f"The variable {node}"


def graph_to_payload(name: str, graph: nx.DiGraph, suffix: str | None = None) -> dict:
    graph_id = f"bnlearn_{name}"
    if suffix:
        graph_id = f"{graph_id}_{suffix}"

    nodes_data = []
    for node in graph.nodes():
        nodes_data.append({
            "id": node,
            "name": node,
            "description": describe_node(node),
        })

    edges_data = [{"source": u, "target": v} for u, v in graph.edges()]

    return {
        "graph_id": graph_id,
        "domain": "Benchmark",
        "source": name,
        "nodes": nodes_data,
        "edges": edges_data,
    }


def load_model(name: str, bif_path: Path | None) -> tuple[str, object]:
    if bif_path is not None:
        reader = BIFReader(str(bif_path))
        model = reader.get_model()
        resolved_name = name or bif_path.stem
        return resolved_name, model

    model = get_example_model(name)
    return name, model


def model_to_digraph(model) -> nx.DiGraph:
    G = nx.DiGraph()
    G.add_nodes_from(model.nodes())
    G.add_edges_from(model.edges())
    return G


def download_bif(name: str, url: str, workdir: Path) -> Path:
    workdir.mkdir(parents=True, exist_ok=True)
    tmp_path = workdir / f"{name}.bif"

    with requests.get(url, stream=True, timeout=60) as r:
        try:
            r.raise_for_status()
        except requests.HTTPError as exc:  # type: ignore[attr-defined]
            print(f"Warning: skipping {name} ({url}) due to HTTP error: {exc}")
            return None
        if url.endswith(".gz"):
            with tempfile.NamedTemporaryFile(delete=False) as gz_tmp:
                shutil.copyfileobj(r.raw, gz_tmp)
                gz_tmp_path = Path(gz_tmp.name)
            with gzip.open(gz_tmp_path, "rb") as gz_f, tmp_path.open("wb") as out_f:
                shutil.copyfileobj(gz_f, out_f)
            gz_tmp_path.unlink(missing_ok=True)
        else:
            with tmp_path.open("wb") as out_f:
                shutil.copyfileobj(r.raw, out_f)

    return tmp_path


def write_graph(output_dir: Path, payload: dict) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / f"{payload['graph_id']}.json"
    path.write_text(json.dumps(payload, indent=2))
    return path


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build BNLearn causal graphs from bnlearn / pgmpy examples.")
    parser.add_argument("--networks", nargs="+", default=list(DEFAULT_NETWORKS), help="Names of networks to export (e.g., asia). Ignored if --bnlearn-all is set.")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR, help="Directory to write JSON graphs")
    parser.add_argument("--bif", type=Path, help="Optional BIF file for a custom network")
    parser.add_argument("--bnlearn-all", action="store_true", help="Fetch and export all networks from the bnlearn repository")
    parser.add_argument("--bnlearn-preset", choices=sorted(BNLEARN_PRESETS.keys()), default="expanded", help="Interpretability preset when using --bnlearn-all (recommended, expanded, all)")
    parser.add_argument("--marginalize-copies", type=int, default=5, help="How many marginalized variants to generate per network")
    parser.add_argument("--marginalize-drop-min", type=float, default=0.2, help="Minimum drop rate for marginalization")
    parser.add_argument("--marginalize-drop-max", type=float, default=0.4, help="Maximum drop rate for marginalization")
    parser.add_argument("--snowball-copies", type=int, default=5, help="How many snowball samples to generate per network")
    parser.add_argument("--snowball-min-nodes", type=int, default=8, help="Minimum nodes in snowball samples")
    parser.add_argument("--snowball-max-nodes", type=int, default=15, help="Maximum nodes in snowball samples")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for augmentation")
    return parser.parse_args(argv)


def main(argv: Iterable[str] | None = None) -> None:
    args = parse_args(argv)

    workdir = args.output_dir / "_tmp"
    augmenter = GraphAugmenter(seed=args.seed)

    drop_min = min(args.marginalize_drop_min, args.marginalize_drop_max)
    drop_max = max(args.marginalize_drop_min, args.marginalize_drop_max)

    if args.bnlearn_all:
        preset_names = BNLEARN_PRESETS[args.bnlearn_preset]
        networks: List[Tuple[str, str | None]] = [(name, BNLEARN_NETWORKS.get(name)) for name in preset_names]
    else:
        networks = []
        for name in args.networks:
            url = BNLEARN_NETWORKS.get(name)
            networks.append((name, url))

    for name, url in networks:
        if url:
            bif_path = download_bif(name, url, workdir)
            if bif_path is None:
                continue
        else:
            bif_path = args.bif

        network_name, model = load_model(name, bif_path)
        base_graph = model_to_digraph(model)

        # Original
        orig_payload = graph_to_payload(network_name, base_graph, suffix="orig")
        output_path = write_graph(args.output_dir, orig_payload)
        print(f"Saved {network_name} to {output_path}")

        # Marginalized variants
        for i in range(max(0, args.marginalize_copies)):
            drop = augmenter.rng.uniform(drop_min, drop_max)
            marg_graph = augmenter.marginalize(base_graph, drop_rate=drop)
            payload = graph_to_payload(network_name, marg_graph, suffix=f"marg_{i}")
            write_graph(args.output_dir, payload)

        # Snowball samples
        if base_graph.number_of_nodes() > args.snowball_min_nodes:
            for i in range(max(0, args.snowball_copies)):
                snow_graph = augmenter.snowball_sample(
                    base_graph,
                    min_nodes=args.snowball_min_nodes,
                    max_nodes=args.snowball_max_nodes,
                )
                payload = graph_to_payload(network_name, snow_graph, suffix=f"sub_{i}")
                write_graph(args.output_dir, payload)

    shutil.rmtree(workdir, ignore_errors=True)


if __name__ == "__main__":
    main()
