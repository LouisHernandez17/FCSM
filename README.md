## Overview

Local, no-LLM causal graph dataset builder. Tier 2 mines ConceptNet for structure (no descriptions). Tier 3 exports benchmark DAGs from the bnlearn repository (static URL map).

## Prerequisites

- Python 3.12+
- [`uv`](https://github.com/astral-sh/uv) for dependency management
- ConceptNet assertions file: place at `conceptnet-assertions-5.7.0.csv/assertions.csv` (or a `.csv.gz` equivalent)

Install dependencies with `uv`:

```bash
uv sync
```

## Usage

### Tier 2: ConceptNet (structure only)

Generates sampled DAGs; nodes include `id` and `name` only (no descriptions).

```bash
uv run python build_conceptnet_dataset.py \
	--input conceptnet-assertions-5.7.0.csv/assertions.csv \
	--output-dir dataset/tier2_conceptnet \
	--num-graphs 5000 \
	--min-nodes 5 \
	--max-nodes 15 \
	--seed 17
```

Flags: `--relations` overrides the kept ConceptNet relations; supports `.csv` or `.csv.gz` input. WordNet is fetched automatically only if needed.

### Tier 3: Gold-standard networks (bnlearn)

Uses a static map of bnlearn network URLs (e.g., `https://www.bnlearn.com/bnrepository/asia/asia.bif.gz`) and exports nodes with `id` and `name`.

Augmentations (latent projection + subgraphs): for each network the script writes:
- `gold_<name>_orig.json` (full graph)
- `gold_<name>_marg_<i>.json` (marginalized via latent projection, default 5 copies, 20–40% node drop)
- `gold_<name>_sub_<i>.json` (snowball subgraphs, default 5 copies, 8–15 nodes, only if graph is large enough)

Presets based on interpretability:

- `recommended`: asia, earthquake, survey, insurance, child
- `expanded` (default): recommended + hailfinder, win95pts, alarm
- `all`: every mapped bnlearn network

All (with chosen preset):

```bash
uv run python build_gold_standard.py --bnlearn-all --bnlearn-preset expanded --output-dir dataset/tier3_gold
```

Tuning augmentation (example: 3 marginals, 2 subgraphs, custom seed):

```bash
uv run python build_gold_standard.py \
	--bnlearn-all --bnlearn-preset recommended \
	--marginalize-copies 3 --marginalize-drop-min 0.25 --marginalize-drop-max 0.35 \
	--snowball-copies 2 --snowball-min-nodes 8 --snowball-max-nodes 15 \
	--seed 123 \
	--output-dir dataset/tier3_gold_custom
```

### One-shot: build everything + manifest

Runs Tier2 ConceptNet and Tier3 bnlearn (with augmentations) and writes a JSONL manifest (optionally gzipped):

```bash
uv run python build_full_dataset.py \
	--conceptnet-input conceptnet-assertions-5.7.0.csv/assertions.csv \
	--conceptnet-output dataset/tier2_conceptnet \
	--conceptnet-num-graphs 5000 --conceptnet-min-nodes 5 --conceptnet-max-nodes 15 --conceptnet-seed 17 \
	--bnlearn-all --bnlearn-preset expanded \
	--marginalize-copies 5 --marginalize-drop-min 0.2 --marginalize-drop-max 0.4 \
	--snowball-copies 5 --snowball-min-nodes 8 --snowball-max-nodes 15 \
	--seed 42 \
	--gold-output dataset/tier3_gold \
	--manifest dataset/full_dataset.jsonl --manifest-gzip
```

Output:
- Tier2 JSON files in `dataset/tier2_conceptnet/`
- Tier3 JSON files (orig/marg/sub) in `dataset/tier3_gold/`
- Combined manifest: `dataset/full_dataset.jsonl` (and `.jsonl.gz` when `--manifest-gzip` is set), one graph per line with `graph_id`, `nodes`, `edges`, `domain`, `source`, and `path`.

## Validation

Automated structural/distribution checks:

```bash
uv run python validate_dataset.py --root dataset --pattern "**/*.json" --limit 0 --report dataset/health_report.json
```

The script reports DAG validity, connectivity warnings, density stats, and writes an optional JSON summary.

## Streamlit explorer

Browse and visualize graphs from the manifest:

```bash
uv run streamlit run streamlit_app.py -- --manifest dataset/full_dataset.jsonl
```

Use sidebar filters (domain, source, node/edge ranges) and view individual graphs with layouts.

Specific network(s) by name (uses bnlearn URL if known, otherwise pgmpy example or `--bif`):

```bash
uv run python build_gold_standard.py --networks asia sachs alarm --output-dir dataset/tier3_gold
```

Use `--bif path/to/network.bif` to load a custom BIF file instead of the built-in example for a given name.

## Output

- Tier 2: `dataset/tier2_conceptnet/graph_*.json`
- Tier 3: `dataset/tier3_gold/gold_<network>.json`

Each JSON contains `graph_id`, `domain`, `source`, `nodes`, and `edges` fields suitable for downstream training.
