## Overview

Local, no-LLM causal graph dataset builder. The ConceptNet dataset provides commonsense structure and WordNet-enriched nodes. The CauseNet dataset mines causal relations from text. The BNLearn dataset exports benchmark DAGs from the bnlearn repository (static URL map).

## Prerequisites

- Python 3.12+
- [`uv`](https://github.com/astral-sh/uv) for dependency management
- ConceptNet assertions file: place at `conceptnet-assertions-5.7.0.csv/assertions.csv` (or a `.csv.gz` equivalent)
- CauseNet JSONL file: place at `causenet-full.jsonl`

Install dependencies with `uv`:

```bash
uv sync
```

## Usage

### ConceptNet (structure + descriptions)

Generates sampled DAGs; nodes include `id`, `name`, and a WordNet `description` when available.

```bash
uv run python build_conceptnet_dataset.py \
	--input conceptnet-assertions-5.7.0.csv/assertions.csv \
	--output-dir dataset/conceptnet \
	--num-graphs 5000 \
	--min-nodes 5 \
	--max-nodes 15 \
	--seed 17
```

Flags: `--relations` overrides the kept ConceptNet relations; supports `.csv` or `.csv.gz` input. WordNet is fetched automatically only if needed.

### CauseNet (text-mined causal relations)

Builds a causal edge graph from `causenet-full.jsonl` and samples DAG subgraphs.

```bash
uv run python build_causenet_dataset.py \
	--input causenet-full.jsonl \
	--output-dir dataset/causenet \
	--num-graphs 5000 \
	--min-nodes 5 \
	--max-nodes 15 \
	--seed 17 \
	--min-support 1
```

By default, CauseNet samples break cycles to keep DAGs. Use `--allow-cycles` to skip DAG filtering or `--no-break-cycles` to keep the original rejection behavior.

### BNLearn networks (bnlearn)

Uses a static map of bnlearn network URLs (e.g., `https://www.bnlearn.com/bnrepository/asia/asia.bif.gz`) and exports nodes with `id` and `name`.

Augmentations (latent projection + subgraphs): for each network the script writes:
- `bnlearn_<name>_orig.json` (full graph)
- `bnlearn_<name>_marg_<i>.json` (marginalized via latent projection, default 5 copies, 20–40% node drop)
- `bnlearn_<name>_sub_<i>.json` (snowball subgraphs, default 5 copies, 8–15 nodes, only if graph is large enough)

Presets based on interpretability:

- `recommended`: asia, earthquake, survey, insurance, child
- `expanded` (default): recommended + hailfinder, win95pts, alarm
- `all`: every mapped bnlearn network

All (with chosen preset):

```bash
uv run python build_bnlearn_dataset.py --bnlearn-all --bnlearn-preset expanded --output-dir dataset/bnlearn
```

Tuning augmentation (example: 3 marginals, 2 subgraphs, custom seed):

```bash
uv run python build_bnlearn_dataset.py \
	--bnlearn-all --bnlearn-preset recommended \
	--marginalize-copies 3 --marginalize-drop-min 0.25 --marginalize-drop-max 0.35 \
	--snowball-copies 2 --snowball-min-nodes 8 --snowball-max-nodes 15 \
	--seed 123 \
	--output-dir dataset/bnlearn_custom
```

### One-shot: build everything + manifest

Runs ConceptNet, CauseNet, and BNLearn (with augmentations) and writes a JSONL manifest (optionally gzipped):

```bash
uv run python build_full_dataset.py \
	--conceptnet-input conceptnet-assertions-5.7.0.csv/assertions.csv \
	--conceptnet-output dataset/conceptnet \
	--conceptnet-num-graphs 5000 --conceptnet-min-nodes 5 --conceptnet-max-nodes 15 --conceptnet-seed 17 \
	--causenet-input causenet-full.jsonl \
	--causenet-output dataset/causenet \
	--causenet-num-graphs 5000 --causenet-min-nodes 5 --causenet-max-nodes 15 --causenet-seed 17 --causenet-min-support 1 \
	--bnlearn-all --bnlearn-preset expanded \
	--marginalize-copies 5 --marginalize-drop-min 0.2 --marginalize-drop-max 0.4 \
	--snowball-copies 5 --snowball-min-nodes 8 --snowball-max-nodes 15 \
	--seed 42 \
	--bnlearn-output dataset/bnlearn \
	--manifest dataset/full_dataset.jsonl --manifest-gzip
```

Output:
- ConceptNet JSON files in `dataset/conceptnet/`
- CauseNet JSON files in `dataset/causenet/`
- BNLearn JSON files (orig/marg/sub) in `dataset/bnlearn/`
- Combined manifest: `dataset/full_dataset.jsonl` (and `.jsonl.gz` when `--manifest-gzip` is set), one graph per line with `graph_id`, `nodes`, `edges`, `domain`, `source`, and `path`.

## Training curriculum

Held-out splits are by network family (not random files) to prevent leakage. Create them once (also done automatically by `run_curriculum.py`):

```bash
uv run python scripts/partition_dataset.py
```

Two-phase training (recall then precision), leaving the Set Transformer unfrozen because it starts from random weights:

```bash
uv run python run_curriculum.py
```

If you call `train.py` directly, avoid `--freeze-encoder` unless you intentionally want a frozen, randomly initialized set encoder.

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
uv run python build_bnlearn_dataset.py --networks asia sachs alarm --output-dir dataset/bnlearn
```

Use `--bif path/to/network.bif` to load a custom BIF file instead of the built-in example for a given name.

## Output

- ConceptNet: `dataset/conceptnet/graph_*.json`
- CauseNet: `dataset/causenet/graph_*.json`
- BNLearn: `dataset/bnlearn/bnlearn_<network>.json`

Each JSON contains `graph_id`, `domain`, `source`, `nodes`, and `edges` fields suitable for downstream training.
