from __future__ import annotations

import json
import gzip
from pathlib import Path
from typing import List, Optional

import matplotlib.pyplot as plt
import networkx as nx
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Causal Graph Explorer", layout="wide")


@st.cache_data(show_spinner=False)
def load_manifest(manifest_path: Path) -> pd.DataFrame:
    rows = []
    open_fn = gzip.open if manifest_path.suffix == ".gz" else open  # type: ignore[name-defined]
    with open_fn(manifest_path, "rt", encoding="utf-8") as f:  # type: ignore[attr-defined]
        for line in f:
            if not line.strip():
                continue
            data = json.loads(line)
            nodes = data.get("nodes", [])
            edges = data.get("edges", [])
            rows.append({
                "graph_id": data.get("graph_id"),
                "domain": data.get("domain"),
                "source": data.get("source"),
                "path": data.get("path"),
                "nodes_count": len(nodes),
                "edges_count": len(edges),
            })
    df = pd.DataFrame(rows)
    if df.empty:
        return df

    df["graph_id"] = df["graph_id"].astype(str)
    df["domain"] = df["domain"].fillna("").astype(str)
    df["source"] = df["source"].fillna("").astype(str)
    df["path"] = df["path"].astype(str)
    df["nodes_count"] = pd.to_numeric(df["nodes_count"], errors="coerce").fillna(0).astype(int)
    df["edges_count"] = pd.to_numeric(df["edges_count"], errors="coerce").fillna(0).astype(int)

    return df


def load_graph(path: Path) -> nx.DiGraph:
    if path.suffix == ".gz":
        import gzip
        with gzip.open(path, "rt", encoding="utf-8") as f:
            data = json.load(f)
    else:
        data = json.loads(path.read_text())
    G = nx.DiGraph()
    node_ids = []
    for n in data.get("nodes", []):
        if isinstance(n, dict) and "id" in n:
            node_ids.append(n["id"])
        else:
            node_ids.append(n)
    G.add_nodes_from(node_ids)
    for e in data.get("edges", []):
        if isinstance(e, dict) and "source" in e and "target" in e:
            G.add_edge(e["source"], e["target"])
        elif isinstance(e, (list, tuple)) and len(e) == 2:
            G.add_edge(e[0], e[1])
    return G


def draw_graph(G: nx.DiGraph, title: str):
    plt.figure(figsize=(6, 4))
    pos = nx.spring_layout(G, seed=42, k=0.5)
    nx.draw_networkx_nodes(G, pos, node_color="#a6cee3", edgecolors="#1f78b4")
    nx.draw_networkx_edges(G, pos, edge_color="#555555", arrowsize=10)
    nx.draw_networkx_labels(G, pos, font_size=8)
    plt.title(title)
    plt.axis("off")
    st.pyplot(plt.gcf())
    plt.close()


def main():
    st.title("Causal Graph Explorer")

    manifest_path = st.sidebar.text_input("Manifest path", "dataset/full_dataset.jsonl")
    manifest_file = Path(manifest_path)
    if not manifest_file.exists():
        st.warning("Manifest file not found. Provide a valid path.")
        return

    df = load_manifest(manifest_file)
    if df.empty:
        st.warning("Manifest is empty.")
        return

    st.sidebar.subheader("Filters")
    domains = sorted(df["domain"].dropna().unique().tolist())
    selected_domains = st.sidebar.multiselect("Domain", domains, default=domains)
    sources = sorted(df["source"].dropna().unique().tolist())
    selected_sources = st.sidebar.multiselect("Source", sources, default=sources)

    min_nodes, max_nodes = st.sidebar.slider(
        "Node count range",
        int(df["nodes_count"].min()),
        int(df["nodes_count"].max()),
        (int(df["nodes_count"].min()), int(df["nodes_count"].max())),
    )
    min_edges, max_edges = st.sidebar.slider(
        "Edge count range",
        int(df["edges_count"].min()),
        int(df["edges_count"].max()),
        (int(df["edges_count"].min()), int(df["edges_count"].max())),
    )

    filtered = df[
        (df["domain"].isin(selected_domains))
        & (df["source"].isin(selected_sources))
        & (df["nodes_count"].between(min_nodes, max_nodes))
        & (df["edges_count"].between(min_edges, max_edges))
    ]

    st.write(f"Filtered graphs: {len(filtered)} / {len(df)}")

    # Random graph selection
    default_graph_id = filtered["graph_id"].iloc[0] if not filtered.empty else None
    random_pick = st.button("🎲 Random graph from filtered") if not filtered.empty else False
    if random_pick and not filtered.empty:
        default_graph_id = filtered.sample(1, random_state=None)["graph_id"].iloc[0]

    col1, col2 = st.columns([1, 1])
    with col1:
        st.subheader("Aggregate stats")
        st.metric("Graphs", len(filtered))
        st.metric("Avg nodes", f"{filtered['nodes_count'].mean():.2f}")
        st.metric("Avg edges", f"{filtered['edges_count'].mean():.2f}")
        st.metric("Edges per node", f"{(filtered['edges_count'] / filtered['nodes_count']).mean():.2f}")

    with col2:
        st.subheader("Pick a graph")
        graph_id = st.selectbox("Graph ID", filtered["graph_id"], index=0 if default_graph_id is None else filtered["graph_id"].tolist().index(default_graph_id))
        row = filtered[filtered["graph_id"] == graph_id].iloc[0]
        st.dataframe(row.to_frame().T, width="stretch")
        path = Path(row["path"])
        if not path.exists():
            st.error(f"File not found: {path}")
        else:
            G = load_graph(path)
            draw_graph(G, f"{graph_id} ({len(G.nodes())} nodes, {len(G.edges())} edges)")

            # Edge list
            edge_rows = [{"source": u, "target": v} for u, v in G.edges()]
            if edge_rows:
                st.subheader("Edges")
                st.dataframe(pd.DataFrame(edge_rows), height=200, width="stretch")


if __name__ == "__main__":
    main()
