"""Generate a PNG image of the DVC pipeline DAG from dvc dag --dot output."""
import subprocess
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import networkx as nx


def layered_layout(graph: nx.DiGraph) -> dict:
    """Assign positions based on topological depth."""
    layers = {}
    for node in nx.topological_sort(graph):
        preds = list(graph.predecessors(node))
        if not preds:
            depth = 0
        else:
            depth = max(layers[p][1] for p in preds) + 1
        layers[node] = (0, depth)

    # Group by depth and spread horizontally
    depth_groups = {}
    for node, (_, depth) in layers.items():
        depth_groups.setdefault(depth, []).append(node)

    positions = {}
    for depth, nodes in depth_groups.items():
        nodes = sorted(nodes)
        n = len(nodes)
        for i, node in enumerate(nodes):
            x = i - (n - 1) / 2
            positions[node] = (x, -depth)
    return positions


def main():
    reports_dir = Path(__file__).resolve().parents[1] / "reports"
    reports_dir.mkdir(exist_ok=True)

    # Generate dot file if missing
    dot_file = reports_dir / "dvc_pipeline.dot"
    if not dot_file.exists():
        result = subprocess.run(["dvc", "dag", "--dot"], capture_output=True, text=True, check=True)
        dot_file.write_text(result.stdout, encoding="utf-8")

    try:
        lines = dot_file.read_text(encoding="utf-8").splitlines()
    except UnicodeDecodeError:
        lines = dot_file.read_text(encoding="utf-16").splitlines()
    edges = []
    for line in lines:
        line = line.strip().rstrip(";")
        if "->" in line:
            parts = line.split("->")
            src = parts[0].strip().strip('"')
            dst = parts[1].strip().strip('"')
            edges.append((src, dst))

    graph = nx.DiGraph()
    graph.add_edges_from(edges)

    pos = layered_layout(graph)
    plt.figure(figsize=(10, 6))
    nx.draw(
        graph,
        pos,
        with_labels=True,
        node_color="lightblue",
        node_size=2600,
        font_size=9,
        font_weight="bold",
        arrows=True,
        arrowsize=20,
        edge_color="gray",
    )
    out_path = reports_dir / "dvc_pipeline.png"
    plt.title("DVC Pipeline DAG")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    print(f"Saved DAG image to {out_path}")


if __name__ == "__main__":
    main()
