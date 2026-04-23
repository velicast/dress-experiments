#!/usr/bin/env python3
"""ogbl-vessel link prediction using DRESS edge values as the predictor.

The idea: compute DRESS on the training graph with node weights derived
from (x,y,z) spatial coordinates.  For each candidate edge (u,v), the
DRESS value IS the link prediction score — no model, no training.

Node features: 3D coordinates (x, y, z).  We map to a single scalar via
a linear combination with irrational coefficients to make each coordinate
unique:  w(v) = x + sqrt(2)*y + sqrt(3)*z, then normalize.

Metric: ROC-AUC (official OGB evaluator).

Usage:
    python ogbl_vessel_dress.py
"""

import time
import numpy as np

# OGB torch.load compat
import torch
_orig_torch_load = torch.load
torch.load = lambda *a, **kw: _orig_torch_load(*a, **{**kw, 'weights_only': False})

from ogb.linkproppred import LinkPropPredDataset, Evaluator
from dress import DRESS


def main():
    print("Loading ogbl-vessel...")
    t0 = time.perf_counter()
    dataset = LinkPropPredDataset(name="ogbl-vessel")
    graph = dataset[0]
    split_edge = dataset.get_edge_split()
    dt_load = time.perf_counter() - t0
    print(f"  Loaded in {dt_load:.1f}s")

    num_nodes = graph["num_nodes"]
    edge_index = graph["edge_index"]  # (2, E) — training edges, bidirectional
    node_feat = graph["node_feat"]    # (N, 3) — x, y, z coordinates

    print(f"  Nodes: {num_nodes:,}")
    print(f"  Training edges (directed): {edge_index.shape[1]:,}")
    print(f"  Node feature shape: {node_feat.shape}")

    # ── Node weights: linear combination with irrational coefficients ──
    coords = node_feat.astype(np.float64)
    raw_weights = coords[:, 0] + np.sqrt(2) * coords[:, 1] + np.sqrt(3) * coords[:, 2]
    # Log normalization: compress range while keeping positivity and uniqueness
    raw_weights = np.log(raw_weights - raw_weights.min() + 2.0)

    print(f"  Node weights: min={raw_weights.min():.4f} max={raw_weights.max():.4f} "
          f"mean={raw_weights.mean():.4f}")

    # ── Deduplicate to undirected edges ──
    src = edge_index[0]
    tgt = edge_index[1]
    mask = src < tgt
    u_src = src[mask].tolist()
    u_tgt = tgt[mask].tolist()
    num_undirected = len(u_src)
    print(f"  Undirected training edges: {num_undirected:,}")

    # ── Build DRESS object and fit ──
    # Use CPU DRESS object (needed for get() on virtual edges)
    print(f"\n  Building DRESS graph (n={num_nodes:,}, m={num_undirected:,})...")
    t0 = time.perf_counter()
    dg = DRESS(num_nodes, u_src, u_tgt, vertex_weights=raw_weights.tolist())
    dt_build = time.perf_counter() - t0
    print(f"  Built in {dt_build:.1f}s")

    print(f"  Fitting DRESS...")
    t0 = time.perf_counter()
    fit_result = dg.fit(max_iterations=200, epsilon=1e-6)
    dt_fit = time.perf_counter() - t0
    print(f"  DRESS fit: {fit_result.iterations} iters, {dt_fit:.1f}s, delta={fit_result.delta:.2e}")

    edge_vals = dg.dress_values
    print(f"  DRESS values: min={edge_vals.min():.6f} max={edge_vals.max():.6f} "
          f"unique={len(np.unique(np.round(edge_vals, 8))):,}/{len(edge_vals):,}")

    # ── Score edges using get(u, v) ──
    evaluator = Evaluator(name="ogbl-vessel")

    for split_name in ["valid", "test"]:
        split = split_edge[split_name]
        pos_edge = split["edge"]        # (num_pos, 2)
        neg_edge = split["edge_neg"]    # (num_pos, 2)

        num_pos = pos_edge.shape[0]
        num_neg = neg_edge.shape[0]
        print(f"\n  Scoring {split_name}: {num_pos:,} pos + {num_neg:,} neg edges...")

        t0 = time.perf_counter()
        pos_scores = np.array([
            dg.get(int(pos_edge[i, 0]), int(pos_edge[i, 1]))
            for i in range(num_pos)
        ], dtype=np.float64)
        dt_pos = time.perf_counter() - t0
        print(f"    Pos scores in {dt_pos:.1f}s  "
              f"(min={pos_scores.min():.6f} max={pos_scores.max():.6f} mean={pos_scores.mean():.6f})")

        t0 = time.perf_counter()
        neg_scores = np.array([
            dg.get(int(neg_edge[i, 0]), int(neg_edge[i, 1]))
            for i in range(num_neg)
        ], dtype=np.float64)
        dt_neg = time.perf_counter() - t0
        print(f"    Neg scores in {dt_neg:.1f}s  "
              f"(min={neg_scores.min():.6f} max={neg_scores.max():.6f} mean={neg_scores.mean():.6f})")

        y_pred_pos = torch.tensor(pos_scores, dtype=torch.float32)
        y_pred_neg = torch.tensor(neg_scores, dtype=torch.float32)

        result_dict = evaluator.eval({
            "y_pred_pos": y_pred_pos,
            "y_pred_neg": y_pred_neg,
        })

        print(f"    {split_name} ROC-AUC = {result_dict['rocauc']:.4f}")

    print(f"\n  Total DRESS fit time: {dt_fit:.1f}s")
    print(f"  Memory: ~{(num_nodes * 8 + num_undirected * 8) / 1e6:.0f} MB for DRESS arrays")


if __name__ == "__main__":
    main()
