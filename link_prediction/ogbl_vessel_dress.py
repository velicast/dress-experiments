#!/usr/bin/env python3
"""ogbl-vessel link prediction using inv-log-length DRESS edge values.

The idea: compute DRESS on the undirected training graph with edge weights
derived from inverse-log endpoint-to-endpoint coordinate distance. For each
candidate edge (u,v), the same transformed distance is passed as the virtual
edge weight to get(u, v, edge_weight=...).

Metric: ROC-AUC (official OGB evaluator).

Usage:
    pip install dress-graph
    python ogbl_vessel_dress.py
"""

import time
import numpy as np

# OGB torch.load compat
import torch
_orig_torch_load = torch.load
torch.load = lambda *a, **kw: _orig_torch_load(*a, **{**kw, "weights_only": False})

from ogb.linkproppred import LinkPropPredDataset, Evaluator
from dress import DRESS


def endpoint_distances(coords, sources, targets):
    source_index = np.asarray(sources, dtype=np.int64)
    target_index = np.asarray(targets, dtype=np.int64)
    return np.linalg.norm(coords[source_index] - coords[target_index], axis=1)


def inv_log_length_scale(lengths):
    log_lengths = np.log1p(lengths)
    positive_logs = log_lengths[lengths > 0.0]
    log_scale = float(np.median(positive_logs)) if positive_logs.size else 1.0
    return max(log_scale, 1e-12)


def inv_log_length_weights(lengths, log_scale):
    weights = log_scale / np.maximum(np.log1p(lengths), 1e-12)
    return np.clip(weights, 1e-6, 1e6)


def build_dress_edge_weights(coords, sources, targets):
    lengths = endpoint_distances(coords, sources, targets)
    log_scale = inv_log_length_scale(lengths)
    edge_weights = inv_log_length_weights(lengths, log_scale)
    print(f"  Edge lengths: min={lengths.min():.4f} max={lengths.max():.4f} "
          f"median={np.median(lengths):.4f} mean={lengths.mean():.4f}")
    print(f"  inv-log-length scale: {log_scale:.6f}")
    return edge_weights, log_scale


def virtual_edge_weights(coords, candidate_edges, log_scale):
    lengths = endpoint_distances(coords, candidate_edges[:, 0], candidate_edges[:, 1])
    return inv_log_length_weights(lengths, log_scale)


def score_candidate_edges(dg, coords, candidate_edges, log_scale):
    query_weights = virtual_edge_weights(coords, candidate_edges, log_scale)
    return np.array([
        dg.get(
            int(candidate_edges[i, 0]),
            int(candidate_edges[i, 1]),
            edge_weight=float(query_weights[i]),
        )
        for i in range(candidate_edges.shape[0])
    ], dtype=np.float64)


def print_weight_stats(name, values):
    if values is None:
        print(f"  {name}: unit")
        return
    print(f"  {name}: min={values.min():.4f} max={values.max():.4f} "
          f"median={np.median(values):.4f} mean={values.mean():.4f}")


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

    coords = node_feat.astype(np.float64)
    print("  DRESS weight mode: inv-log-length")

    # ── Deduplicate to undirected edges ──
    src = edge_index[0]
    tgt = edge_index[1]
    mask = src < tgt
    u_src = src[mask].tolist()
    u_tgt = tgt[mask].tolist()
    num_undirected = len(u_src)
    print(f"  Undirected training edges: {num_undirected:,}")

    edge_weights, log_scale = build_dress_edge_weights(coords, u_src, u_tgt)
    print_weight_stats("Node weights", None)
    print_weight_stats("Edge weights", edge_weights)
    print("  Virtual edge weights: candidate get(u, v) uses the same "
          "inv-log-length transform")

    # ── Build DRESS object and fit ──
    # Use CPU DRESS object (needed for get() on virtual edges)
    print(f"\n  Building DRESS graph (n={num_nodes:,}, m={num_undirected:,})...")
    t0 = time.perf_counter()
    dg = DRESS(
        num_nodes,
        u_src,
        u_tgt,
        weights=edge_weights.tolist(),
    )
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

    # ── Score edges using get(u, v), with matching virtual edge weights ──
    evaluator = Evaluator(name="ogbl-vessel")

    for split_name in ["valid", "test"]:
        split = split_edge[split_name]
        pos_edge = split["edge"]        # (num_pos, 2)
        neg_edge = split["edge_neg"]    # (num_pos, 2)

        num_pos = pos_edge.shape[0]
        num_neg = neg_edge.shape[0]
        print(f"\n  Scoring {split_name}: {num_pos:,} pos + {num_neg:,} neg edges...")

        t0 = time.perf_counter()
        pos_scores = score_candidate_edges(
            dg, coords, pos_edge, log_scale
        )
        dt_pos = time.perf_counter() - t0
        print(f"    Pos scores in {dt_pos:.1f}s  "
              f"(min={pos_scores.min():.6f} max={pos_scores.max():.6f} mean={pos_scores.mean():.6f})")

        t0 = time.perf_counter()
        neg_scores = score_candidate_edges(
            dg, coords, neg_edge, log_scale
        )
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
