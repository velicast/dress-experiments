#!/usr/bin/env python3
"""
Δ¹-DRESS vs ALL Hard Graph Families  (CUDA)
=============================================

Runs Δ^k-DRESS (default k=1) on every constructible hard graph family
and every SRG file on disk **using the CUDA backend**.  For each graph
it computes BOTH the histogram fingerprint and the canonicalized
multiset fingerprint, then cross-checks that they agree on separation.

Usage:
    python destroy_hard_families.py                    # Δ¹ on everything
    python destroy_hard_families.py --k 0              # plain DRESS baseline
    python destroy_hard_families.py --k 2              # Δ² fallback
    python destroy_hard_families.py --skip-files       # constructed only
    python destroy_hard_families.py --skip-constructed # SRG files only
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")

import numpy as np

from dress import delta_dress_fit

from hard_families import build_all_families

# ── Constants ─────────────────────────────────────────────────────

SRG_DIR = Path(__file__).resolve().parent.parent / "datasets" / "srgs"
EPSILON = 1e-6
MAX_ITER = 100


# ═══════════════════════════════════════════════════════════════════
#  Fingerprinting (histogram + multiset double-check)
# ═══════════════════════════════════════════════════════════════════

def histogram_hash(histogram: list[int]) -> str:
    """SHA-256 of the histogram — canonical invariant."""
    arr = np.array(histogram, dtype=np.int64)
    return hashlib.sha256(arr.tobytes()).hexdigest()


def multiset_hash(multisets: np.ndarray) -> str:
    """Canonicalize the C(N,k)×E multiset matrix and SHA-256 it.

    Uses the sorted multiplicity signature of rows — an integer-valued
    invariant immune to floating-point noise.  Within a single graph,
    rows from automorphically equivalent vertex deletions are computed
    via identical arithmetic paths and are bitwise equal, so the row
    multiplicity counts are exact.
    """
    ms = np.where(np.isnan(multisets), -1.0, multisets)
    ms = np.sort(ms, axis=1)                 # sort within each subgraph
    # Compute multiplicity signature: how many times each distinct row appears
    row_tuples = [tuple(r) for r in ms]
    counts = Counter(row_tuples)
    # Sorted multiplicity vector (e.g. [1,1,1,...,2] for 38 singletons + 1 pair)
    mult_sig = tuple(sorted(counts.values()))
    # Also hash the rounded matrix for full discriminating power
    ms_sorted = ms[np.lexsort(ms.T[::-1])]
    ms_rounded = np.round(ms_sorted, 10)
    h = hashlib.sha256(ms_rounded.tobytes())
    # Append the exact integer multiplicity signature
    h.update(str(mult_sig).encode())
    return h.hexdigest()


def multiplicity_signature(multisets: np.ndarray) -> tuple[int, ...]:
    """Compute the sorted orbit-size signature from multiset rows.

    Within a single graph, automorphically equivalent vertex deletions
    produce bitwise-identical DRESS rows.  Counting duplicates gives
    the orbit partition — a robust integer invariant immune to float noise.
    """
    ms = np.where(np.isnan(multisets), -1.0, multisets)
    ms = np.sort(ms, axis=1)
    row_tuples = [tuple(r) for r in ms]
    counts = Counter(row_tuples)
    return tuple(sorted(counts.values()))


def compute_fingerprint(n, sources, targets, k=1):
    """Run CUDA delta_dress_fit with keep_multisets=True.

    Returns (hist_hash, multiset_hash, hist_size, num_subgraphs,
             raw_histogram, raw_multisets, mult_sig).
    """
    if len(sources) == 0:
        empty = hashlib.sha256(b"empty").hexdigest()
        return empty, empty, 0, 0, np.array([], dtype=np.int64), None, ()
    r = delta_dress_fit(n, sources, targets, k=k,
                        epsilon=EPSILON, max_iterations=MAX_ITER,
                        keep_multisets=True)
    hh = histogram_hash(r.histogram)
    mh = multiset_hash(r.multisets) if r.multisets is not None else hh
    ms = np.array(r.multisets) if r.multisets is not None else None
    msig = multiplicity_signature(ms) if ms is not None else ()
    raw_hist = np.array(r.histogram, dtype=np.int64)
    return hh, mh, r.hist_size, r.num_subgraphs, raw_hist, r.multisets, msig


def _g6_to_edges(g6_string: str):
    """Convert graph6 string → (n, sources, targets) with u < v."""
    import networkx as nx
    G = nx.from_graph6_bytes(g6_string.encode("ascii"))
    nodes = list(G.nodes())
    idx_map = {v: i for i, v in enumerate(nodes)}
    s, t = [], []
    for u, v in G.edges():
        a, b = idx_map[u], idx_map[v]
        s.append(min(a, b))
        t.append(max(a, b))
    return len(nodes), s, t


def _spence_filename_to_edges(filename: str, n: int = 36):
    """Decode Spence binary-filename format → (n, sources, targets).

    The filename is a string of '0's and '1's of length C(n,2),
    encoding the upper triangle of the adjacency matrix row by row.
    """
    expected = n * (n - 1) // 2
    assert len(filename) == expected, (
        f"Expected {expected} bits, got {len(filename)}")
    sources, targets = [], []
    k = 0
    for i in range(n):
        for j in range(i + 1, n):
            if filename[k] == '1':
                sources.append(i)
                targets.append(j)
            k += 1
    return n, sources, targets


def load_spence_dir(dirpath: Path, n: int = 36,
                    max_graphs: int = 0) -> list[tuple[str, int, list, list]]:
    """Load Spence-format graphs from a directory of zero-byte files.

    Each filename is C(n,2) bits encoding the upper triangle.
    Returns list of (filename, n, sources, targets).
    """
    expected_len = n * (n - 1) // 2
    entries = []
    for f in sorted(dirpath.iterdir()):
        if max_graphs > 0 and len(entries) >= max_graphs:
            break
        name = f.name
        if len(name) == expected_len and set(name) <= {'0', '1'}:
            _, s, t = _spence_filename_to_edges(name, n)
            entries.append((name, n, s, t))
    return entries


# ═══════════════════════════════════════════════════════════════════
#  Run on constructed families
# ═══════════════════════════════════════════════════════════════════

def run_constructed_families(k: int):
    """Run Δ^k-DRESS (CUDA) on all constructed hard families."""
    print(f"\n{'╔' + '═' * 68 + '╗'}")
    print(f"{'║'}  PART 1: Constructed Hard Families  (Δ^{k}-DRESS CUDA){' ' * (14 - len(str(k)))}{'║'}")
    print(f"{'╚' + '═' * 68 + '╝'}")

    families = build_all_families(include_random=True)

    all_results = []  # list of family result dicts
    mismatches = 0    # histogram vs multiset disagreements

    for family_name, graphs in sorted(families.items()):
        n_graphs = len(graphs)
        if n_graphs == 0:
            continue

        print(f"\n{'─' * 70}")
        print(f"  {family_name}  ({n_graphs} graphs)")
        print(f"{'─' * 70}")

        t0 = time.perf_counter()
        hist_results = {}  # name → hist_hash
        ms_results = {}    # name → multiset_hash
        msig_results = {}  # name → multiplicity_signature

        for name, n, s, t in graphs:
            hh, mh, hsz, nsub, _, _, msig = compute_fingerprint(n, s, t, k=k)
            hist_results[name] = hh
            ms_results[name] = mh
            msig_results[name] = msig
            print(f"    ✓ {name:<40s}  hist_size={hsz:>6d}  "
                  f"subs={nsub:>5d}  hist={hh[:10]}  ms={mh[:10]}")

        elapsed = time.perf_counter() - t0

        # Analyze separation by histogram
        h_to_names = defaultdict(list)
        for name, fph in hist_results.items():
            h_to_names[fph].append(name)
        n_unique_hist = len(h_to_names)

        # Analyze separation by multiset
        m_to_names = defaultdict(list)
        for name, fph in ms_results.items():
            m_to_names[fph].append(name)
        n_unique_ms = len(m_to_names)

        sep_hist = (n_unique_hist == n_graphs)
        sep_ms = (n_unique_ms == n_graphs)

        # Cross-check: do histogram and multiset agree on equivalence classes?
        agree = True
        for name in hist_results:
            h_group = set(n2 for n2 in hist_results
                          if hist_results[n2] == hist_results[name])
            m_group = set(n2 for n2 in ms_results
                          if ms_results[n2] == ms_results[name])
            if h_group != m_group:
                agree = False
                break

        if not agree:
            mismatches += 1
            print(f"    ⚠  HISTOGRAM vs MULTISET DISAGREE!")
            print(f"       hist: {n_unique_hist} unique, "
                  f"multiset: {n_unique_ms} unique")

        # Group by (n, degree) for within-class separation
        param_classes = defaultdict(list)
        for name, n, s, t in graphs:
            deg = Counter()
            for u, v in zip(s, t):
                deg[u] += 1; deg[v] += 1
            d = deg[0] if deg else 0
            param_classes[(n, d)].append(name)

        within_separated = True
        for (nv, d), names in sorted(param_classes.items()):
            if len(names) < 2:
                continue
            hashes = [hist_results[name] for name in names]
            unique_h = len(set(hashes))
            if unique_h < len(names):
                within_separated = False
                print(f"    ✗ Param class (n={nv}, d={d}): "
                      f"{unique_h}/{len(names)} separated")
            else:
                print(f"    ✓ Param class (n={nv}, d={d}): "
                      f"{len(names)}/{len(names)} fully separated")

        status = "✓ ALL SEPARATED" if sep_hist else (
            "✓ WITHIN-CLASS SEP" if within_separated else "✗ COLLISIONS")
        chk = "✓" if agree else "⚠"
        print(f"\n    {status}  hist={n_unique_hist}/{n_graphs}  "
              f"ms={n_unique_ms}/{n_graphs}  cross-check={chk}  "
              f"{elapsed:.2f}s")

        all_results.append({
            "family": family_name,
            "n_graphs": n_graphs,
            "n_unique_hist": n_unique_hist,
            "n_unique_ms": n_unique_ms,
            "separated": sep_hist,
            "within_separated": within_separated,
            "agree": agree,
            "time": elapsed,
            "hist_results": hist_results,
            "ms_results": ms_results,
        })

    if mismatches:
        print(f"\n  ⚠  {mismatches} families had histogram/multiset "
              f"disagreements!")
    else:
        print(f"\n  ✓ All families: histogram and multiset fingerprints "
              f"agree perfectly.")

    return all_results


# ═══════════════════════════════════════════════════════════════════
#  Run on SRG files from disk
# ═══════════════════════════════════════════════════════════════════

def load_g6_file(filepath: Path, max_graphs: int = 0) -> list[str]:
    """Load graph6 strings from file."""
    import networkx as nx
    strings = []
    with open(filepath) as f:
        for i, line in enumerate(f):
            if max_graphs > 0 and i >= max_graphs:
                break
            line = line.strip()
            if not line:
                continue
            strings.append(line)
    return strings


def _g6_checkpoint_path(g6_path: Path) -> Path:
    """Derive checkpoint filename from g6 filepath."""
    return Path(__file__).resolve().parent / f"g6_checkpoint_{g6_path.stem}.json"


def run_g6_file(filepath: Path, k: int, max_graphs: int = 0) -> dict:
    """Run Δ^k-DRESS (CUDA) on a single g6 file with checkpoint/resume and npz saving."""

    g6_strings = load_g6_file(filepath, max_graphs=max_graphs)
    n_graphs = len(g6_strings)
    if n_graphs == 0:
        return {}

    label = filepath.stem

    print(f"\n{'─' * 70}")
    print(f"  {filepath.name}  ({n_graphs} graphs)")
    print(f"{'─' * 70}")

    # ── Load checkpoint ───────────────────────────────────────────
    cpfile = _g6_checkpoint_path(filepath)
    ckpt = {}
    if cpfile.exists():
        with open(cpfile) as f:
            ckpt = json.load(f)
    if ckpt.get('k') != k:
        if ckpt:
            print(f"  ⚠  Checkpoint k={ckpt.get('k')} ≠ {k} — starting fresh")
        ckpt = {'k': k, 'n_graphs': n_graphs, 'hist': {}, 'ms': {}, 'msig': {}}
    else:
        ckpt['n_graphs'] = n_graphs

    hist_results = {int(i): h for i, h in ckpt.get('hist', {}).items()}
    ms_results = {int(i): h for i, h in ckpt.get('ms', {}).items()}
    msig_results = {int(i): tuple(v) for i, v in ckpt.get('msig', {}).items()}
    start_idx = len(hist_results)

    # ── Raw data directory for npz traceability ───────────────────
    raw_dir = Path(__file__).resolve().parent / f"g6_raw_{label}"
    raw_dir.mkdir(exist_ok=True)

    if start_idx > 0:
        n_unique_resume = len(set(hist_results.values()))
        print(f"  ✓ Resuming from checkpoint: {start_idx}/{n_graphs} done "
              f"({n_unique_resume} unique so far)")
    else:
        print(f"  Raw data dir: {raw_dir}")

    SAVE_EVERY = 50
    t0 = time.perf_counter()
    computed = 0

    for idx in range(start_idx, n_graphs):
        g6 = g6_strings[idx]
        n, s, t = _g6_to_edges(g6)
        hh, mh, hsz, nsub, raw_hist, raw_ms, msig = compute_fingerprint(n, s, t, k=k)
        hist_results[idx] = hh
        ms_results[idx] = mh
        msig_results[idx] = msig
        computed += 1

        # Save raw sparse data per graph
        npz_path = raw_dir / f"{idx:06d}.npz"
        if not npz_path.exists():
            nz = np.nonzero(raw_hist)[0]
            save_dict = {
                'hist_idx': nz.astype(np.int32),
                'hist_val': raw_hist[nz],
            }
            if raw_ms is not None:
                save_dict['multisets'] = raw_ms
            np.savez_compressed(npz_path, **save_dict)

        # Save checkpoint periodically
        if computed % SAVE_EVERY == 0:
            ckpt['hist'] = {str(i): h for i, h in hist_results.items()}
            ckpt['ms'] = {str(i): h for i, h in ms_results.items()}
            ckpt['msig'] = {str(i): list(v) for i, v in msig_results.items()}
            tmp = cpfile.with_suffix('.tmp')
            with open(tmp, 'w') as f:
                json.dump(ckpt, f)
            tmp.rename(cpfile)

        if (idx + 1) % max(1, n_graphs // 20) == 0 or idx + 1 == n_graphs:
            elapsed = time.perf_counter() - t0
            rate = computed / elapsed if elapsed > 0 else 0
            n_unique_so_far = len(set(hist_results.values()))
            print(f"    [{idx+1:>5}/{n_graphs}]  "
                  f"{elapsed:>7.1f}s  ({rate:.1f} g/s)  "
                  f"unique: {n_unique_so_far}", flush=True)

    # Final checkpoint save
    ckpt['hist'] = {str(i): h for i, h in hist_results.items()}
    ckpt['ms'] = {str(i): h for i, h in ms_results.items()}
    ckpt['msig'] = {str(i): list(v) for i, v in msig_results.items()}
    ckpt['completed'] = True
    tmp = cpfile.with_suffix('.tmp')
    with open(tmp, 'w') as f:
        json.dump(ckpt, f)
    tmp.rename(cpfile)

    elapsed = time.perf_counter() - t0

    # Histogram separation
    h_to_indices = defaultdict(list)
    for idx, fph in sorted(hist_results.items()):
        h_to_indices[fph].append(idx)
    n_unique_hist = len(h_to_indices)

    # Multiset separation
    m_to_indices = defaultdict(list)
    for idx, fph in sorted(ms_results.items()):
        m_to_indices[fph].append(idx)
    n_unique_ms = len(m_to_indices)

    # Multiplicity signature separation
    msig_to_indices = defaultdict(list)
    for idx, sig in sorted(msig_results.items()):
        msig_to_indices[sig].append(idx)
    n_unique_msig = len(msig_to_indices)

    # Combined: hist + msig
    combined = defaultdict(list)
    for idx in sorted(hist_results.keys()):
        key = (hist_results[idx], msig_results.get(idx, ()))
        combined[key].append(idx)
    n_unique_combined = len(combined)

    sep_combined = (n_unique_combined == n_graphs)

    # Cross-check (sample-based)
    agree = True
    check_limit = min(n_graphs, 500)
    for idx in list(hist_results.keys())[:check_limit]:
        h_group = set(i for i in hist_results
                      if hist_results[i] == hist_results[idx])
        m_group = set(i for i in ms_results
                      if ms_results[i] == ms_results[idx])
        if h_group != m_group:
            agree = False
            break

    print(f"\n    Unique fingerprints:  hist={n_unique_hist}/{n_graphs}  "
          f"ms={n_unique_ms}/{n_graphs}  "
          f"hist+msig={n_unique_combined}/{n_graphs}")
    status = "✓ FULLY SEPARATED" if sep_combined else "✗ COLLISIONS"
    chk = "✓" if agree else "⚠ DISAGREE"
    print(f"    {status}  cross-check={chk}  "
          f"{elapsed:.1f}s  ({elapsed/max(1,computed):.3f}s/graph)")

    if not sep_combined:
        collisions = {k_: idxs for k_, idxs in combined.items()
                      if len(idxs) > 1}
        n_col = sum(len(v) for v in collisions.values())
        print(f"    Collision groups: {len(collisions)} "
              f"({n_col} graphs in collisions)")
        for fph, idxs in sorted(collisions.items(),
                                 key=lambda x: -len(x[1]))[:10]:
            preview = idxs[:8]
            suffix = f" +{len(idxs)-8}" if len(idxs) > 8 else ""
            print(f"      [{len(idxs):>3}]  {preview}{suffix}")

    if not agree:
        print(f"    ⚠  Multiset found {n_unique_ms} classes vs "
              f"histogram {n_unique_hist}")

    return {
        "file": filepath.name,
        "n_graphs": n_graphs,
        "n_unique_hist": n_unique_hist,
        "n_unique_ms": n_unique_ms,
        "n_unique_combined": n_unique_combined,
        "separated": sep_combined,
        "agree": agree,
        "time": elapsed,
    }


def run_srg_files(k: int, max_graphs: int = 0, extra_g6: list[Path] | None = None):
    """Run Δ^k-DRESS (CUDA) on all SRG g6 files in datasets/SRGs/ plus any extra g6 files."""

    files = []
    if SRG_DIR.exists():
        files = sorted(f for f in SRG_DIR.iterdir()
                       if f.suffix in (".txt", ".g6")
                       and not f.name.endswith(":Zone.Identifier"))
    if extra_g6:
        for p in extra_g6:
            if p.exists() and p not in files:
                files.append(p)

    if not files:
        print("\n  No g6 files found.")
        return []

    print(f"\n{'╔' + '═' * 68 + '╗'}")
    print(f"{'║'}  PART 2: SRG g6 Files  (Δ^{k}-DRESS CUDA){' ' * (27 - len(str(k)))}{'║'}")
    print(f"{'╚' + '═' * 68 + '╝'}")
    print(f"  Files: {[f.name for f in files]}")

    all_results = []
    for filepath in files:
        result = run_g6_file(filepath, k=k, max_graphs=max_graphs)
        if result:
            all_results.append(result)

    return all_results


# ═══════════════════════════════════════════════════════════════════
#  Run on Spence-format directories (binary filenames)
# ═══════════════════════════════════════════════════════════════════

SPENCE_DIR = Path(__file__).resolve().parent.parent / "datasets" / "srgs"
SPENCE_DATASETS = Path(__file__).resolve().parent.parent / "datasets" / "srgs"
SPENCE_DEFAULT_BZ2 = SPENCE_DATASETS / "36-15-6-6.bz2"


def load_spence_bz2(filepath: Path,
                     max_graphs: int = 0) -> list[tuple[str, int, list, list]]:
    """Load Spence-format graphs from a bz2-compressed or plain-text adjacency-matrix file.

    Format: each graph is n lines of n chars ('0'/'1'), full adjacency matrix
    rows, separated by blank lines.  Detects bz2 vs plain text automatically.
    Returns list of (label, n, sources, targets).
    """
    import bz2

    entries: list[tuple[str, int, list, list]] = []
    current_rows: list[str] = []

    def _flush():
        if not current_rows:
            return
        n = len(current_rows)
        sources, targets = [], []
        for i in range(n):
            row = current_rows[i]
            for j in range(i + 1, n):
                if row[j] == '1':
                    sources.append(i)
                    targets.append(j)
        entries.append((f"spence-{len(entries)}", n, sources, targets))

    # Auto-detect format: try bz2 first, fall back to plain text
    is_bz2 = filepath.suffix == '.bz2'
    opener = (lambda: bz2.open(filepath, 'rt')) if is_bz2 else (lambda: open(filepath, 'r'))

    with opener() as f:
        for line in f:
            line = line.rstrip('\n')
            if not line:
                _flush()
                current_rows = []
                if max_graphs > 0 and len(entries) >= max_graphs:
                    break
            elif set(line) <= {'0', '1'}:
                current_rows.append(line)
            # else: skip header / comment lines

    # Handle last graph if no trailing blank line
    if max_graphs <= 0 or len(entries) < max_graphs:
        _flush()

    return entries


def _checkpoint_path(bz2_path: Path) -> Path:
    """Derive checkpoint filename from bz2 filename."""
    return Path(__file__).resolve().parent / f"spence_checkpoint_{bz2_path.stem}.json"


def _load_checkpoint(bz2_path: Path) -> dict:
    """Load existing Spence checkpoint or return empty dict."""
    cp = _checkpoint_path(bz2_path)
    if cp.exists():
        with open(cp) as f:
            return json.load(f)
    return {}


def _save_checkpoint(bz2_path: Path, data: dict):
    """Atomically write checkpoint (write tmp then rename)."""
    cp = _checkpoint_path(bz2_path)
    tmp = cp.with_suffix('.tmp')
    with open(tmp, 'w') as f:
        json.dump(data, f)
    tmp.rename(cp)


def run_spence_bz2(k: int, max_graphs: int = 0, bz2_path: Path | None = None):
    """Run Δ^k-DRESS (CUDA) on a Spence bz2 dataset.

    Supports checkpoint/resume: fingerprints are saved to
    spence_checkpoint_<stem>.json after every batch of graphs.
    If the process dies, re-running picks up from where it left off.
    """
    if bz2_path is None:
        bz2_path = SPENCE_DEFAULT_BZ2
    if not bz2_path.exists():
        print(f"\n  Spence bz2 file not found: {bz2_path}")
        return []

    label = bz2_path.stem  # e.g. "35-18-9-9"
    cpfile = _checkpoint_path(bz2_path)

    print(f"\n{'╔' + '═' * 68 + '╗'}")
    print(f"{'║'}  PART 3: Spence {label}  (Δ^{k}-DRESS CUDA){' ' * max(0, 34 - len(label) - len(str(k)))}{'║'}")
    print(f"{'╚' + '═' * 68 + '╝'}")
    print(f"  File: {bz2_path}")
    print(f"  Checkpoint: {cpfile}")

    print(f"  Loading graphs ...", end="  ", flush=True)
    t_load = time.perf_counter()
    graphs = load_spence_bz2(bz2_path, max_graphs=max_graphs)
    t_load = time.perf_counter() - t_load
    n_graphs = len(graphs)
    print(f"{n_graphs} graphs loaded in {t_load:.1f}s")

    if n_graphs == 0:
        return []

    # Verify first graph
    _, n, s, t = graphs[0]
    deg = Counter()
    for u, v in zip(s, t):
        deg[u] += 1; deg[v] += 1
    d = deg[0] if deg else 0
    print(f"  Verified: n={n}, d={d}, edges={len(s)}")

    # ── Load checkpoint ───────────────────────────────────────────
    ckpt = _load_checkpoint(bz2_path)
    # Validate checkpoint: only k must match; n_graphs can grow
    if ckpt.get('k') != k:
        if ckpt:
            print(f"  ⚠  Checkpoint k={ckpt.get('k')} ≠ {k} — starting fresh")
        ckpt = {'k': k, 'n_graphs': n_graphs, 'hist': {}, 'ms': {}, 'msig': {}}
    else:
        ckpt['n_graphs'] = n_graphs

    hist_results = {int(i): h for i, h in ckpt.get('hist', {}).items()}
    ms_results = {int(i): h for i, h in ckpt.get('ms', {}).items()}
    msig_results = {int(i): tuple(v) for i, v in ckpt.get('msig', {}).items()}
    start_idx = len(hist_results)

    if start_idx > 0:
        n_unique_resume = len(set(hist_results.values()))
        print(f"  ✓ Resuming from checkpoint: {start_idx}/{n_graphs} done "
              f"({n_unique_resume} unique so far)")
    else:
        print(f"  Starting fresh (no checkpoint)")

    # ── Raw data directory for traceability ────────────────────────
    raw_dir = Path(__file__).resolve().parent / f"spence_raw_{bz2_path.stem}"
    raw_dir.mkdir(exist_ok=True)
    print(f"  Raw data dir: {raw_dir}")

    print(f"  Running Δ^{k}-DRESS with keep_multisets=True ...")

    SAVE_EVERY = 50  # checkpoint every N graphs

    t0 = time.perf_counter()
    computed = 0

    for idx in range(start_idx, n_graphs):
        fname, n, s, t = graphs[idx]
        hh, mh, hsz, nsub, raw_hist, raw_ms, msig = compute_fingerprint(n, s, t, k=k)
        hist_results[idx] = hh
        ms_results[idx] = mh
        msig_results[idx] = msig
        computed += 1

        # Save raw sparse data per graph
        npz_path = raw_dir / f"{idx:06d}.npz"
        if not npz_path.exists():
            nz = np.nonzero(raw_hist)[0]
            save_dict = {
                'hist_idx': nz.astype(np.int32),
                'hist_val': raw_hist[nz],
            }
            if raw_ms is not None:
                save_dict['multisets'] = raw_ms
            np.savez_compressed(npz_path, **save_dict)

        # Save checkpoint periodically
        if computed % SAVE_EVERY == 0:
            ckpt['hist'] = {str(i): h for i, h in hist_results.items()}
            ckpt['ms'] = {str(i): h for i, h in ms_results.items()}
            ckpt['msig'] = {str(i): list(v) for i, v in msig_results.items()}
            _save_checkpoint(bz2_path, ckpt)

        if (idx + 1) % max(1, n_graphs // 40) == 0 or idx + 1 == n_graphs:
            elapsed = time.perf_counter() - t0
            rate = computed / elapsed if elapsed > 0 else 0
            n_unique_so_far = len(set(hist_results.values()))
            print(f"    [{idx+1:>5}/{n_graphs}]  "
                  f"{elapsed:>7.1f}s  ({rate:.1f} g/s)  "
                  f"unique: {n_unique_so_far}", flush=True)

    # Final checkpoint save
    ckpt['hist'] = {str(i): h for i, h in hist_results.items()}
    ckpt['ms'] = {str(i): h for i, h in ms_results.items()}
    ckpt['msig'] = {str(i): list(v) for i, v in msig_results.items()}
    ckpt['completed'] = True
    _save_checkpoint(bz2_path, ckpt)

    elapsed = time.perf_counter() - t0

    # Histogram separation
    h_to_indices = defaultdict(list)
    for idx, fph in sorted(hist_results.items()):
        h_to_indices[fph].append(idx)
    n_unique_hist = len(h_to_indices)

    # Multiset separation
    m_to_indices = defaultdict(list)
    for idx, fph in sorted(ms_results.items()):
        m_to_indices[fph].append(idx)
    n_unique_ms = len(m_to_indices)

    # Multiplicity signature separation
    msig_to_indices = defaultdict(list)
    for idx, sig in sorted(msig_results.items()):
        msig_to_indices[sig].append(idx)
    n_unique_msig = len(msig_to_indices)

    # Combined: hist + msig (the library-level fingerprint)
    combined = defaultdict(list)
    for idx in sorted(hist_results.keys()):
        key = (hist_results[idx], msig_results.get(idx, ()))
        combined[key].append(idx)
    n_unique_combined = len(combined)

    sep_hist = (n_unique_hist == n_graphs)
    sep_ms = (n_unique_ms == n_graphs)
    sep_combined = (n_unique_combined == n_graphs)

    # Cross-check (sample-based)
    agree = True
    check_limit = min(n_graphs, 500)
    for idx in list(hist_results.keys())[:check_limit]:
        h_group = set(i for i in hist_results
                      if hist_results[i] == hist_results[idx])
        m_group = set(i for i in ms_results
                      if ms_results[i] == ms_results[idx])
        if h_group != m_group:
            agree = False
            break

    print(f"\n    Unique fingerprints:  hist={n_unique_hist}/{n_graphs}  "
          f"ms={n_unique_ms}/{n_graphs}  "
          f"hist+msig={n_unique_combined}/{n_graphs}")
    status = "✓ FULLY SEPARATED" if sep_combined else "✗ COLLISIONS"
    chk = "✓" if agree else "⚠ DISAGREE"
    print(f"    {status}  cross-check={chk}  "
          f"{elapsed:.1f}s  ({elapsed/max(1,computed):.3f}s/graph)")

    if not sep_combined:
        collisions = {k: idxs for k, idxs in combined.items()
                      if len(idxs) > 1}
        n_col = sum(len(v) for v in collisions.values())
        print(f"    Collision groups: {len(collisions)} "
              f"({n_col} graphs in collisions)")
        for fph, idxs in sorted(collisions.items(),
                                 key=lambda x: -len(x[1]))[:10]:
            preview = idxs[:8]
            suffix = f" +{len(idxs)-8}" if len(idxs) > 8 else ""
            print(f"      [{len(idxs):>3}]  {preview}{suffix}  "
                  f"{fph[:16]}...")

    if not agree:
        print(f"    ⚠  Multiset found {n_unique_ms} classes vs "
              f"histogram {n_unique_hist}")

    return [{
        "file": f"Spence-{label}",
        "n_graphs": n_graphs,
        "n_unique_hist": n_unique_hist,
        "n_unique_ms": n_unique_ms,
        "n_unique_combined": n_unique_combined,
        "separated": sep_combined,
        "agree": agree,
        "time": elapsed,
    }]


def run_spence_dirs(k: int, max_graphs: int = 0):
    """Run Δ^k-DRESS (CUDA) on Spence-format SRG directories.

    Expects datasets/spence/<param>/ directories where each contains
    zero-byte files whose names are binary adjacency strings.
    """
    if not SPENCE_DIR.exists():
        print(f"\n  Spence data directory not found: {SPENCE_DIR}")
        return []

    subdirs = sorted(d for d in SPENCE_DIR.iterdir() if d.is_dir())
    if not subdirs:
        print(f"\n  No subdirectories in {SPENCE_DIR}")
        return []

    print(f"\n{'╔' + '═' * 68 + '╗'}")
    print(f"{'║'}  PART 3: Spence SRG Collections  (Δ^{k}-DRESS CUDA){' ' * (17 - len(str(k)))}{'║'}")
    print(f"{'╚' + '═' * 68 + '╝'}")
    print(f"  Data dir: {SPENCE_DIR}")
    print(f"  Subdirs:  {[d.name for d in subdirs]}")

    all_results = []

    for dirpath in subdirs:
        # Infer n from first valid filename length: C(n,2) = len
        # Default to 36 for SRG(36,15,6,6)
        n_vertices = 36
        for f in dirpath.iterdir():
            name = f.name
            if set(name) <= {'0', '1'} and len(name) > 0:
                bits = len(name)
                # Solve C(n,2) = bits → n = (1 + sqrt(1 + 8*bits)) / 2
                import math
                n_vertices = int((1 + math.isqrt(1 + 8 * bits)) // 2)
                break

        graphs = load_spence_dir(dirpath, n=n_vertices,
                                 max_graphs=max_graphs)
        n_graphs = len(graphs)
        if n_graphs == 0:
            continue

        # Verify first graph
        _, n, s, t = graphs[0]
        m = len(s)
        from collections import Counter as Ctr
        deg = Ctr()
        for u, v in zip(s, t):
            deg[u] += 1; deg[v] += 1
        d = deg[0] if deg else 0

        print(f"\n{'─' * 70}")
        print(f"  {dirpath.name}  ({n_graphs} graphs, n={n}, d={d})")
        print(f"{'─' * 70}")

        t0 = time.perf_counter()
        hist_results = {}
        ms_results = {}
        msig_results = {}

        for idx, (fname, n, s, t) in enumerate(graphs):
            hh, mh, hsz, nsub, _, _, msig = compute_fingerprint(n, s, t, k=k)
            hist_results[idx] = hh
            ms_results[idx] = mh
            msig_results[idx] = msig
            if (idx + 1) % max(1, n_graphs // 20) == 0 or idx + 1 == n_graphs:
                elapsed = time.perf_counter() - t0
                rate = (idx + 1) / elapsed if elapsed > 0 else 0
                print(f"    [{idx+1:>5}/{n_graphs}]  "
                      f"{elapsed:>6.1f}s  ({rate:.1f} graphs/s)",
                      flush=True)

        elapsed = time.perf_counter() - t0

        h_to_indices = defaultdict(list)
        for idx, fph in sorted(hist_results.items()):
            h_to_indices[fph].append(idx)
        n_unique_hist = len(h_to_indices)

        m_to_indices = defaultdict(list)
        for idx, fph in sorted(ms_results.items()):
            m_to_indices[fph].append(idx)
        n_unique_ms = len(m_to_indices)

        sep_hist = (n_unique_hist == n_graphs)

        # Cross-check (sample-based for large collections)
        agree = True
        check_limit = min(n_graphs, 500)
        for idx in list(hist_results.keys())[:check_limit]:
            h_group = set(i for i in hist_results
                          if hist_results[i] == hist_results[idx])
            m_group = set(i for i in ms_results
                          if ms_results[i] == ms_results[idx])
            if h_group != m_group:
                agree = False
                break

        print(f"\n    Unique fingerprints:  hist={n_unique_hist}/{n_graphs}  "
              f"ms={n_unique_ms}/{n_graphs}")
        status = "✓ FULLY SEPARATED" if sep_hist else "✗ COLLISIONS"
        chk = "✓" if agree else "⚠ DISAGREE"
        print(f"    {status}  cross-check={chk}  "
              f"{elapsed:.1f}s  ({elapsed/n_graphs:.3f}s/graph)")

        if not sep_hist:
            collisions = {fph: idxs for fph, idxs in h_to_indices.items()
                          if len(idxs) > 1}
            print(f"    Histogram collision groups: {len(collisions)}")
            for fph, idxs in sorted(collisions.items(),
                                     key=lambda x: -len(x[1]))[:5]:
                preview = idxs[:8]
                suffix = f" +{len(idxs)-8}" if len(idxs) > 8 else ""
                print(f"      [{len(idxs):>3}]  {preview}{suffix}  "
                      f"{fph[:16]}...")

        all_results.append({
            "file": dirpath.name,
            "n_graphs": n_graphs,
            "n_unique_hist": n_unique_hist,
            "n_unique_ms": n_unique_ms,
            "separated": sep_hist,
            "agree": agree,
            "time": elapsed,
        })

    return all_results


# ═══════════════════════════════════════════════════════════════════
#  Grand summary
# ═══════════════════════════════════════════════════════════════════

def print_summary(constructed_results, file_results, k):
    print(f"\n{'═' * 70}")
    print(f"  GRAND SUMMARY — Δ^{k}-DRESS (CUDA)")
    print(f"{'═' * 70}")

    all_pass = True
    all_agree = True
    total_graphs = 0
    total_unique_hist = 0
    total_unique_ms = 0
    total_time = 0.0

    # Constructed families
    if constructed_results:
        print(f"\n  {'Family':<40s}  {'Hist':>9s}  {'Multiset':>9s}  "
              f"{'Chk':>3s}  {'Time':>7s}")
        print(f"  {'─' * 40}  {'─' * 9}  {'─' * 9}  {'─' * 3}  {'─' * 7}")
        for r in constructed_results:
            tag = "✓" if r["separated"] else (
                "~" if r.get("within_separated") else "✗")
            chk = "✓" if r["agree"] else "⚠"
            print(f"  {r['family']:<40s}  "
                  f"{tag} {r['n_unique_hist']:>3}/{r['n_graphs']:<4}  "
                  f"{tag} {r['n_unique_ms']:>3}/{r['n_graphs']:<4}  "
                  f" {chk}   {r['time']:>6.2f}s")
            total_graphs += r["n_graphs"]
            total_unique_hist += r["n_unique_hist"]
            total_unique_ms += r["n_unique_ms"]
            total_time += r["time"]
            if not r["separated"]:
                all_pass = False
            if not r["agree"]:
                all_agree = False

    # SRG files
    if file_results:
        print()
        for r in file_results:
            tag = "✓" if r["separated"] else "✗"
            chk = "✓" if r["agree"] else "⚠"
            print(f"  {r['file']:<40s}  "
                  f"{tag} {r['n_unique_hist']:>3}/{r['n_graphs']:<4}  "
                  f"{tag} {r['n_unique_ms']:>3}/{r['n_graphs']:<4}  "
                  f" {chk}   {r['time']:>6.1f}s")
            total_graphs += r["n_graphs"]
            total_unique_hist += r["n_unique_hist"]
            total_unique_ms += r.get("n_unique_ms", r["n_unique_hist"])
            total_time += r["time"]
            if not r["separated"]:
                all_pass = False
            if not r.get("agree", True):
                all_agree = False

    print(f"\n  {'─' * 65}")
    print(f"  Histogram:  {total_unique_hist}/{total_graphs} unique")
    print(f"  Multiset:   {total_unique_ms}/{total_graphs} unique")
    print(f"  Cross-check: {'✓ AGREE' if all_agree else '⚠ DISAGREEMENTS'}")
    print(f"  Time:        {total_time:.1f}s")

    if all_pass and all_agree:
        print(f"\n  ★★★  Δ^{k}-DRESS (CUDA) DESTROYS ALL HARD FAMILIES  ★★★")
        print(f"  ★★★  Histogram & multiset fingerprints both confirm  ★★★")
    elif all_pass:
        print(f"\n  ★ Δ^{k}-DRESS separates all — but cross-check has warnings.")
    else:
        print(f"\n  Some collisions remain — try k={k+1}.")


# ═══════════════════════════════════════════════════════════════════
#  CLI
# ═══════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Δ^k-DRESS (CUDA) vs ALL hard graph families")
    parser.add_argument("--k", type=int, default=1,
                        help="Deletion depth (default: 1)")
    parser.add_argument("--max-graphs", type=int, default=0,
                        help="Max graphs per SRG file (0 = all)")
    parser.add_argument("--skip-files", action="store_true",
                        help="Skip SRG file datasets")
    parser.add_argument("--skip-constructed", action="store_true",
                        help="Skip constructed families")
    parser.add_argument("--skip-spence", action="store_true",
                        help="Skip Spence bz2 datasets")
    parser.add_argument("--spence-bz2", type=str, default=None,
                        help="Path to Spence .bz2 file (default: all in datasets/)")
    parser.add_argument("--g6", type=str, nargs="+", default=None,
                        help="Extra g6 file(s) to run (in addition to datasets/SRGs/)")
    args = parser.parse_args()

    k = args.k

    print(f"╔{'═' * 68}╗")
    print(f"║  Δ^{k}-DRESS (CUDA) vs ALL Hard Graph Families{' ' * (21 - len(str(k)))}║")
    print(f"╠{'═' * 68}╣")
    print(f"║  Backend:   CUDA (keep_multisets=True){' ' * 30}║")
    print(f"║  k:         {k:<55}║")
    print(f"╚{'═' * 68}╝")

    t_total = time.perf_counter()

    constructed_results = []
    file_results = []
    spence_results = []

    if not args.skip_constructed:
        constructed_results = run_constructed_families(k=k)

    if not args.skip_files:
        extra_g6 = [Path(p) for p in args.g6] if args.g6 else None
        file_results = run_srg_files(k=k, max_graphs=args.max_graphs,
                                     extra_g6=extra_g6)

    if not args.skip_spence:
        if args.spence_bz2:
            bz2_files = [Path(args.spence_bz2)]
        else:
            # Discover bz2 files + plain text Spence files (NN-NN-N-N pattern)
            import re
            bz2_files = sorted(SPENCE_DATASETS.glob("*.bz2"))
            spence_re = re.compile(r'^\d+-\d+-\d+-\d+$')
            text_files = sorted(p for p in SPENCE_DATASETS.iterdir()
                                if p.is_file() and spence_re.match(p.name))
            bz2_files = bz2_files + text_files
        for bz2f in bz2_files:
            spence_results += run_spence_bz2(k=k, max_graphs=args.max_graphs,
                                             bz2_path=bz2f)

    t_total = time.perf_counter() - t_total

    print_summary(constructed_results, file_results + spence_results, k)
    print(f"\n  Wall time: {t_total:.1f}s")


if __name__ == "__main__":
    main()
