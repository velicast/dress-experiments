#!/usr/bin/env python3
"""
cfi.py  -  CFI graph isomorphism test via Delta^k-DRESS

Tests whether Delta^k-DRESS (iterated node-deletion DRESS) can distinguish
non-isomorphic CFI graph pairs.

Expects CFI edge-list pairs in datasets/cfi/ (relative to repo root).
Each pair has files <name>1.txt and <name>2.txt with one edge "u v" per
line (single direction).

Usage:
    python cfi.py                           # Delta^2 on all pairs
    python cfi.py --k 0                     # plain DRESS baseline
    python cfi.py --k 2 --only triangle 4-clique
    python cfi.py --k 3 --max-n 96
"""

from __future__ import annotations

import argparse
import re
import time
from collections import Counter
from math import comb
from pathlib import Path

from dress.cuda import delta_dress_fit


# ═══════════════════════════════════════════════════════════════════
#  File I/O
# ═══════════════════════════════════════════════════════════════════

def load_graph(filepath: Path) -> tuple[int, list[int], list[int]]:
    """Load edge-list file, relabel to 0..n-1, return (n, sources, targets)."""
    edges = []
    with open(filepath) as f:
        for line in f:
            parts = line.split()
            if len(parts) >= 2:
                edges.append((int(parts[0]), int(parts[1])))

    node_set = set()
    for u, v in edges:
        node_set.add(u)
        node_set.add(v)

    node_map = {nd: i for i, nd in enumerate(sorted(node_set))}
    n = len(node_set)
    sources = [node_map[u] for u, v in edges]
    targets = [node_map[v] for u, v in edges]
    return n, sources, targets


# ═══════════════════════════════════════════════════════════════════
#  Pair discovery
# ═══════════════════════════════════════════════════════════════════

def discover_pairs(cfi_dir: Path) -> list[tuple[str, Path, Path]]:
    """Find CFI pairs: files matching <prefix><digit>.txt grouped by prefix."""
    pat = re.compile(r'^(.+?)(\d+)\.txt$')
    groups: dict[str, list[Path]] = {}

    for f in sorted(cfi_dir.iterdir()):
        if not f.is_file() or f.suffix != '.txt':
            continue
        if f.name.startswith('mapping'):
            continue
        m = pat.match(f.name)
        if m:
            groups.setdefault(m.group(1), []).append(f)

    pairs = []
    for prefix, flist in sorted(groups.items()):
        if len(flist) != 2:
            print(f"  Warning: prefix '{prefix}' has {len(flist)} files, expected 2")
            continue
        flist.sort()
        name = prefix.rstrip('-_')
        pairs.append((name, flist[0], flist[1]))

    pairs.sort(key=lambda p: p[0])
    return pairs


# ═══════════════════════════════════════════════════════════════════
#  Verification
# ═══════════════════════════════════════════════════════════════════

def verify_pair(name: str, n1: int, s1: list, t1: list,
                n2: int, s2: list, t2: list) -> bool:
    ok = True

    if n1 != n2:
        print(f"  X {name}: node count mismatch ({n1} vs {n2})")
        ok = False

    if len(s1) != len(s2):
        print(f"  X {name}: edge count mismatch ({len(s1)} vs {len(s2)})")
        ok = False

    for label, n, s, t in [("G1", n1, s1, t1), ("G2", n2, s2, t2)]:
        deg = Counter()
        for u, v in zip(s, t):
            deg[u] += 1
            deg[v] += 1
        degrees = set(deg.values())
        if len(degrees) > 1:
            print(f"  X {name} {label}: not regular")
            ok = False
        else:
            d = degrees.pop() if degrees else 0
            print(f"  ok {name} {label}: {d}-regular on {n} nodes, {len(s)} edges")

    return ok


# ═══════════════════════════════════════════════════════════════════
#  Main
# ═══════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="CFI graph isomorphism test via Delta^k-DRESS")
    parser.add_argument("--k", type=int, default=2,
                        help="Deletion depth (default: 2). k=0 is plain DRESS.")
    parser.add_argument("--only", nargs="+", default=[],
                        help="Run only on pairs whose name contains any of these strings.")
    parser.add_argument("--max-n", type=int, default=0,
                        help="Skip pairs with more than N nodes (0 = no limit).")
    args = parser.parse_args()

    k = args.k
    max_n = args.max_n
    epsilon = 1e-6
    max_iter = 100

    # Resolve datasets/cfi/ relative to this script
    script_dir = Path(__file__).resolve().parent
    cfi_dir = script_dir.parent / "datasets" / "cfi"
    if not cfi_dir.exists():
        cfi_dir = Path("datasets/cfi")
    if not cfi_dir.exists():
        print("Cannot find datasets/cfi/ directory")
        return 1

    print("=" * 64)
    print(f"  CFI Graph Isomorphism Test (Python / dress-graph)")
    print(f"  Delta^{k}-DRESS", end="")
    if max_n > 0:
        print(f"  (max-n={max_n})", end="")
    print()
    print("=" * 64)
    print()

    pairs = discover_pairs(cfi_dir)

    # Filter
    if args.only:
        pairs = [p for p in pairs
                 if any(f in p[0] for f in args.only)]

    if not pairs:
        print("No CFI pairs found!")
        return 1

    print(f"Found {len(pairs)} pair(s):\n")

    # Load
    loaded = []
    for name, f1, f2 in pairs:
        n1, s1, t1 = load_graph(f1)
        n2, s2, t2 = load_graph(f2)
        print(f"  {name:<20s}  {n1} nodes, {len(s1)} edges")
        loaded.append((name, n1, s1, t1, n2, s2, t2))

    # Verify
    print()
    print("=" * 64)
    print("  PRE-FLIGHT VERIFICATION")
    print("=" * 64)
    print()

    all_ok = True
    for name, n1, s1, t1, n2, s2, t2 in loaded:
        if not verify_pair(name, n1, s1, t1, n2, s2, t2):
            all_ok = False

    if not all_ok:
        print("\n  VERIFICATION FAILED -- aborting\n")
        return 1
    print("\n  ALL CHECKS PASSED")

    # Run
    print()
    print("=" * 64)
    print(f"  Exhaustive Delta^{k}-DRESS  (epsilon={epsilon})")
    print("=" * 64)
    print()

    results = []  # (name, result): 1=distinguished, 0=fail, -1=skipped

    for name, n1, s1, t1, n2, s2, t2 in loaded:
        if max_n > 0 and n1 > max_n:
            results.append((name, -1))
            print(f"  {name:<20s}  ({n1} nodes > {max_n}) ... ---")
            continue

        n_subsets = comb(n1, k)
        print(f"  {name:<20s}  (C({n1},{k}) = {n_subsets} subgraphs) ... ",
              end="", flush=True)

        t0 = time.perf_counter()
        h1 = delta_dress_fit(n1, s1, t1, k=k, epsilon=epsilon,
                             max_iterations=max_iter)
        h2 = delta_dress_fit(n2, s2, t2, k=k, epsilon=epsilon,
                             max_iterations=max_iter)
        dt = time.perf_counter() - t0

        same = (h1.histogram == h2.histogram)
        results.append((name, 0 if same else 1))

        total = sum(h1.histogram)
        tag = "FAIL (identical)" if same else "DISTINGUISHED"
        print(f"{tag:<20s}  ({dt:.1f}s, {total} values)")

    # Summary
    print()
    print("=" * 64)
    print(f"  SUMMARY  (Delta^{k}-DRESS)")
    print("=" * 64)
    print()

    print(f"{'Pair':<20s}  {'Result':>12s}")
    print("-" * 34)
    for name, r in results:
        cell = "---" if r < 0 else ("Y" if r == 1 else "N")
        print(f"{name:<20s}  {cell:>12s}")

    dist = sum(1 for _, r in results if r == 1)
    fail = sum(1 for _, r in results if r == 0)
    skip = sum(1 for _, r in results if r < 0)
    print(f"\nDistinguished: {dist}/{dist + fail}", end="")
    if skip:
        print(f" ({skip} skipped)", end="")
    print()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
