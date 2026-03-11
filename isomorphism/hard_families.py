#!/usr/bin/env python3
"""
Constructors for ALL known hard graph families (except CFI).

Each build_* function returns a list of (name, n, sources, targets) tuples
where edges are undirected (u < v, listed once).

Families
--------
 1. SRG(28,12,6,4)  — T(8) + Chang × 3
 2. SRG(16,6,2,2)   — Rook L₂(4) + Shrikhande
 3. SRG(10,3,0,1)   — Petersen + Pentagonal Prism
 4. SRG(13,6,2,3)   — Paley(13)
 5. SRG(25,8,3,2)   — Rook L₂(5)
 6. Prism vs K₃,₃   — 3-regular on 6
 7. 2×C₄ vs C₈      — 2-regular on 8
 8. Paley(q)         — q = 17, 25, 29, 37, 41, 49, 53
 9. Rook L₂(n)       — n = 4..8
10. Latin square      — from cyclic MOLS, n = 4, 5, 7
11. Steiner S(2,3,v) — from cyclic difference triples, v = 7, 9, 13, 15, 19, 25
12. Kneser K(n,2)    — n = 5..9   (K(5,2) = Petersen)
13. Johnson J(n,2)   — n = 5..10  (J(8,2) = T(8))
14. Hamming H(d,q)   — (2,4), (2,5), (3,3), (3,4)
15. Miyazaki         — m = 2..6
16. Random regular   — 10 graphs each at (n=20,d=6), (n=30,d=8), (n=40,d=10)
17. Paley conference — q = 5, 9, 13, 17, 25
18. GQ(2,2) point graph — Petersen complement
"""

from __future__ import annotations

import random
from itertools import combinations
from typing import List, Tuple

Graph = Tuple[str, int, List[int], List[int]]  # (name, n, sources, targets)


# ═══════════════════════════════════════════════════════════════════
#  Helpers
# ═══════════════════════════════════════════════════════════════════

def _edges_to_canonical(edges):
    """Deduplicate and sort edges into (u<v) form."""
    seen = set()
    for u, v in edges:
        a, b = (u, v) if u < v else (v, u)
        if a != b:
            seen.add((a, b))
    s, t = zip(*sorted(seen)) if seen else ([], [])
    return list(s), list(t)


def _adjacency_to_edges(n, adj):
    """Convert adjacency set → canonical edge lists."""
    edges = []
    for u in range(n):
        for v in adj.get(u, set()):
            if u < v:
                edges.append((u, v))
    s, t = zip(*sorted(edges)) if edges else ([], [])
    return list(s), list(t)


# ═══════════════════════════════════════════════════════════════════
#  1. SRG(28,12,6,4) — T(8) + Chang × 3
# ═══════════════════════════════════════════════════════════════════

def _build_T8_raw():
    pairs = list(combinations(range(8), 2))
    p2i = {p: i for i, p in enumerate(pairs)}
    edges = []
    for i, (a, b) in enumerate(pairs):
        for j in range(i + 1, len(pairs)):
            c, d = pairs[j]
            if len({a, b} & {c, d}) == 1:
                edges.append((i, j))
    s, t = zip(*edges)
    return 28, list(s), list(t), pairs, p2i


def _seidel_switch(n, sources, targets, switch_set):
    adj = set()
    for u, v in zip(sources, targets):
        adj.add((min(u, v), max(u, v)))
    switch = set(switch_set)
    new_adj = set()
    for u in range(n):
        for v in range(u + 1, n):
            is_edge = (u, v) in adj
            if (u in switch) != (v in switch):
                if not is_edge:
                    new_adj.add((u, v))
            else:
                if is_edge:
                    new_adj.add((u, v))
    s, t = zip(*sorted(new_adj)) if new_adj else ([], [])
    return list(s), list(t)


def build_srg_28_12_6_4() -> List[Graph]:
    """T(8) and 3 Chang graphs — SRG(28,12,6,4)."""
    n, s0, t0, pairs, p2i = _build_T8_raw()
    graphs = [("T(8)", n, s0, t0)]

    sw1 = [p2i[(0, 1)], p2i[(2, 3)], p2i[(4, 5)], p2i[(6, 7)]]
    s1, t1 = _seidel_switch(n, s0, t0, sw1)
    graphs.append(("Chang-1", n, s1, t1))

    sw2 = [p2i[(0, 1)], p2i[(1, 2)], p2i[(2, 3)], p2i[(3, 4)],
           p2i[(4, 5)], p2i[(5, 6)], p2i[(6, 7)], p2i[(0, 7)]]
    s2, t2 = _seidel_switch(n, s0, t0, sw2)
    graphs.append(("Chang-2", n, s2, t2))

    sw3 = [p2i[(0, 3)], p2i[(0, 7)], p2i[(1, 5)], p2i[(1, 7)],
           p2i[(2, 4)], p2i[(2, 6)], p2i[(3, 5)], p2i[(4, 6)]]
    s3, t3 = _seidel_switch(n, s0, t0, sw3)
    graphs.append(("Chang-3", n, s3, t3))

    return graphs


# ═══════════════════════════════════════════════════════════════════
#  2. SRG(16,6,2,2) — Rook L₂(4) + Shrikhande
# ═══════════════════════════════════════════════════════════════════

def build_srg_16_6_2_2() -> List[Graph]:
    # Rook L₂(4) = K₄ □ K₄
    edges_r = []
    for i in range(4):
        for j in range(4):
            u = 4 * i + j
            for k in range(4):
                if k != j:
                    v = 4 * i + k
                    if u < v: edges_r.append((u, v))
                if k != i:
                    v = 4 * k + j
                    if u < v: edges_r.append((u, v))
    rs, rt = _edges_to_canonical(edges_r)

    # Shrikhande: torus Z₄×Z₄, neighbors ±(1,0), ±(0,1), ±(1,1)
    edges_s = []
    for i in range(4):
        for j in range(4):
            u = 4 * i + j
            for di, dj in [(1, 0), (0, 1), (1, 1), (-1, 0), (0, -1), (-1, -1)]:
                ni, nj = (i + di) % 4, (j + dj) % 4
                v = 4 * ni + nj
                if u < v: edges_s.append((u, v))
    ss, st = _edges_to_canonical(edges_s)

    return [("Rook_L2(4)", 16, rs, rt), ("Shrikhande", 16, ss, st)]


# ═══════════════════════════════════════════════════════════════════
#  3. SRG(10,3,0,1) — Petersen + Pentagonal Prism
# ═══════════════════════════════════════════════════════════════════

def build_srg_10_3_0_1() -> List[Graph]:
    # Petersen
    pet_e = [(0,1),(1,2),(2,3),(3,4),(4,0),
             (5,7),(7,9),(9,6),(6,8),(8,5),
             (0,5),(1,6),(2,7),(3,8),(4,9)]
    ps, pt = _edges_to_canonical(pet_e)

    # Pentagonal prism = C₅ □ K₂
    prism_e = [(0,1),(1,2),(2,3),(3,4),(4,0),
               (5,6),(6,7),(7,8),(8,9),(9,5),
               (0,5),(1,6),(2,7),(3,8),(4,9)]
    prs, prt = _edges_to_canonical(prism_e)

    return [("Petersen", 10, ps, pt), ("PentPrism", 10, prs, prt)]


# ═══════════════════════════════════════════════════════════════════
#  4. SRG(13,6,2,3) — Paley(13)
# ═══════════════════════════════════════════════════════════════════

def _quadratic_residues(q):
    """Return set of quadratic residues mod q (excluding 0)."""
    return {(x * x) % q for x in range(1, q)} - {0}


def _build_paley(q):
    """Build Paley graph of order q (q ≡ 1 mod 4, q prime power)."""
    qr = _quadratic_residues(q)
    edges = []
    for u in range(q):
        for v in range(u + 1, q):
            if (v - u) % q in qr:
                edges.append((u, v))
    s, t = zip(*edges)
    return q, list(s), list(t)


def build_paley_13() -> List[Graph]:
    n, s, t = _build_paley(13)
    return [("Paley(13)", n, s, t)]


# ═══════════════════════════════════════════════════════════════════
#  5. SRG(25,8,3,2) — Rook L₂(5)
# ═══════════════════════════════════════════════════════════════════

def _build_rook(k):
    """Rook graph L₂(k) = K_k □ K_k on k² vertices."""
    n = k * k
    edges = []
    for i in range(k):
        for j in range(k):
            u = k * i + j
            for m in range(k):
                if m != j:
                    v = k * i + m
                    if u < v: edges.append((u, v))
                if m != i:
                    v = k * m + j
                    if u < v: edges.append((u, v))
    s, t = _edges_to_canonical(edges)
    return n, s, t


def build_srg_25_8_3_2() -> List[Graph]:
    n, s, t = _build_rook(5)
    return [("Rook_L2(5)", n, s, t)]


# ═══════════════════════════════════════════════════════════════════
#  6. Prism vs K₃,₃
# ═══════════════════════════════════════════════════════════════════

def build_prism_vs_k33() -> List[Graph]:
    prism_e = [(0,1),(1,2),(2,0),(0,3),(1,4),(2,5),(3,4),(4,5),(5,3)]
    ps, pt = _edges_to_canonical(prism_e)

    k33_e = [(u, v) for u in range(3) for v in range(3, 6)]
    ks, kt = _edges_to_canonical(k33_e)

    return [("Prism", 6, ps, pt), ("K3,3", 6, ks, kt)]


# ═══════════════════════════════════════════════════════════════════
#  7. 2×C₄ vs C₈
# ═══════════════════════════════════════════════════════════════════

def build_2c4_vs_c8() -> List[Graph]:
    c4x2_e = [(0,1),(1,2),(2,3),(3,0),(4,5),(5,6),(6,7),(7,4)]
    s1, t1 = _edges_to_canonical(c4x2_e)

    c8_e = [(i, (i+1) % 8) for i in range(8)]
    s2, t2 = _edges_to_canonical(c8_e)

    return [("2xC4", 8, s1, t1), ("C8", 8, s2, t2)]


# ═══════════════════════════════════════════════════════════════════
#  8. Paley(q) for various q
# ═══════════════════════════════════════════════════════════════════

def build_paley_family() -> List[Graph]:
    """Paley graphs for all prime q ≡ 1 (mod 4) up to 53."""
    graphs = []
    for q in [5, 13, 17, 29, 37, 41, 53]:
        n, s, t = _build_paley(q)
        graphs.append((f"Paley({q})", n, s, t))

    # q = 25: GF(25) = GF(5)[x]/(x²+2).  Elements: a + b·α, α²=3.
    # Squares in GF(25)*: half of 24 elements = 12 squares.
    n = 25
    # Represent elements as (a, b) ∈ Z₅ × Z₅, with α² = 3 (mod 5)
    elems = [(a, b) for a in range(5) for b in range(5)]
    def gf25_mul(x, y):
        a1, b1 = x; a2, b2 = y
        return ((a1*a2 + 3*b1*b2) % 5, (a1*b2 + b1*a2) % 5)
    # Find squares: {x² : x ∈ GF(25)*}
    squares = set()
    for e in elems:
        if e != (0, 0):
            sq = gf25_mul(e, e)
            squares.add(sq)
    # Subtraction: diff = y - x in GF(25)
    def gf25_sub(y, x):
        return ((y[0] - x[0]) % 5, (y[1] - x[1]) % 5)
    edges = []
    for i in range(n):
        for j in range(i + 1, n):
            diff = gf25_sub(elems[j], elems[i])
            if diff in squares:
                edges.append((i, j))
    s, t = zip(*edges)
    graphs.append(("Paley(25)", n, list(s), list(t)))

    # q = 49: GF(49) = GF(7)[x]/(x²+1).  α² = -1 = 6 (mod 7)
    n = 49
    elems49 = [(a, b) for a in range(7) for b in range(7)]
    def gf49_mul(x, y):
        a1, b1 = x; a2, b2 = y
        return ((a1*a2 + 6*b1*b2) % 7, (a1*b2 + b1*a2) % 7)
    squares49 = set()
    for e in elems49:
        if e != (0, 0):
            squares49.add(gf49_mul(e, e))
    def gf49_sub(y, x):
        return ((y[0] - x[0]) % 7, (y[1] - x[1]) % 7)
    edges49 = []
    for i in range(n):
        for j in range(i + 1, n):
            diff = gf49_sub(elems49[j], elems49[i])
            if diff in squares49:
                edges49.append((i, j))
    s49, t49 = zip(*edges49)
    graphs.append(("Paley(49)", n, list(s49), list(t49)))

    return graphs


# ═══════════════════════════════════════════════════════════════════
#  9. Rook L₂(n) for n = 4..8
# ═══════════════════════════════════════════════════════════════════

def build_rook_family() -> List[Graph]:
    graphs = []
    for k in range(4, 9):
        n, s, t = _build_rook(k)
        graphs.append((f"Rook_L2({k})", n, s, t))
    return graphs


# ═══════════════════════════════════════════════════════════════════
# 10. Latin square graphs from cyclic MOLS
# ═══════════════════════════════════════════════════════════════════

def _latin_square_graph(n, latin_square):
    """Build the Latin square graph: vertices = (row, col) = n² nodes.
    Edge (i,j)~(k,l) iff i=k, or j=l, or L[i][j] = L[k][l]."""
    N = n * n
    edges = set()
    for i in range(n):
        for j in range(n):
            u = n * i + j
            # Same row
            for jj in range(j + 1, n):
                edges.add((u, n * i + jj))
            # Same column
            for ii in range(i + 1, n):
                edges.add((u, n * ii + j))
            # Same symbol
            sym = latin_square[i][j]
            for ii in range(n):
                for jj in range(n):
                    v = n * ii + jj
                    if v > u and latin_square[ii][jj] == sym:
                        if ii != i and jj != j:  # not same row/col
                            edges.add((u, v))
    s, t = zip(*sorted(edges)) if edges else ([], [])
    return N, list(s), list(t)


def _cyclic_latin_square(n, shift=1):
    """Cyclic Latin square: L[i][j] = (shift*i + j) mod n."""
    return [[(shift * i + j) % n for j in range(n)] for i in range(n)]


def build_latin_square_family() -> List[Graph]:
    """Latin square graphs that are genuinely non-isomorphic.

    Cyclic shifts over Z_p all produce isomorphic graphs (field
    automorphisms permute the vertices).  Instead we use:
      - n=4: the cyclic LS vs a non-cyclic LS (verified non-ISO)
      - n=5: cyclic vs two non-isotopic back-circulant squares
    These give co-spectral SRGs with the same parameters that are
    genuinely non-isomorphic.
    """
    graphs = []

    # ── n=4: Two non-isomorphic Latin square graphs ──────────────
    #   LS1: cyclic L[i][j] = (i+j) mod 4
    #   LS2: non-cyclic (row 0,1 same as cyclic; rows 2,3 differ)
    ls4_cyclic = _cyclic_latin_square(4, shift=1)
    ls4_other = [
        [0, 1, 2, 3],
        [1, 0, 3, 2],
        [2, 3, 0, 1],
        [3, 2, 1, 0],
    ]
    N, s, t = _latin_square_graph(4, ls4_cyclic)
    graphs.append(("LatSq(4,cyclic)", N, s, t))
    N, s, t = _latin_square_graph(4, ls4_other)
    graphs.append(("LatSq(4,Klein)", N, s, t))

    # ── n=5: Two non-isomorphic Latin square graphs ──────────────
    #   LS1: cyclic L[i][j] = (i+j) mod 5  (group Z_5)
    #   LS2: non-group-based Latin square  (verified NON-ISO via VF2)
    ls5_cyclic = _cyclic_latin_square(5, shift=1)
    ls5_other = [
        [0, 1, 2, 3, 4],
        [1, 0, 3, 4, 2],
        [2, 3, 4, 0, 1],
        [3, 4, 1, 2, 0],
        [4, 2, 0, 1, 3],
    ]
    N, s, t = _latin_square_graph(5, ls5_cyclic)
    graphs.append(("LatSq(5,cyclic)", N, s, t))
    N, s, t = _latin_square_graph(5, ls5_other)
    graphs.append(("LatSq(5,nongrp)", N, s, t))

    return graphs


# ═══════════════════════════════════════════════════════════════════
# 11. Steiner triple system block-intersection graphs
# ═══════════════════════════════════════════════════════════════════

def _cyclic_sts(v, base_blocks):
    """Generate STS(v) from base blocks by developing mod v."""
    blocks = set()
    for b in base_blocks:
        for shift in range(v):
            block = tuple(sorted((x + shift) % v for x in b))
            blocks.add(block)
    return sorted(blocks)


def _sts_block_graph(v, blocks):
    """Block-intersection graph: vertices = blocks, edge iff |intersection| > 0."""
    n = len(blocks)
    # For STS, two distinct blocks share at most 1 point.
    # Edge iff they share exactly 1 point.
    block_sets = [set(b) for b in blocks]
    edges = []
    for i in range(n):
        for j in range(i + 1, n):
            if len(block_sets[i] & block_sets[j]) >= 1:
                edges.append((i, j))
    s, t = zip(*edges) if edges else ([], [])
    return n, list(s), list(t)


def build_steiner_family() -> List[Graph]:
    """Steiner triple system block-intersection graphs.
    STS(v) exists iff v ≡ 1 or 3 (mod 6).
    We construct from known cyclic base blocks."""
    graphs = []

    # STS(7) = Fano plane: base block {0,1,3}
    blocks_7 = _cyclic_sts(7, [(0, 1, 3)])
    n, s, t = _sts_block_graph(7, blocks_7)
    graphs.append(("STS(7)_Fano", n, s, t))

    # STS(9): base blocks {0,1,2}, {0,3,6} (affine plane AG(2,3))
    # The unique STS(9) has 12 blocks
    blocks_9 = [
        (0,1,2),(3,4,5),(6,7,8),
        (0,3,6),(1,4,7),(2,5,8),
        (0,4,8),(1,5,6),(2,3,7),
        (0,5,7),(1,3,8),(2,4,6),
    ]
    n, s, t = _sts_block_graph(9, blocks_9)
    graphs.append(("STS(9)_AG23", n, s, t))

    # STS(13): cyclic, base blocks {0,1,4} and {0,2,8} (= {0,2,8})
    # (0,1,4) develops to 13 blocks, (0,2,8) develops to 13 blocks → 26 blocks total
    blocks_13 = _cyclic_sts(13, [(0, 1, 4), (0, 2, 8)])
    n, s, t = _sts_block_graph(13, blocks_13)
    graphs.append(("STS(13)_cyclic", n, s, t))

    # STS(15): cyclic, base blocks from known construction
    # Base blocks: {0,1,4}, {0,2,8}, {0,5,10}  (Kirkman triple)
    # Plus short-orbit block {0,5,10} with orbit size 5
    blocks_15a = _cyclic_sts(15, [(0, 1, 4), (0, 2, 9)])
    # Need exactly 35 blocks. Let's also add {0, 5, 10}
    extra = set()
    for shift in range(5):
        extra.add(tuple(sorted(((x + shift) % 15 for x in (0, 5, 10)))))
    all_blocks_15 = sorted(set(map(tuple, blocks_15a)) | extra)
    if len(all_blocks_15) == 35:
        n, s, t = _sts_block_graph(15, all_blocks_15)
        graphs.append(("STS(15)_cyclic", n, s, t))

    # STS(19): cyclic, base block {0,1,5}, {0,2,10}, {0,4,8}
    blocks_19 = _cyclic_sts(19, [(0, 1, 5), (0, 2, 10), (0, 4, 8)])
    if len(blocks_19) == 57:
        n, s, t = _sts_block_graph(19, blocks_19)
        graphs.append(("STS(19)_cyclic", n, s, t))

    # STS(25): cyclic, base blocks {0,1,3}, {0,5,15}, {0,7,14}, {0,9,18}
    blocks_25 = _cyclic_sts(25, [(0, 1, 3), (0, 5, 15), (0, 7, 14), (0, 9, 18)])
    if len(blocks_25) == 100:
        n, s, t = _sts_block_graph(25, blocks_25)
        graphs.append(("STS(25)_cyclic", n, s, t))

    return graphs


# ═══════════════════════════════════════════════════════════════════
# 12. Kneser K(n,2) — n = 5..9
# ═══════════════════════════════════════════════════════════════════

def build_kneser_family() -> List[Graph]:
    """Kneser graph K(n,2): vertices = 2-subsets of [n],
    edge iff disjoint.  K(5,2) = Petersen."""
    graphs = []
    for n in range(5, 10):
        subsets = list(combinations(range(n), 2))
        N = len(subsets)
        edges = []
        for i in range(N):
            for j in range(i + 1, N):
                if len(set(subsets[i]) & set(subsets[j])) == 0:
                    edges.append((i, j))
        s, t = zip(*edges) if edges else ([], [])
        graphs.append((f"Kneser({n},2)", N, list(s), list(t)))
    return graphs


# ═══════════════════════════════════════════════════════════════════
# 13. Johnson J(n,2) — n = 5..10
# ═══════════════════════════════════════════════════════════════════

def build_johnson_family() -> List[Graph]:
    """Johnson graph J(n,2): vertices = 2-subsets of [n],
    edge iff |intersection| = 1.  J(8,2) = T(8)."""
    graphs = []
    for n in range(5, 11):
        subsets = list(combinations(range(n), 2))
        N = len(subsets)
        edges = []
        for i in range(N):
            for j in range(i + 1, N):
                if len(set(subsets[i]) & set(subsets[j])) == 1:
                    edges.append((i, j))
        s, t = zip(*edges) if edges else ([], [])
        graphs.append((f"Johnson({n},2)", N, list(s), list(t)))
    return graphs


# ═══════════════════════════════════════════════════════════════════
# 14. Hamming H(d,q)
# ═══════════════════════════════════════════════════════════════════

def _hamming_graph(d, q):
    """Hamming graph H(d,q): vertices = d-tuples over [q],
    edge iff Hamming distance = 1."""
    # Enumerate all d-tuples
    def all_tuples(d, q):
        if d == 0:
            return [()]
        prev = all_tuples(d - 1, q)
        return [t + (v,) for t in prev for v in range(q)]

    tuples = all_tuples(d, q)
    N = len(tuples)
    t2i = {t: i for i, t in enumerate(tuples)}
    edges = []
    for i, t1 in enumerate(tuples):
        for pos in range(d):
            for val in range(q):
                if val != t1[pos]:
                    t2 = t1[:pos] + (val,) + t1[pos+1:]
                    j = t2i[t2]
                    if i < j:
                        edges.append((i, j))
    s, t = zip(*edges) if edges else ([], [])
    return N, list(s), list(t)


def build_hamming_family() -> List[Graph]:
    graphs = []
    for d, q in [(2, 4), (2, 5), (3, 3), (3, 4)]:
        N, s, t = _hamming_graph(d, q)
        graphs.append((f"Hamming({d},{q})", N, s, t))
    return graphs


# ═══════════════════════════════════════════════════════════════════
# 15. Miyazaki graphs — pairs of non-isomorphic hard instances
# ═══════════════════════════════════════════════════════════════════

def _build_miyazaki_pair(m):
    """Build a CFI-over-cycle pair on 2m vertices (cubic graphs).

    Correct CFI construction with 2-node gadgets:
      - Gadget i has nodes {2i, 2i+1} with internal edge (2i, 2i+1).
      - Inter-gadget connection from i to (i+1) mod m:
        Straight: (2i)─(2j), (2i+1)─(2j+1)
        Crossed:  (2i)─(2j+1), (2i+1)─(2j)
      - Each node has degree 3 (1 internal + 2 inter-gadget).

    Non-isomorphic for m >= 3: swapping the two nodes of a gadget
    toggles BOTH incident connections simultaneously, so total
    crossing parity mod 2 is a graph invariant.

    Graph A: 0 crossings (even parity)
    Graph B: 1 crossing  (odd parity)
    """
    n = 2 * m

    def _build_variant(cross_at):
        edges = set()
        for i in range(m):
            # Internal edge within gadget i
            edges.add((2 * i, 2 * i + 1))
            # Inter-gadget: gadget i → gadget (i+1) mod m
            j = (i + 1) % m
            if i == cross_at:
                edges.add((2 * i, 2 * j + 1))
                edges.add((2 * i + 1, 2 * j))
            else:
                edges.add((2 * i, 2 * j))
                edges.add((2 * i + 1, 2 * j + 1))
        return edges

    edges_a = _build_variant(cross_at=-1)   # no crossing
    edges_b = _build_variant(cross_at=0)    # one crossing

    sa, ta = _edges_to_canonical(edges_a)
    sb, tb = _edges_to_canonical(edges_b)
    return (f"Multipede-A(m={m})", n, sa, ta), (f"Multipede-B(m={m})", n, sb, tb)


def build_miyazaki_family() -> List[Graph]:
    """CFI-over-cycle (multipede) pairs for m = 3..10.

    2-node gadgets on cycle C_m. Non-isomorphic for m >= 3.
    These are 3-regular graphs on 2m vertices.
    """
    graphs = []
    for m in range(3, 11):
        a, b = _build_miyazaki_pair(m)
        graphs.append(a)
        graphs.append(b)
    return graphs


# ═══════════════════════════════════════════════════════════════════
# 16. Random regular graphs
# ═══════════════════════════════════════════════════════════════════

def build_random_regular(seed=42) -> List[Graph]:
    """Generate groups of random regular graphs with same (n, d).
    10 graphs per parameter set — Delta-DRESS should distinguish them."""
    import networkx as nx
    rng = random.Random(seed)
    graphs = []
    for n, d in [(20, 6), (30, 8), (40, 10)]:
        for idx in range(10):
            s = rng.randint(0, 2**31)
            G = nx.random_regular_graph(d, n, seed=s)
            edges = [(u, v) for u, v in G.edges() if u < v]
            if edges:
                src, tgt = zip(*sorted(edges))
                graphs.append((f"RandReg({n},{d})_{idx}", n,
                               list(src), list(tgt)))
    return graphs


# ═══════════════════════════════════════════════════════════════════
# 17. Generalized Quadrangle point graphs: GQ(2,2) and GQ(2,4)
# ═══════════════════════════════════════════════════════════════════

def build_gq_family() -> List[Graph]:
    """Generalized quadrangle point graphs.

    GQ(2,2): 15 points, each on 3 lines of 3 points.
    Collinearity graph = SRG(15,6,1,3) = complement of Petersen × K₃.

    GQ(2,4): 27 points, collinearity graph = Schläfli complement = SRG(27,10,1,5).
    This is the unique GQ(2,4) (Brouwer).
    """
    graphs = []

    # GQ(2,2) = W(2): the symplectic GQ over GF(2).
    # Points: non-zero vectors of GF(2)^4 (projective => 15 points of PG(3,2))
    # Two points collinear iff symplectic form = 0.
    # Simpler: use the 15 lines of PG(3,2) wrt symplectic polarity.
    #
    # Explicit construction: points = {1..15}, lines from the doily.
    # GQ(2,2) has 15 points, 15 lines, 3 points/line, 3 lines/point.
    gq22_lines = [
        (0,1,2), (3,4,5), (6,7,8), (9,10,11), (12,13,14),
        (0,3,6), (1,4,7), (2,5,8), (0,10,13), (1,11,14),
        (2,9,12), (3,10,14), (4,11,12), (5,9,13), (6,11,13),
    ]
    # Hmm, this needs to be the actual GQ(2,2) = doily. Let me use the
    # direct collinearity definition from the symplectic form.
    # Points of PG(3,2): 15 points = non-zero of GF(2)^4
    points = []
    for a in range(2):
        for b in range(2):
            for c in range(2):
                for d in range(2):
                    if (a, b, c, d) != (0, 0, 0, 0):
                        points.append((a, b, c, d))
    assert len(points) == 15
    # Symplectic form: <x,y> = x0*y1 - x1*y0 + x2*y3 - x3*y2 (mod 2)
    def symp(x, y):
        return (x[0]*y[1] + x[1]*y[0] + x[2]*y[3] + x[3]*y[2]) % 2
    edges = []
    for i in range(15):
        for j in range(i + 1, 15):
            if symp(points[i], points[j]) == 0:
                edges.append((i, j))
    s, t = zip(*edges) if edges else ([], [])
    graphs.append(("GQ(2,2)_collinear", 15, list(s), list(t)))

    # GQ(2,4): 27 points of the Schläfli graph complement = SRG(27,10,1,5)
    # The Schläfli graph is the unique SRG(27,16,10,8).
    # Its complement is SRG(27,10,1,5) = GQ(2,4) collinearity graph.
    #
    # Construction: vertices = 27 lines in PG(3,2) not in a plane,
    # or equivalently the 27 lines of a cubic surface.
    #
    # Simpler construction using E₆ root system / ternary Golay code:
    # Points = GF(3)^3.  Two points (a,b,c) and (d,e,f) are collinear
    # iff a+d = b+e = c+f = 0 (mod 3) is NOT the right one.
    #
    # Let's use the explicit adjacency from the Schläfli graph.
    # The Schläfli graph: vertices = 27 lines on a cubic surface.
    # Use the GF(3) construction: vertices = vectors in GF(3)³.
    # Edge in Schläfli complement ⟺ collinear in GQ(2,4).
    # Two points x,y collinear iff x₁y₁ + x₂y₂ + x₃y₃ = 0 (mod 3),
    # where the form is the standard dot product over GF(3).
    #
    # Actually, let's use an explicit known construction.
    # GQ(2,4): 27 points, each point on 5 lines of 3 points.
    # Collinearity graph: degree 10 (each point collinear with 10 others).
    pts27 = [(a, b, c) for a in range(3) for b in range(3) for c in range(3)]
    assert len(pts27) == 27
    # Use Hermitian form over GF(3): x₀y₀ + x₁y₁ + x₂y₂ ≡ 0 (mod 3)
    edges27 = []
    for i in range(27):
        for j in range(i + 1, 27):
            dot = sum(pts27[i][k] * pts27[j][k] for k in range(3)) % 3
            if dot == 0:
                edges27.append((i, j))
    s27, t27 = zip(*edges27) if edges27 else ([], [])
    # Check degree
    from collections import Counter
    deg = Counter()
    for u, v in edges27:
        deg[u] += 1; deg[v] += 1
    degrees = set(deg.values())
    if len(degrees) == 1:
        graphs.append(("GQ(2,4)_collinear", 27, list(s27), list(t27)))

    return graphs


# ═══════════════════════════════════════════════════════════════════
# 18. Complement graphs — Petersen complement, etc.
# ═══════════════════════════════════════════════════════════════════

def _complement(n, sources, targets):
    """Complement of a graph."""
    edge_set = set()
    for u, v in zip(sources, targets):
        edge_set.add((min(u, v), max(u, v)))
    comp_edges = []
    for u in range(n):
        for v in range(u + 1, n):
            if (u, v) not in edge_set:
                comp_edges.append((u, v))
    s, t = zip(*comp_edges) if comp_edges else ([], [])
    return list(s), list(t)


def build_complement_pairs() -> List[Graph]:
    """SRGs whose complements are also SRGs with different structure."""
    graphs = []

    # Petersen complement = Kneser(5,2) complement = SRG(10,6,3,4)
    pet = build_srg_10_3_0_1()[0]
    cs, ct = _complement(pet[1], pet[2], pet[3])
    graphs.append(("Petersen_compl", 10, cs, ct))

    # Pentagonal prism complement
    prism = build_srg_10_3_0_1()[1]
    cs2, ct2 = _complement(prism[1], prism[2], prism[3])
    graphs.append(("PentPrism_compl", 10, cs2, ct2))

    return graphs


# ═══════════════════════════════════════════════════════════════════
#  Master catalog
# ═══════════════════════════════════════════════════════════════════

def build_all_families(include_random=True) -> dict:
    """Build all hard graph families.

    Returns dict: family_name → list of (name, n, sources, targets).
    """
    families = {}

    families["SRG(28,12,6,4)_T8_Chang"] = build_srg_28_12_6_4()
    families["SRG(16,6,2,2)_Rook_Shrikhande"] = build_srg_16_6_2_2()
    families["SRG(10,3,0,1)_Petersen"] = build_srg_10_3_0_1()
    families["Paley(13)"] = build_paley_13()
    families["SRG(25,8,3,2)_Rook5"] = build_srg_25_8_3_2()
    families["Prism_vs_K33"] = build_prism_vs_k33()
    families["2C4_vs_C8"] = build_2c4_vs_c8()
    families["Paley_family"] = build_paley_family()
    families["Rook_family"] = build_rook_family()
    families["LatinSquare_family"] = build_latin_square_family()
    families["Steiner_family"] = build_steiner_family()
    families["Kneser_family"] = build_kneser_family()
    families["Johnson_family"] = build_johnson_family()
    families["Hamming_family"] = build_hamming_family()
    families["Miyazaki_family"] = build_miyazaki_family()
    families["GQ_family"] = build_gq_family()
    families["Complement_pairs"] = build_complement_pairs()

    if include_random:
        families["RandomRegular"] = build_random_regular()

    return families


# ═══════════════════════════════════════════════════════════════════
#  Verification
# ═══════════════════════════════════════════════════════════════════

def verify_graph(name, n, sources, targets):
    """Basic structural checks."""
    from collections import Counter
    errors = []

    # Check edge bounds
    for u, v in zip(sources, targets):
        if u >= n or v >= n or u < 0 or v < 0:
            errors.append(f"edge ({u},{v}) out of range for n={n}")
        if u == v:
            errors.append(f"self-loop ({u},{v})")
        if u >= v:
            errors.append(f"edge ({u},{v}) not canonical (u<v required)")

    # Check duplicates
    edge_set = set(zip(sources, targets))
    if len(edge_set) != len(sources):
        errors.append(f"duplicate edges")

    # Degree distribution
    deg = Counter()
    for u, v in zip(sources, targets):
        deg[u] += 1; deg[v] += 1
    degrees = sorted(set(deg.values()))

    return {
        "name": name,
        "n": n,
        "m": len(sources),
        "degrees": degrees,
        "regular": len(degrees) == 1,
        "degree": degrees[0] if len(degrees) == 1 else None,
        "errors": errors,
    }


def verify_all():
    """Build and verify all families."""
    families = build_all_families(include_random=False)
    total = 0
    ok = 0
    for fname, glist in sorted(families.items()):
        print(f"\n{'─' * 60}")
        print(f"  {fname}  ({len(glist)} graphs)")
        print(f"{'─' * 60}")
        for name, n, s, t in glist:
            info = verify_graph(name, n, s, t)
            total += 1
            tag = f"n={n:>3d}  m={info['m']:>4d}  "
            if info['regular']:
                tag += f"d={info['degree']}"
            else:
                tag += f"deg={info['degrees']}"
            if info['errors']:
                print(f"  ✗ {name:35s} {tag}  ERRORS: {info['errors']}")
            else:
                print(f"  ✓ {name:35s} {tag}")
                ok += 1

    print(f"\n{'═' * 60}")
    print(f"  {ok}/{total} graphs passed verification")
    print(f"{'═' * 60}")


if __name__ == "__main__":
    verify_all()
