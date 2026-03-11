# Isomorphism experiments

Evaluating the distinguishing power of Delta^k-DRESS on hard
graph-isomorphism benchmarks: 51,718 strongly regular graphs across
16 parameter families plus 18 constructed hard families (102 graphs),
achieving 100% within-family separation on 576,107,835 non-isomorphic pairs.

## Scripts

- `hard_families.py`: constructs all 18 hard families programmatically
- `destroy_hard_families.py`: runs Delta^k-DRESS on every constructed
  family and every SRG file on disk, with checkpoint/resume
- `cfi.py`: runs Delta^k-DRESS on CFI graph pairs (`--k N`, `--only NAME`,
  `--max-n N`)

## Data sources

All SRG datasets live under `datasets/srgs/`.

### Spence SRG collection (12 families, 43,703 graphs)

Definitive enumeration of strongly regular graphs on up to 64 vertices,
compiled by Edward Spence.

Source: <https://www.maths.gla.ac.uk/~es/srgraphs.php>

| File | Parameters | Graphs | Pairs |
|------|-----------|--------|-------|
| `25-12-5-6` | SRG(25,12,5,6) | 15 | 105 |
| `26-10-3-4` | SRG(26,10,3,4) | 10 | 45 |
| `28-12-6-4` | SRG(28,12,6,4) | 4 | 6 |
| `29-14-6-7` | SRG(29,14,6,7) | 41 | 820 |
| `35-18-9-9.bz2` | SRG(35,18,9,9) | 3,854 | 7,424,731 |
| `36-14-4-6` | SRG(36,14,4,6) | 180 | 16,110 |
| `36-15-6-6.bz2` | SRG(36,15,6,6) | 32,548 | 529,669,878 |
| `37-18-8-9.bz2` | SRG(37,18,8,9) | 6,760 | 22,845,420 |
| `40-12-2-4` | SRG(40,12,2,4) | 28 | 378 |
| `45-12-3-3` | SRG(45,12,3,3) | 78 | 3,003 |
| `50-21-8-9` | SRG(50,21,8,9) | 18 | 153 |
| `64-18-2-6` | SRG(64,18,2,6) | 167 | 13,861 |

Formats: Spence adjacency-matrix text (one `n x n` binary matrix per
graph, separated by blank lines), optionally bz2-compressed.

### Additional SRG families (4 families, 8,015 graphs)

From McKay's online graph collections and combinatorial design
catalogues.

Source: <https://users.cecs.anu.edu.au/~bdm/data/graphs.html>

| File | Parameters | Graphs | Pairs |
|------|-----------|--------|-------|
| `SRG45-22-10-11.txt` | SRG(45,22,10,11) | 6 | 15 |
| `SRG63-32-16-26-S-2-4-28.txt` | SRG(63,32,16,26)-S | 4,466 | 9,970,345 |
| `SRG63-32-16-26-quasi.txt` | SRG(63,32,16,26)-Q | 3,511 | 6,161,805 |
| `sr65321516some.g6` | SRG(65,32,15,16) | 32 | 496 |

Formats: graph6 (`.g6`) or one adjacency matrix per graph (`.txt`).

### Constructed hard families (18 families, 102 graphs)

Generated programmatically by `hard_families.py`.
No external data files required.

| Family | Graphs | Pairs | Description |
|--------|--------|-------|-------------|
| SRG(28,12,6,4) | 4 | 6 | T(8) + 3 Chang graphs |
| SRG(16,6,2,2) | 2 | 1 | Rook L2(4) + Shrikhande |
| SRG(10,3,0,1) | 2 | 1 | Petersen + pentagonal prism |
| Paley(13) | 1 | 0 | Paley graph of order 13 |
| SRG(25,8,3,2) | 1 | 0 | Rook L2(5) |
| Prism vs K3,3 | 2 | 1 | 3-regular on 6 vertices |
| 2xC4 vs C8 | 2 | 1 | 2-regular on 8 vertices |
| Paley family | 9 | 36 | Paley(q) for q = 5..53 |
| Rook family | 5 | 10 | Rook L2(n) for n = 4..8 |
| Latin square | 4 | 6 | From cyclic MOLS, n = 4, 5, 7 |
| Steiner | 6 | 15 | S(2,3,v) from cyclic difference triples |
| Kneser | 5 | 10 | K(n,2) for n = 5..9 |
| Johnson | 6 | 15 | J(n,2) for n = 5..10 |
| Hamming | 4 | 6 | H(d,q): (2,4), (2,5), (3,3), (3,4) |
| Miyazaki | 16 | 120 | CFI-over-cycle, m = 2..6 |
| GQ(2,2) | 1 | 0 | Generalized quadrangle point graph |
| Complement pairs | 2 | 1 | Graph + complement pair |
| Random regular | 30 | 435 | 10 graphs each at 3 (n,d) settings |

## CFI staircase

The Cai-Furer-Immerman (CFI) construction produces the canonical hard
instances for the WL hierarchy: distinguishing CFI(K_n) from CFI'(K_n)
requires at least (n-1)-WL. We tested Delta^k-DRESS for k = 0, 1, 2, 3.

Results from the [k-DRESS paper](https://github.com/velicast/dress-graph/blob/main/research/k-DRESS.tex):

| Base | Vertices | WL req. | Delta^0 | Delta^1 | Delta^2 | Delta^3 |
|------|----------|---------|---------|---------|---------|---------|
| K_3  | 6   | 2-WL    | Y       | Y       | Y       | Y       |
| K_4  | 16  | 3-WL    | N       | Y       | Y       | Y       |
| K_5  | 40  | 4-WL    | N       | N       | Y       | Y       |
| K_6  | 96  | 5-WL    | N       | N       | N       | Y       |
| K_7  | 224 | 6-WL    | N       | N       | N       | N       |

Each deletion level adds exactly one WL dimension of expressiveness:
Delta^k-DRESS distinguishes CFI(K_{k+3}) (requiring (k+2)-WL) and fails
on CFI(K_{k+4}) (requiring (k+3)-WL), empirically matching the (k+2)-WL
boundary.

**DRESS-WL Dominance Conjecture:** Delta^k-DRESS >= (k+2)-WL for all
k >= 0. The base case (k = 0) is proved; the inductive step (k >= 1)
remains open.

### CFI data

Edge-list pairs live under `datasets/cfi/`. Each base graph has two
files (`<name>1.txt` and `<name>2.txt`):

| Base graph | Files | Vertices |
|------------|-------|----------|
| triangle   | `triangle1.txt`, `triangle2.txt`   | 6   |
| 4-clique   | `4-clique1.txt`, `4-clique2.txt`   | 16  |
| 5-clique   | `5-clique1.txt`, `5-clique2.txt`   | 40  |
| 6-clique   | `6-clique1.txt`, `6-clique2.txt`   | 96  |
| 7-clique   | `7-clique1.txt`, `7-clique2.txt`   | 224 |
| 8-clique   | `8-clique1.txt`, `8-clique2.txt`   | 512 |
| 9-clique   | `9-clique1.txt`, `9-clique2.txt`   | 1152 |
| 10-clique  | `10-clique1.txt`, `10-clique2.txt` | 2560 |
| frucht     | `frucht1.txt`, `frucht2.txt`       | 24  |
