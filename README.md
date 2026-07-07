# Micro-Bump & C4-Bump Placement and Routing Benchmark Suite

A synthetic benchmark toolkit for micro-bump and C4-bump placement and routing optimization problems in 2.5D/3D IC designs.
The suite consists of three independent modules:

| Module | Script | Purpose |
|---|---|---|
| **Generator** | `generator.py` | Produce synthetic micro-bump placement instances |
| **Validator** | `validator.py` | Check routing output for placement & routing feasibility |
| **Visualizer** | `visualizer.py` | Standalone routing visualization (per-layer / 3D / projection) |

---

## Requirements

```bash
pip install pyyaml matplotlib numpy
```

Python 3.10 or later is recommended.

---

## Repository layout

```
.
├── generator.py          # Benchmark instance generator
├── validator.py          # Placement + routing feasibility checker & metrics
├── visualizer.py         # Standalone routing visualizer
├── input.yaml            # Shared configuration (generator, validator, visualizer all read this)
├── requirements.txt
├── outputs/              # Default output directory (generated, not version-controlled)
└── benchmark datasets/
    ├── data/             # Problem definitions (input configs + generated layouts)
    │   ├── small/        # N ≈ 100
    │   │   └── instance_small_{1..5}/   # input_small_N.yaml + micro_coordinate + C4_candidate
    │   ├── medium/       # N ≈ 556
    │   │   └── instance_medium_{1..5}/
    │   └── large/        # N ≈ 902
    │       └── instance_large_{1..5}/   # 1-3: solvable · 4-5: intentionally unsolvable
    └── results/          # Algorithm outputs (routing solutions + metrics)
        ├── small/
        │   └── instance_small_{1..5}/   # summary.json + routing_result.json + plots + metrics
        ├── medium/
        │   └── instance_medium_{1..5}/
        └── large/
            └── instance_large_{1..3}/   # FEASIBLE results only
```

---

## 1. Generator (`generator.py`)

Generates clustered micro-bump layouts that reflect structural patterns commonly observed in practical 2.5D/3D IC designs.

### Usage

```bash
python generator.py -c input.yaml
```

### Optional arguments

| Argument | Default | Description |
|---|---|---|
| `--num-partitions` | `4` | Number of vertical partitions |
| `--sub-width-pitch` | `8` | Sub-partition strip width in pitch units |
| `--cluster-gap-x` | *(yaml)* | Minimum horizontal gap between clusters (μm) — overrides `generator.g_x` in yaml |
| `--cluster-gap-y` | *(yaml)* | Minimum vertical gap between clusters (μm) — overrides `generator.g_y` in yaml |
| `--cluster-height-slack` | `1.15` | Height slack factor when estimating cluster bounding boxes |
| `--mix-ratio` | `0.15` | Fraction of top-group signals that bleed into adjacent groups' partitions |
| `--mix-top-k` | `2` | Number of largest groups to apply cross-partition mixing |
| `--cluster-shuffle-ratio` | `0.20` | Fraction of all clusters randomly shuffled post-placement |
| `--spatial-mode` | `random` | Partition allocation strategy: `random` or `structured` (see below) |
| `--num-instances` | `1` | Number of instances to generate with consecutive seeds |
| `--no-plot` | — | Skip visualization |

> **Note:** `g_x` and `g_y` are read from `input.yaml` (`generator.g_x` / `generator.g_y`) by default.
> The CLI flags `--cluster-gap-x` / `--cluster-gap-y` override the yaml values when explicitly provided.

Example:

```bash
python generator.py -c input.yaml --no-plot

# Structured T-shape layout (matches real IC spatial patterns):
python generator.py -c input.yaml --spatial-mode structured --no-plot

# Tighter vertical packing:
python generator.py -c input.yaml --cluster-gap-y 0
```

### Output

All files are written to the directory specified by `output.dir` in the config (default: `outputs/`).

| File | Description |
|---|---|
| `micro_coordinate.csv` | Bump coordinates: `group`, `micro_id`, `micro_x`, `micro_y` |
| `micro_coordinate.png` | Scatter plot, color-coded by group; signal and dummy bumps drawn at the same size |
| `C4_candidate.csv` | Package-bump candidate grid: `candidate_id`, `C4_x`, `C4_y` |
| `C4_candidate.png` | Scatter plot of the C4 bump candidate grid |

`micro_coordinate.csv` contains both **signal bumps** (group = `G1`…`G5`) and **dummy bumps** (group = `dummy`, id = `null_XXXXX`). Dummy bumps fill every grid position not occupied by a signal bump, including the 5% outer margin.

## Benchmark Instances & Results

Instances are organized into three size tiers (N ≈ 100 / 556 / 902), each with 5 instances varying in group count K, layout shape W×H, dummy bump density φ, cluster size distribution, and routing constraints (σ, ρ). Large instances 4–5 are intentionally unsolvable under the given constraints and serve as negative-case benchmarks.

The routing results in `benchmark datasets/results/` were produced by the two-stage placement-and-routing algorithm from:

> Y. Kim, S. Han, B.I. Kim, D.G. Choi, H. Kim, J. Kim, **"A two-stage placement and routing framework for HBM interposer design"**, *Submitted*.

These results are provided as a reference baseline — users developing their own optimization methods should treat the `input_*.yaml` configs as the actual benchmark problems to solve.

### Summary

Full results are available in [results_summary.md](benchmark%20datasets/results/results_summary.md).

---

## 2. Validator (`validator.py`)

Validates a routing output JSON file for placement and routing feasibility, computes routing metrics, and saves visualization plots.

### Usage

```bash
python validator.py -r <routing_output.json>
```

The validator **auto-derives** all other paths from the routing file's location:

| Derived path | Description |
|---|---|
| `../input.yaml` | Configuration (one level above the output dir) |
| `./C4_candidate.csv` | Package-bump candidate grid (same output dir) |
| `./micro_coordinate.csv` | Micro-bump placement (same output dir) |
| `./routing_metrics.csv` | Per-net metrics output (written here) |
| `./summary.json` | All placement/routing violations, grouped by category (written here) |

### Routing JSON format

The routing output must be a JSON array. Each entry represents one net:

```json
[
  {
    "netname": "G1_0001",
    "m1": [[x1, y1], [x2, y2]],
    "m2": [[x2, y2], [x3, y3]],
    "m3": [[x3, y3], [x4, y4]],
    "m4": [[x4, y4], [x5, y5]]
  }
]
```

Rules:
- All `L` layers (`m1`…`mL`) must be present for every net.
- Each layer must have **≥ 2 waypoints**.
- Last point of `mN` must equal first point of `m(N+1)` (via junction).
- `m1[0]` = C4 bump position (bottom layer); `mL[-1]` = micro-bump position (top layer).
- Waypoints only appear at **turn points** (no redundant intermediate waypoints).

### Checks performed

**Placement feasibility:**

| Check | Description |
|---|---|
| micro match | Every net in `micro_coordinate.csv` appears in routing output with matching coordinates, and vice versa |
| Layout bounds | All components (micro, via, C4) lie within `[0, width] × [0, height]` |
| Bump candidate | Every C4 bump must be located on a predefined candidate grid point |
| Minimum spacing | Component-to-component center-to-center distance checks (micro↔via, via↔via, C4↔C4, etc.) |
| Window density | Number of C4 bumps within any `q × q` area must not exceed `placement.p` |

**Routing feasibility:**

| Check | Description |
|---|---|
| Connectivity | Route must be unbroken from C4 bump (`m1[0]`) to micro (`mL[-1]`) across all layers |
| Bounds | All routing waypoints lie within layout bounds |
| Turn angle | Every bend must be exactly ±45° (interior angle 135° or 225°); 90°, 180°, U-turns are forbidden |
| Min advance | Each segment must span ≥ 1 full grid step before any direction change |
| Octagonal routing | All segments must be horizontal, vertical, or 45° diagonal |
| Self-crossing | No wire may cross itself on the same layer (same net) |
| Routing spacing | Wire-to-micro, wire-to-via, wire-to-C4, and wire-to-wire center-to-center distance checks (grid-based occupancy check; dummy bumps also counted as obstacles) |
| Layer usage ratio | Per-layer routing length fraction ≤ `routing.rho` (0.0 = unconstrained) |
| Group deviation | Within each group, `(max_length − min_length) / min_length ≤ routing.sigma` |

**Routing metrics** (saved to `routing_metrics.csv`):

| Column | Description |
|---|---|
| `netname` | Net identifier (e.g. `G1_0001`) |
| `routing_length` | Total Euclidean wire length across all layers |
| `bend_count` | Number of valid ±45° direction changes across all layers |
| `m1_length` … `mL_length` | Wire length on each individual layer |
| `TOTAL` | Sum across all nets |
| `GROUP_G{k}` | Total wire length for group Gk |
| `LAYER_M{l}` | Total wire length on layer Ml |

**Visualization** (PNG files saved alongside the routing JSON):

| File | Description |
|---|---|
| `<base>_layer{N}.png` | Per-layer routing plot, colored by signal group |
| `<base>_3d.png` | 3D multi-layer view with layer planes and via connections |
| `<base>_projection.png` | Top-down projection of all layers, colored by layer |

---

## 3. Visualizer (`visualizer.py`)

Standalone visualization for routing output JSON. Produces the same three plot types as the validator's built-in plots, but can target any output directory.

### Usage

```bash
python visualizer.py -c input.yaml -r routing_output.json
python visualizer.py -c input.yaml -r routing_output.json -o my_plots/
```

### Arguments

| Argument | Description |
|---|---|
| `-c`, `--config` | Path to `input.yaml` |
| `-r`, `--routing` | Path to routing output JSON |
| `-o`, `--outdir` | Output directory for plots (default: same directory as routing JSON) |

### Output

| File | Description |
|---|---|
| `<base>_layer{N}.png` | Routing on layer N, colored by signal group |
| `<base>_3d.png` | 3D multi-layer view |
| `<base>_projection.png` | Top-down projection (all layers overlaid, colored by layer) |

---

## Configuration (`input.yaml`)

All three modules share the same `input.yaml`. Key parameters:

### `generator`

| Parameter | Description |
|---|---|
| `seed` | Random seed for reproducibility |
| `g_x` | $g_x$: minimum horizontal gap between micro bump clusters (μm) |
| `g_y` | $g_y$: minimum vertical gap between micro bump clusters (μm) |

### `layout`

| Parameter | Description |
|---|---|
| `W`, `H` | $W, H$: layout boundary |
| `L` | $L$: number of layers |

### `spec`

| Parameter | Description |
|---|---|
| `delta` | $\Delta$: unit grid size (equal to wire width) |
| `d_micro`, `d_C4`, `d_via` | $d_i$: diameter of component $i$ where $i \in$ {micro bump, C4 bump, via} |
| `sc.micro`, `sc.C4` | $sc_i$: minimum spacing between bumps of type $i$ where $i \in$ {micro bump, C4 bump} |
| `sc_via.*` | $sc_{\text{via},i}$: minimum spacing between via and element $i$ where $i \in$ {micro bump, C4 bump, via} |
| `sc_wire.*` | $sc_{\text{wire},i}$: minimum spacing between wire and component $i$ where $i \in$ {micro bump, C4 bump, via, wire} |

### `signal`

| Parameter | Description |
|---|---|
| `N` | $N$: number of nets |
| `K` | $K$: number of groups |
| `G1` … `GK` | $G_k$: number of nets in group $k$ ($k = 1, \ldots, K$) |
| `c_min` | $c_\min$: lower bound of cluster size as a fraction of total signal count |
| `c_max` | $c_\max$: upper bound of cluster size as a fraction of total signal count |

### `placement`

| Parameter | Description |
|---|---|
| `p`, `q` | $p, q$: at most $p$ C4 bumps in any $q \times q$ density-checking window of the C4 bump candidate grid |
| `phi` | $\phi$: fraction of residual ML-layer grid positions (not occupied by signal micro bumps) filled with dummy bumps as routing obstacles (`1.0` = fully filled; `0.0` = none) |

### `routing`

| Parameter | Description |
|---|---|
| `rho` | $\rho_l$: maximum ratio of routing length assigned to layer $l$ ($l = 1, \ldots, L$) |
| `sigma` | $\sigma_k$: routing-length skew for group $k$ |

See [`input.yaml`](input.yaml) for a fully annotated example.

---

## Placement Algorithm (generator)

Micro-bump coordinates are generated through the following pipeline:

### Step 1 — 5% margin

A fixed 5% margin is applied on all four sides. Signal bumps are placed only within the inner 90% of the layout. Dummy bumps fill the entire grid including the outer margin.

```
Full layout (width × height)
  └─ Signal placement region: 90% × 90% (center, snapped to pitch grid)
       └─ Candidate grid: xs_u[], ys_u[]  (pitch-spaced lattice)
```

### Step 2 — Vertical partitioning

The candidate grid is divided into `num_partitions` equal vertical strips. Each strip spans the full usable height.

### Step 3 — Group-to-partition allocation (`--spatial-mode`)

**`random` (default)**
Groups are assigned to partitions probabilistically in descending size order:

- The number of partitions a group occupies scales with its share of total signals (< 25% → 1 partition, 25–50% → 2, 50–75% → 3, ≥ 75% → 4).
- A "coverage pass" first assigns each group to an unoccupied partition; once all partitions are covered, remaining groups are placed randomly.
- When exactly 2 partitions are chosen, non-adjacent partitions (`|p1 − p2| > 1`) are preferred to maximise spatial separation.

**`structured`**
A fixed template that mirrors the spatial pattern found in real IC layouts:

| Rank | Group | Partition assignment |
|---|---|---|
| 0 (largest, e.g. G2) | Background | All partitions — 60% to inner, 40% to outer |
| 1 (2nd largest, e.g. G1) | Peripheral | Leftmost + rightmost partitions only (equal split) |
| 2 (3rd largest, e.g. G3) | Inner | Inner partitions only |
| 3+ (remaining) | Small | Spread evenly across all partitions |

Together, ranks 0–2 produce an inverted-T spatial profile matching real bump maps.

### Step 4 — Cross-partition mixing

After allocation, signals are partially exchanged between groups to create spatial overlap and avoid hard boundaries:

- The top `mix_top_k` groups (by signal count) each bleed `mix_ratio` of their home-partition signals into the other top groups' home partitions.
- **Any group that is the sole occupant of a partition is automatically added to the mixing pool**, regardless of `mix_top_k`. This prevents isolated regions where only one group appears.
- The bleed amount is divided equally among destination partitions, and is clamped so the source keeps at least 1 signal and the destination does not exceed grid capacity.

### Step 5 — Sub-partition strips

Each partition is sliced into narrow vertical strips of width `sub_width_pitch × pitch`, separated by minimum horizontal gaps of `g_x` μm. This spatially separates clusters of the same group within a partition.

### Step 6 — Cluster size sampling

The signals assigned to each group in each partition are broken into clusters whose sizes are drawn from a triangular distribution over `[N × c_min, N × c_max]`, where N is the total signal count. Larger groups are biased toward the upper bound (larger clusters).

### Step 7 — Cluster placement (zigzag pattern)

Each cluster is placed inside a bounding box within a strip. Multiple clusters within a strip are spread evenly with `g_y` μm minimum gaps between them. Bump positions follow one of two zigzag patterns, sampled with equal probability:

```
zigzag_right_high          zigzag_left_high
  row 0:  ●  ●  ●  ●        row 0:   ● ●  ●
  row 1:   ● ●  ●            row 1:  ●  ●  ●  ●
  row 2:  ●  ●  ●  ●        row 2:   ● ●  ●
  ...                         ...
  (even rows: N cols,          (even rows: N−1 cols,
   odd rows:  N−1 cols)         odd rows:  N cols)
```

X spacing = 2 × pitch (every other grid column); Y spacing = 1 × pitch (every row). Maximum 4 columns per cluster.

### Step 8 — Post-placement cluster shuffle

After all clusters are placed, `cluster_shuffle_ratio` (default 20%) of all clusters are randomly selected and their coordinate sets are pairwise swapped. Only clusters of equal size are paired, so group signal counts are exactly preserved. This introduces spatial randomness without directional bias.

### Step 9 — Dummy bumps

All candidate grid positions not occupied by a signal bump (including the outer 5% margin) are filled with dummy bumps. These represent physically present but electrically unconnected bumps in a real die.

---
```
