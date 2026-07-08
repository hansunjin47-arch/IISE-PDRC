# generator.py
#
# Generates synthetic micro placement instances for
# routing optimization benchmarks.
#
# Usage:
#   pip install pyyaml matplotlib
#   python generator.py -c input.yaml
#   python generator.py -c input.yaml --num-partitions 4 --sub-width-pitch 8

from __future__ import annotations

import argparse
import math
import os
import random
from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Tuple, Optional

import yaml
import matplotlib.pyplot as plt


# -------------------------
# 1) Config dataclasses
# -------------------------

@dataclass(frozen=True)
class GeneratorCfg:
    seed: int         # random seed for reproducibility
    g_x: int = 192    # g_x: minimum horizontal gap between micro bump clusters (μm)
    g_y: int = 192    # g_y: minimum vertical gap between micro bump clusters (μm)


@dataclass(frozen=True)
class LayoutCfg:
    L: int    # L: number of routing layers
    W: int    # W: layout width  (grid units); x range: [0, W]
    H: int    # H: layout height (grid units); y range: [0, H]

    @property
    def x_min(self) -> int: return 0
    @property
    def x_max(self) -> int: return self.W
    @property
    def y_min(self) -> int: return 0
    @property
    def y_max(self) -> int: return self.H


@dataclass(frozen=True)
class SpecCfg:
    delta: int           # Δ: unit grid size (= wire width)
    d_micro: int         # d_micro: micro bump diameter
    d_C4: int            # d_C4: C4 bump diameter
    d_via: int           # d_via: via diameter
    sc: Dict[str, int]       # sc_i: bump-to-bump center-to-center spacing
    sc_via: Dict[str, int]   # sc_{via,i}: via-to-component center-to-center spacing
    sc_wire: Dict[str, int]  # sc_{wire,i}: wire-to-component center-to-center spacing


@dataclass(frozen=True)
class signalCfg:
    signal: Dict[str, int]  # number of nets per group, e.g. {"G1": 200, "G2": 400}
    c_min: float  # lower bound of cluster size as fraction of total N
    c_max: float  # upper bound of cluster size as fraction of total N


@dataclass(frozen=True)
class RoutingCfg:
    rho: Dict[str, Optional[float]]    # ρ_L: fraction of total routing length allocated to each layer (None = unconstrained)
    sigma: Dict[str, Optional[float]]  # σ_k: allowed routing-length deviation per group (None = unconstrained)


@dataclass(frozen=True)
class PlacementCfg:
    q: int                 # q: side length of the C4 bump density-check window (in C4 pitches)
    p: int                 # p: maximum number of C4 bumps allowed within any q×q window
    # phi: dummy bump density on the ML layer (fraction of non-micro-bump grid
    # positions that get a dummy bump / obstacle). 1.0 = every empty position is
    # a dummy bump (original behavior, max masking). 0.0 = no dummy bumps at all.
    phi: float


@dataclass(frozen=True)
class OutputCfg:
    dir: str = "outputs"  # directory where generated files are saved


@dataclass(frozen=True)
class ProblemCfg:
    """Top-level configuration assembled from input.yaml."""
    schema_version: int
    generator: GeneratorCfg
    objectives: List[str]
    layout: LayoutCfg
    spec: SpecCfg
    signals: signalCfg
    placement: PlacementCfg
    routing: RoutingCfg
    output: OutputCfg


# -------------------------
# 2) YAML load
# -------------------------

class ConfigError(ValueError):
    """Raised when a required field is missing or invalid in the YAML config."""
    pass


def _req(d: Mapping[str, Any], key: str) -> Any:
    """Return d[key], raising ConfigError if the key is absent."""
    if key not in d:
        raise ConfigError(f"Missing required key: '{key}'")
    return d[key]


def load_cfg(path: str) -> ProblemCfg:
    """Parse input.yaml and return a fully validated ProblemCfg."""
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        raw = yaml.safe_load(f)

    if not isinstance(raw, dict):
        raise ConfigError("Top-level YAML must be a mapping (dict).")

    gen_raw = _req(raw, "generator")
    lay_raw = _req(raw, "layout")
    spec_raw = _req(raw, "spec")
    plc_raw = _req(raw, "placement")
    rt_raw = _req(raw, "routing")
    out_raw = raw.get("output", {"dir": "outputs"})

    phi = float(plc_raw.get("phi", 1.0))
    if not (0.0 <= phi <= 1.0):
        raise ConfigError(f"placement.phi ({phi}) must be in [0.0, 1.0].")

    W = int(_req(lay_raw, "W"))
    H = int(_req(lay_raw, "H"))
    if W <= 0:
        raise ConfigError(f"layout.W ({W}) must be positive.")
    if H <= 0:
        raise ConfigError(f"layout.H ({H}) must be positive.")

    signal_raw_full = dict(_req(raw, "signal"))
    # Exclude meta-keys — keep only group entries (G1, G2, ...)
    signal_raw = {k: v for k, v in signal_raw_full.items() if k.startswith("G")}
    if not any(int(v) > 0 for v in signal_raw.values()):
        raise ConfigError("At least one group in 'signal' must have a positive count.")
    cluster_u = float(_req(signal_raw_full, "c_min"))
    cluster_v = float(_req(signal_raw_full, "c_max"))
    if not (0 < cluster_u < cluster_v <= 1.0):
        raise ConfigError(f"signal.c_min/c_max must satisfy 0 < min < max <= 1.0 "
                          f"(got min={cluster_u}, max={cluster_v}).")

    rho_raw = dict(_req(rt_raw, "rho"))

    # ── Spacing validation ────────────────────────────────────────────────────
    # All spec spacing values are center-to-center distances (in grid units) and
    # must be non-negative multiples of delta so that they snap to the routing grid.
    delta_val = int(_req(spec_raw, "delta"))
    if delta_val <= 0:
        raise ConfigError(f"spec.delta ({delta_val}) must be a positive integer.")
    for _section in ("sc", "sc_via", "sc_wire"):
        for _sp_key, _sp_val in dict(_req(spec_raw, _section)).items():
            _sp_int = int(_sp_val)
            if _sp_int < 0:
                raise ConfigError(
                    f"spec.{_section}.{_sp_key} ({_sp_int}) must be non-negative."
                )
            # sc_wire values are checked by the validator as continuous distances,
            # so grid alignment is not required.
            if _section != "sc_wire" and _sp_int % delta_val != 0:
                raise ConfigError(
                    f"spec.{_section}.{_sp_key} = {_sp_int} (grid units) is not a multiple of "
                    f"delta = {delta_val}. All spacing values must align to the routing grid."
                )

    # ── Overlap guard: spacing must be ≥ sum of component radii ─────────────
    d_micro_v = int(_req(spec_raw, "d_micro"))
    d_C4_v    = int(_req(spec_raw, "d_C4"))
    d_via_v   = int(_req(spec_raw, "d_via"))
    _min_c2c = {
        ("sc",      "micro"): d_micro_v,
        ("sc",      "C4"):    d_C4_v,
        ("sc_via",  "micro"): (d_via_v + d_micro_v) // 2,
        ("sc_via",  "C4"):    (d_via_v + d_C4_v)    // 2,
        ("sc_via",  "via"):   d_via_v,
        ("sc_wire", "micro"): (delta_val + d_micro_v) // 2,
        ("sc_wire", "C4"):    (delta_val + d_C4_v)    // 2,
        ("sc_wire", "via"):   (delta_val + d_via_v)   // 2,
        ("sc_wire", "wire"):  delta_val,
    }
    for (_sect, _key), _min in _min_c2c.items():
        _val = int(dict(_req(spec_raw, _sect)).get(_key, 0))
        if _val < _min:
            raise ConfigError(
                f"spec.{_sect}.{_key} = {_val} < {_min} (sum of component radii) — "
                f"components would physically overlap."
            )

    return ProblemCfg(
        schema_version=int(_req(raw, "schema_version")),
        generator=GeneratorCfg(
            seed=int(gen_raw.get("seed", 42)),
            g_x=int(gen_raw.get("g_x", 192)),
            g_y=int(gen_raw.get("g_y", 192)),
        ),
        objectives=list(_req(raw, "objectives")),
        layout=LayoutCfg(
            L=int(_req(lay_raw, "L")),
            W=W,
            H=H,
        ),
        spec=SpecCfg(
            delta=int(_req(spec_raw, "delta")),
            d_micro=int(_req(spec_raw, "d_micro")),
            d_C4=int(_req(spec_raw, "d_C4")),
            d_via=int(_req(spec_raw, "d_via")),
            sc=dict(_req(spec_raw, "sc")),
            sc_via=dict(_req(spec_raw, "sc_via")),
            sc_wire=dict(_req(spec_raw, "sc_wire")),
        ),
        signals=signalCfg(signal=signal_raw, c_min=cluster_u, c_max=cluster_v),
        placement=PlacementCfg(
            q=int(plc_raw.get("q", 3)),
            p=int(plc_raw.get("p", 6)),
            phi=phi,
        ),
        routing=RoutingCfg(
            rho=rho_raw,
            sigma=dict(_req(rt_raw, "sigma")),
        ),
        output=OutputCfg(dir=str(out_raw.get("dir", "outputs"))),
    )


# -------------------------
# 3) Grid utilities
# -------------------------

def get_base_pitch(cfg: ProblemCfg) -> int:
    """Return the center-to-center pitch between micro bumps (= spec.sc.micro)."""
    pitch = int(cfg.spec.sc.get("micro", 0))
    if pitch <= 0:
        raise RuntimeError("spec.sc.micro must be a positive integer.")
    return pitch


def align_up(v: int, pitch: int) -> int:
    """Round v up to the nearest multiple of pitch."""
    return int(math.ceil(v / pitch) * pitch)


def align_down(v: int, pitch: int) -> int:
    """Round v down to the nearest multiple of pitch."""
    return int(math.floor(v / pitch) * pitch)


def usable_rect_center(cfg: ProblemCfg, ratio: float = 0.8) -> Tuple[int, int, int, int]:
    """
    Return the axis-aligned bounding box of the central usable region.

    The usable area is the inner `ratio` fraction of the full layout boundary,
    then snapped to the micro pitch grid.
    Returns (ux_min, ux_max, uy_min, uy_max) in grid units.
    """
    x_min, x_max = cfg.layout.x_min, cfg.layout.x_max
    y_min, y_max = cfg.layout.y_min, cfg.layout.y_max

    w = x_max - x_min
    h = y_max - y_min
    mx = (1 - ratio) / 2 * w
    my = (1 - ratio) / 2 * h

    ux_min = int(round(x_min + mx))
    ux_max = int(round(x_max - mx))
    uy_min = int(round(y_min + my))
    uy_max = int(round(y_max - my))

    pitch = get_base_pitch(cfg)
    ux_min = align_up(ux_min, pitch)
    ux_max = align_down(ux_max, pitch)
    uy_min = align_up(uy_min, pitch)
    uy_max = align_down(uy_max, pitch)

    if ux_min >= ux_max or uy_min >= uy_max:
        raise RuntimeError(
            f"Usable area collapsed after pitch alignment. "
            f"center_ratio={ratio}, pitch={pitch}, "
            f"usable x: [{ux_min}, {ux_max}], usable y: [{uy_min}, {uy_max}]. "
            f"Try increasing the layout size, reducing the pitch (spec.sc.micro), "
            f"or increasing center_ratio."
        )

    return ux_min, ux_max, uy_min, uy_max


# -------------------------
# 4) Fixed vertical partitions
# -------------------------

def build_vertical_partitions(NX: int, NY: int, num_parts: int = 4) -> Dict[int, Tuple[int, int, int, int]]:
    """
    Divide the usable grid (NX columns × NY rows) into `num_parts` vertical strips
    of equal width.  Each strip spans the full height.

    Returns a dict mapping partition_id (1-indexed) -> (ix0, ix1, iy0, iy1)
    where indices are into the xs_u / ys_u coordinate arrays.
    """
    num_parts = max(1, min(num_parts, NX))
    part_w = NX // num_parts
    if part_w <= 0:
        raise RuntimeError("Too many partitions for usable grid width.")

    parts: Dict[int, Tuple[int, int, int, int]] = {}
    for p in range(num_parts):
        ix0 = p * part_w
        ix1 = (p + 1) * part_w - 1 if p < num_parts - 1 else NX - 1
        iy0 = 0
        iy1 = NY - 1
        parts[p + 1] = (ix0, ix1, iy0, iy1)
    return parts


# -------------------------
# 5) Cluster size builder
# -------------------------

def build_clusters_range_sample(N: int, min_size: int, max_size: int, *, bias: float = 0.5) -> List[int]:
    """
    Decompose N bumps into a list of cluster sizes by repeatedly sampling
    from a triangular distribution over [min_size, max_size].

    Args:
        N:        total number of bumps to distribute.
        min_size: minimum cluster size (inclusive).
        max_size: maximum cluster size (inclusive).
        bias:     controls the mode of the triangular distribution.
                  0.0 → mode = min_size (prefer small clusters),
                  1.0 → mode = max_size (prefer large clusters).

    The last cluster absorbs any remainder smaller than min_size to avoid
    creating undersized clusters.
    """
    if N <= 0:
        return []
    if min_size <= 0 or max_size < min_size:
        raise ValueError(f"Invalid cluster range: min={min_size}, max={max_size}")

    bias = max(0.0, min(1.0, bias))
    mode = min_size + (max_size - min_size) * bias

    sizes: List[int] = []
    remaining = N

    while remaining > 0:
        if remaining <= max_size:
            if sizes and remaining < min_size:
                sizes[-1] += remaining
            else:
                sizes.append(remaining)
            break

        upper = min(max_size, remaining - min_size)
        if upper < min_size:
            if sizes:
                sizes[-1] += remaining
            else:
                sizes.append(remaining)
            break

        # 삼각분포로 샘플링 후 [min_size, upper] 범위로 클램핑
        clamped_mode = max(float(min_size), min(float(upper), mode))
        k = int(round(random.triangular(min_size, upper, clamped_mode)))
        k = max(min_size, min(upper, k))
        sizes.append(k)
        remaining -= k

    return sizes


# -------------------------
# 6) Point patterns
# -------------------------

def pattern_rect_fill(xs: List[int], ys: List[int], k: int) -> List[Tuple[int, int]]:
    """Fill a rectangular grid row-by-row and return the first k points."""
    pts = [(x, y) for y in ys for x in xs]
    return pts[:k]

def pattern_zigzag_right_high(xs: List[int], ys: List[int], k: int) -> List[Tuple[int, int]]:
    """
    Zigzag pattern where even rows use the original column positions and
    odd rows are shifted right by half a column pitch, staying inside the
    bounding box.  Falls back to rect_fill when there is only one column.

    Row order:  full / shifted / full / shifted / ...
    """
    pts: List[Tuple[int, int]] = []

    if len(xs) <= 1 or len(ys) == 0:
        return pattern_rect_fill(xs, ys, k)

    step = xs[1] - xs[0]
    half = step // 2

    # bounding box 안쪽에만 들어가는 shifted row
    shifted = [x + half for x in xs[:-1]]

    for j, y in enumerate(ys):
        row = xs if (j % 2 == 0) else shifted
        for x in row:
            pts.append((x, y))
            if len(pts) >= k:
                return pts

    return pts[:k]


def pattern_zigzag_left_high(xs: List[int], ys: List[int], k: int) -> List[Tuple[int, int]]:
    """
    Mirror of zigzag_right_high: even rows are shifted and odd rows are full.
    Shifted row has one fewer column to stay within the cluster bounding box.

    Row order:  shifted(N-1) / full(N) / shifted(N-1) / full(N) / ...
    """
    pts: List[Tuple[int, int]] = []

    if len(xs) <= 1 or len(ys) == 0:
        return pattern_rect_fill(xs, ys, k)

    step = xs[1] - xs[0]
    half = step // 2

    # bounding box 안쪽에만 들어가는 shifted row
    shifted = [x + half for x in xs[:-1]]

    for j, y in enumerate(ys):
        row = shifted if (j % 2 == 0) else xs
        for x in row:
            pts.append((x, y))
            if len(pts) >= k:
                return pts

    return pts[:k]


def generate_cluster_points(shape_name: str, xs: List[int], ys: List[int], k: int) -> List[Tuple[int, int]]:
    """Dispatch to the appropriate pattern function and return k (x, y) coordinates."""
    if shape_name == "zigzag_right_high":
        return pattern_zigzag_right_high(xs, ys, k)
    if shape_name == "zigzag_left_high":
        return pattern_zigzag_left_high(xs, ys, k)
    return pattern_zigzag_right_high(xs, ys, k)

def sample_cluster_shape() -> str:
    """
    Randomly select an internal arrangement pattern for a cluster.

    Probabilities:
      50% zigzag_right_high   (N / N-1 / N / N-1 ...)
      50% zigzag_left_high    (N-1 / N / N-1 / N ...)
    """
    r = random.random()
    if r < 0.50:
        return "zigzag_right_high"
    return "zigzag_left_high"

# -------------------------
# 7) Partition allocation
# -------------------------

def desired_partition_count(group_count: int, total_count: int) -> int:
    """
    Return how many vertical partitions a group should occupy,
    based on its share of the total signal count.

      < 25% of total → 1 partition
      25–50%         → 2 partitions
      50–75%         → 3 partitions
      ≥ 75%          → 4 partitions
    """
    if total_count <= 0:
        return 1
    ratio = group_count / float(total_count)
    if ratio < 0.25:
        return 1
    if ratio < 0.50:
        return 2
    if ratio < 0.75:
        return 3
    return 4


def allocate_count_with_capacity(
    count: int,
    chosen_parts: List[int],
    all_parts: List[int],
    cap_remain: Dict[int, int],
) -> Dict[int, int]:
    """
    Greedily distribute `count` units across `chosen_parts`, respecting each
    partition's remaining capacity.  If the chosen partitions are exhausted,
    spills into other partitions from `all_parts`.

    Returns a dict of {partition_id: allocated_count} for non-zero entries.
    """
    if count <= 0:
        return {}

    selected: List[int] = []
    for p in chosen_parts:
        if p not in selected:
            selected.append(p)

    alloc: Dict[int, int] = {p: 0 for p in all_parts}
    remaining = int(count)

    while remaining > 0:
        feasible = [p for p in selected if cap_remain.get(p, 0) > 0]
        if not feasible:
            extras = [p for p in all_parts if p not in selected and cap_remain.get(p, 0) > 0]
            if not extras:
                break
            random.shuffle(extras)
            selected.append(extras[0])
            continue

        target = int(math.ceil(remaining / max(1, len(feasible))))
        progressed = False

        ranked = feasible[:]
        random.shuffle(ranked)
        ranked.sort(key=lambda p: cap_remain[p], reverse=True)

        for p in ranked:
            if remaining <= 0:
                break
            take = min(cap_remain[p], target, remaining)
            if take <= 0:
                continue
            alloc[p] += take
            cap_remain[p] -= take
            remaining -= take
            progressed = True

        if not progressed:
            break

    if remaining > 0:
        raise RuntimeError(
            f"Partition allocation failed: remaining={remaining}, "
            f"count={count}, chosen_parts={chosen_parts}"
        )

    return {p: alloc[p] for p in all_parts if alloc[p] > 0}


def choose_nonadjacent_pair(
    feasible_parts: List[int],
    load_assigned: Dict[int, int],
) -> List[int]:
    """
    Choose exactly 2 non-adjacent partitions (|p1 - p2| > 1) from feasible_parts.
    Prefers pairs whose combined existing load is highest (spatial clustering
    within each chosen partition).  Falls back to any 2 feasible partitions
    if no non-adjacent pair exists (e.g., only 2 partitions total).
    """
    nonadj_pairs = [
        (p1, p2)
        for i, p1 in enumerate(feasible_parts)
        for p2 in feasible_parts[i + 1:]
        if abs(p1 - p2) > 1
    ]

    if nonadj_pairs:
        scored = [
            (load_assigned.get(p1, 0) + load_assigned.get(p2, 0) + random.random() * 0.05, [p1, p2])
            for p1, p2 in nonadj_pairs
        ]
        scored.sort(key=lambda x: x[0], reverse=True)
        return scored[0][1]

    # Fallback: non-adjacent pair impossible (too few partitions), pick any 2
    fallback = feasible_parts[:]
    random.shuffle(fallback)
    return fallback[:2]


def choose_k_partitions(
    feasible_parts: List[int],
    k: int,
) -> List[int]:
    """
    Randomly choose k partitions from feasible_parts with no contiguity constraint.
    Used for k >= 3 where any combination of partitions is acceptable
    (e.g., [1,3,4] is just as valid as [1,2,3]).
    """
    chosen = feasible_parts[:]
    random.shuffle(chosen)
    return chosen[:k]


def allocate_groups_to_partitions(
    counts: Dict[str, int],
    parts: Dict[int, Tuple[int, int, int, int]],
) -> Dict[str, Dict[int, int]]:
    """
    Assign each net group to one or more vertical partitions, distributing
    signals proportionally while respecting partition capacity.

    Strategy:
      1. Process groups in descending order of size.
      2. Initially favour partitions not yet assigned to any group (coverage pass).
      3. Once all partitions are covered, switch to adjacent-window selection
         to keep signals spatially localised.
      4. Exception: when exactly 2 partitions are chosen, always pick non-adjacent
         ones (|p1 - p2| > 1).  3+ partitions keep the contiguous-window rule.

    Returns a nested dict: {group_id: {partition_id: count}}.
    """
    boxes = sorted(parts.keys())
    capacities = {
        p: (parts[p][1] - parts[p][0] + 1) * (parts[p][3] - parts[p][2] + 1)
        for p in boxes
    }
    cap_remain = dict(capacities)

    total = sum(int(v) for v in counts.values())
    groups_order = sorted(counts.keys(), key=lambda g: counts[g], reverse=True)

    alloc: Dict[str, Dict[int, int]] = {g: {p: 0 for p in boxes} for g in counts.keys()}
    covered_parts: set[int] = set()
    switched_to_random = False

    for g in groups_order:
        Ng = int(counts[g])
        if Ng <= 0:
            continue

        desired = desired_partition_count(Ng, total)

        if (not switched_to_random) and len(covered_parts) < len(boxes):
            unassigned = [p for p in boxes if p not in covered_parts and cap_remain.get(p, 0) > 0]
            random.shuffle(unassigned)
            if unassigned:
                k = min(desired, len(unassigned))
                if k == 2:
                    # Prefer non-adjacent partitions when exactly 2 are chosen
                    load_assigned = {p: capacities[p] - cap_remain[p] for p in boxes}
                    chosen = choose_nonadjacent_pair(unassigned, load_assigned)
                else:
                    chosen = unassigned[:k]
            else:
                switched_to_random = True
                chosen = []
        else:
            switched_to_random = True
            chosen = []

        if switched_to_random:
            feasible = [p for p in boxes if cap_remain.get(p, 0) > 0]
            if not feasible:
                raise RuntimeError("No feasible partition remains.")

            k = min(len(feasible), max(2, desired))
            load_assigned = {p: capacities[p] - cap_remain[p] for p in boxes}
            if k == 2:
                # Prefer non-adjacent partitions when exactly 2 are chosen
                chosen = choose_nonadjacent_pair(feasible, load_assigned)
            else:
                # k >= 3: any combination allowed (no contiguity constraint)
                chosen = choose_k_partitions(feasible, k)

        part_alloc = allocate_count_with_capacity(
            count=Ng,
            chosen_parts=chosen,
            all_parts=boxes,
            cap_remain=cap_remain,
        )

        for p, v in part_alloc.items():
            alloc[g][p] = v
            if v > 0:
                covered_parts.add(p)

        if len(covered_parts) == len(boxes):
            switched_to_random = True

    s = 0
    for g in alloc:
        s += sum(alloc[g].values())
    if s != total:
        raise RuntimeError(f"Allocation failed: allocated={s} != total={total}")

    return alloc


# -------------------------
# 7-a2) Structured allocation
# -------------------------

def allocate_groups_structured(
    counts: Dict[str, int],
    parts: Dict[int, Tuple[int, int, int, int]],
) -> Dict[str, Dict[int, int]]:
    """
    Structured spatial allocation that mirrors the T-shape pattern observed
    in real IC layouts:

      rank 0 (largest group, e.g. G2):
        All partitions, center-weighted (60% in inner partitions, 40% in outer).
      rank 1 (2nd largest, e.g. G1):
        Only the two extreme partitions (leftmost + rightmost), equal split.
        This creates the "horizontal bar of the T" at the periphery.
      rank 2 (3rd largest, e.g. G3):
        Inner partitions only (symmetric around center).
      rank 3+ (remaining small groups):
        Spread evenly across all partitions.

    Returns a nested dict: {group_id: {partition_id: count}}.
    """
    boxes = sorted(parts.keys())
    n = len(boxes)
    alloc: Dict[str, Dict[int, int]] = {g: {p: 0 for p in boxes} for g in counts.keys()}

    groups_by_size = sorted(
        [g for g in counts if counts[g] > 0],
        key=lambda g: counts[g],
        reverse=True,
    )

    for rank, g in enumerate(groups_by_size):
        Ng = counts[g]
        if Ng <= 0:
            continue

        if rank == 0:
            # Largest: center-weighted distribution across all partitions
            # Center = middle half of partitions, outer = remaining
            if n >= 4:
                outer_parts = [boxes[0], boxes[-1]]
                center_parts = boxes[1:-1]
            else:
                outer_parts = []
                center_parts = boxes[:]

            center_total = round(Ng * 0.6) if center_parts else 0
            outer_total = Ng - center_total

            if center_parts:
                per_center = center_total // len(center_parts)
                remainder = center_total - per_center * len(center_parts)
                for i, p in enumerate(center_parts):
                    alloc[g][p] = per_center + (1 if i < remainder else 0)

            if outer_parts:
                per_outer = outer_total // len(outer_parts)
                remainder = outer_total - per_outer * len(outer_parts)
                for i, p in enumerate(outer_parts):
                    alloc[g][p] = per_outer + (1 if i < remainder else 0)

        elif rank == 1:
            # 2nd largest: extreme partitions only (symmetric periphery)
            if n >= 2:
                extreme_parts = [boxes[0], boxes[-1]]
            else:
                extreme_parts = boxes[:]
            half1 = Ng // 2
            half2 = Ng - half1
            alloc[g][extreme_parts[0]] = half1
            alloc[g][extreme_parts[-1]] = half2

        elif rank == 2:
            # 3rd largest: inner partitions only
            if n >= 3:
                inner_parts = boxes[1:-1]
            else:
                inner_parts = boxes[:]
            per = Ng // len(inner_parts)
            remainder = Ng - per * len(inner_parts)
            for i, p in enumerate(inner_parts):
                alloc[g][p] = per + (1 if i < remainder else 0)

        else:
            # Remaining small groups: spread evenly across all partitions
            per = Ng // n
            remainder = Ng - per * n
            for i, p in enumerate(boxes):
                alloc[g][p] = per + (1 if i < remainder else 0)

    # Validate total
    total_expected = sum(counts.values())
    total_allocated = sum(alloc[g][p] for g in alloc for p in boxes)
    if total_allocated != total_expected:
        raise RuntimeError(
            f"Structured allocation mismatch: expected={total_expected}, got={total_allocated}"
        )

    return alloc


# -------------------------
# 7-b) Cross-partition mixing
# -------------------------

def apply_cross_partition_mixing(
    alloc: Dict[str, Dict[int, int]],
    counts: Dict[str, int],
    parts: Dict[int, Tuple[int, int, int, int]],
    *,
    mix_ratio: float,
    mix_top_k: int,
) -> None:
    """
    Post-process partition allocation so that the top `mix_top_k` groups
    (by signal count) partially share each other's home partitions.

    Algorithm
    ---------
    1. Find each top group's "home partition" = partition with the most
       signals assigned to it.
    2. For every ordered pair (g_src, g_dst) of distinct top groups whose
       home partitions differ, move
           bleed = round(alloc[g_src][home_src] * mix_ratio
                         / (n_destinations))
       signals from g_src's home into g_dst's home partition.
       Dividing by the number of destinations keeps the *total* bleed from
       g_src always equal to mix_ratio × home signals, regardless of how
       many top groups exist.
    3. Capacity-safe: bleed is clamped so the destination partition never
       exceeds its grid capacity, and the source keeps at least 1 signal.

    Modifies `alloc` in-place; the total signal count is preserved.
    """
    if mix_ratio <= 0.0 or mix_top_k < 2:
        return

    boxes = sorted(parts.keys())

    # Grid capacity of each partition (max signals it can hold)
    capacities: Dict[int, int] = {
        p: (parts[p][1] - parts[p][0] + 1) * (parts[p][3] - parts[p][2] + 1)
        for p in boxes
    }

    # Top-K groups by signal count (largest first)
    top_groups: List[str] = sorted(
        [g for g in counts if counts[g] > 0],
        key=lambda g: counts[g],
        reverse=True,
    )[:mix_top_k]

    # Force-include any group that is the SOLE occupant of some partition.
    # A "sole occupant" partition has exactly one group with signals > 0.
    # Such groups are isolated by construction and must be mixed so that
    # other groups gain presence there (reducing the "boxed-off" appearance).
    for p in boxes:
        groups_in_p = [g for g in counts if alloc[g].get(p, 0) > 0]
        if len(groups_in_p) == 1:
            sole = groups_in_p[0]
            if sole not in top_groups:
                top_groups.append(sole)

    if len(top_groups) < 2:
        return

    # Home partition = where each top group currently has the most signals
    primary: Dict[str, int] = {
        g: max(boxes, key=lambda p: alloc[g][p])   # type: ignore[arg-type]
        for g in top_groups
    }

    # Running total load per partition (to track remaining capacity)
    load: Dict[int, int] = {
        p: sum(alloc[g][p] for g in counts)
        for p in boxes
    }

    for g_src in top_groups:
        p_src = primary[g_src]

        # Destination: other top groups with a *different* home partition
        dsts = [
            (g_dst, primary[g_dst])
            for g_dst in top_groups
            if g_dst != g_src and primary[g_dst] != p_src
        ]
        if not dsts:
            continue

        n_dsts = len(dsts)

        for _, p_dst in dsts:
            # Each destination receives an equal share of the total bleed
            bleed = round(alloc[g_src][p_src] * mix_ratio / n_dsts)

            # Safety clamps
            avail  = capacities[p_dst] - load[p_dst]          # free cells in dst
            budget = alloc[g_src][p_src] - 1                  # keep ≥ 1 in src home
            bleed  = max(0, min(bleed, avail, budget))

            if bleed == 0:
                continue

            alloc[g_src][p_src] -= bleed
            alloc[g_src][p_dst] += bleed
            load[p_src]         -= bleed
            load[p_dst]         += bleed

    print(
        f"[Mixing] {len(top_groups)} groups (top-{mix_top_k} + sole-occupants), "
        f"mix_ratio={mix_ratio:.2f}, home partitions: { {g: primary[g] for g in top_groups} }"
    )


# -------------------------
# 8) Sub-partitions (vertical strips)
# -------------------------

def build_subpartitions_in_partition(
    part_rect: Tuple[int, int, int, int],
    *,
    sub_w_cells: int,
    sub_gap_cells: int,
) -> List[Tuple[int, int, int, int]]:
    """
    Slice a partition into narrow vertical strips (sub-partitions) of width
    `sub_w_cells`, with `sub_gap_cells` as a **minimum** gap between strips.

    Any extra horizontal space (beyond the minimum gaps) is distributed evenly
    as padding before the first strip, between strips, and after the last strip
    (n+1 slots for n strips) — symmetric, analogous to how cluster_gap_y works
    in the vertical direction.

    If the partition is too narrow for even one strip, the whole partition
    is returned as a single strip.

    Returns a list of (sx0, sx1, sy0, sy1) index tuples.
    """
    ix0, ix1, iy0, iy1 = part_rect
    total_w = ix1 - ix0 + 1

    if sub_w_cells <= 0:
        raise RuntimeError("sub_w_cells must be positive.")
    if sub_gap_cells < 0:
        raise RuntimeError("sub_gap_cells must be non-negative.")

    # Fallback: partition too narrow for even one strip at full width
    if sub_w_cells > total_w:
        return [(ix0, ix1, iy0, iy1)]

    # How many strips fit with minimum gap?
    # n strips need: n * sub_w + (n-1) * min_gap
    n_strips = max(1, (total_w + sub_gap_cells) // (sub_w_cells + sub_gap_cells))

    # Distribute extra horizontal space evenly in (n+1) slots:
    # [ extra | strip | min_gap+extra | strip | min_gap+extra | ... | strip | extra ]
    need = n_strips * sub_w_cells + (n_strips - 1) * sub_gap_cells
    free = total_w - need

    extra_slots = n_strips + 1
    extra_each  = free // extra_slots
    extra_rem   = free % extra_slots
    extras = [extra_each] * extra_slots
    for i in range(extra_rem):
        extras[i] += 1

    strips: List[Tuple[int, int, int, int]] = []
    cur_x = ix0 + extras[0]

    for i in range(n_strips):
        sx0 = cur_x
        sx1 = cur_x + sub_w_cells - 1
        strips.append((sx0, sx1, iy0, iy1))
        if i < n_strips - 1:
            cur_x = sx1 + 1 + sub_gap_cells + extras[i + 1]

    return strips


# -------------------------
# 9) Cluster allocation to strips
# -------------------------

# Maximum number of point columns within a single cluster grid.
# Keeping this small enforces a tall, narrow cluster shape.
MAX_CLUSTER_POINT_COLS = 4

# Spacing between sampled points within a cluster, measured in grid cells.
# A value of 2 means one empty cell is left between adjacent C4 positions.
POINT_STEP = 2


def estimate_cluster_height_cells(
    k: int,
    strip_w: int,
    slack: float,
    *,
    point_step: int = POINT_STEP,
    y_point_step: int = 1,          # Y samples every row (step=1) for true diagonal
    max_point_cols: int = MAX_CLUSTER_POINT_COLS,
) -> int:
    """
    Estimate the vertical height (in grid cells) needed to fit k bumps in a
    strip of width `strip_w`, given the point-step spacing and a slack factor.

    X uses `point_step` (= POINT_STEP = 2): effective X pitch = 2 × base pitch.
    Y uses `y_point_step` (= 1):           every ys_u row is used.
    Together, the zigzag diagonal is exactly one step right + one step down.

    The slack factor inflates the cell count so that not every slot is filled,
    leaving room for the chosen placement pattern.

    Zigzag correction: zigzag_right_high / zigzag_left_high patterns place
    N columns on even rows but only N-1 columns on odd rows, so the average
    capacity per row is (N + N-1) / 2 = N - 0.5.  Using this effective column
    count prevents underestimating the required height.
    """
    eff_cells = int(math.ceil(k * max(1.0, float(slack))))
    raw_point_cols = int(math.ceil(strip_w / float(point_step)))
    point_cols = max(1, min(max_point_cols, raw_point_cols))
    # Zigzag patterns alternate N and N-1 columns per row → effective avg = N - 0.5
    effective_cols = max(1.0, float(point_cols) - 0.5)
    point_rows = int(math.ceil(eff_cells / effective_cols))
    h = point_rows * y_point_step       # height in ys_u grid cells
    return max(y_point_step, h)


def allocate_clusters_to_strips_with_global_remain(
    cluster_specs: List[Tuple[int, int]],   # (size, num_cols)
    chosen_strip_ids: List[int],
    strips: List[Tuple[int, int, int, int]],
    strip_remain_height: Dict[int, int],
    *,
    cluster_height_slack: float,
    cluster_gap_rows: int,
) -> List[Tuple[int, int, int]]:            # (strip_idx, size, num_cols)
    """
    Assign each cluster to one of the chosen strips, fitting larger clusters
    first (descending order) into the strip with the most remaining height.

    Each cluster carries its own `num_cols` which controls its width and thus
    the height needed to fit its k bumps (fewer columns → taller cluster).

    Updates `strip_remain_height` in-place as clusters are placed.
    Returns a list of (strip_index, size, num_cols) triples in the original
    cluster order.
    """
    if not chosen_strip_ids:
        raise RuntimeError("No chosen strips.")

    alloc: List[Optional[Tuple[int, int, int]]] = [None] * len(cluster_specs)

    order = list(range(len(cluster_specs)))
    order.sort(key=lambda i: cluster_specs[i][0], reverse=True)

    for idx in order:
        k, nc = cluster_specs[idx]

        feasible: List[Tuple[int, int]] = []
        for si in chosen_strip_ids:
            sx0, sx1, sy0, sy1 = strips[si]
            sw = sx1 - sx0 + 1

            need_h = estimate_cluster_height_cells(
                k,
                sw,
                cluster_height_slack,
                y_point_step=1,
                point_step=POINT_STEP,
                max_point_cols=nc,   # per-cluster column count
            )

            extra_gap = cluster_gap_rows if strip_remain_height[si] < (sy1 - sy0 + 1) else 0
            total_need_h = need_h + extra_gap

            if strip_remain_height[si] >= total_need_h:
                feasible.append((si, total_need_h))

        if not feasible:
            raise RuntimeError(
                f"Failed to allocate cluster k={k} (cols={nc}) to chosen strips. "
                f"Try reduce cluster density or increase strip count."
            )

        feasible.sort(key=lambda x: (-strip_remain_height[x[0]], x[0]))
        chosen_si, used_h = feasible[0]

        alloc[idx] = (chosen_si, k, nc)
        strip_remain_height[chosen_si] -= used_h

    out: List[Tuple[int, int, int]] = []
    for v in alloc:
        if v is None:
            raise RuntimeError("Internal error: cluster strip allocation incomplete.")
        out.append(v)
    return out


# -------------------------
# 10) Spread clusters over strip height
# -------------------------

def build_cluster_rects_in_strip(
    cluster_specs_in_order: List[Tuple[int, int]],  # (size, num_cols)
    strip_rect: Tuple[int, int, int, int],
    *,
    cluster_gap_rows: int,
    cluster_height_slack: float,
) -> List[Tuple[int, int, int, int]]:
    """
    Distribute clusters vertically within a strip, spreading any leftover
    space evenly as padding above, between, and below clusters.

    Each cluster's height is estimated using its own `num_cols` so that
    narrower clusters (fewer columns) correctly get taller bounding boxes.

    Returns a list of (sx0, sx1, cy0, cy1) bounding boxes, one per cluster,
    in the same order as `cluster_specs_in_order`.
    """
    sx0, sx1, sy0, sy1 = strip_rect
    sw = sx1 - sx0 + 1
    sh = sy1 - sy0 + 1

    heights = [
        estimate_cluster_height_cells(
            k,
            sw,
            cluster_height_slack,
            point_step=POINT_STEP,
            y_point_step=1,
            max_point_cols=nc,   # per-cluster column count
        )
        for k, nc in cluster_specs_in_order
    ]
    n = len(heights)
    if n == 0:
        return []

    total_core = sum(heights)
    total_fixed_gap = cluster_gap_rows * (n - 1)
    need = total_core + total_fixed_gap

    if need > sh:
        raise RuntimeError(
            f"Strip vertical spread overflow: need={need}, strip_h={sh}, strip_w={sw}"
        )

    free = sh - need
    extra_slots = n + 1
    extra_each = free // extra_slots
    extra_rem  = free % extra_slots
    extras = [extra_each] * extra_slots
    for i in range(extra_rem):
        extras[i] += 1

    rects: List[Tuple[int, int, int, int]] = []
    cur_top = sy1 - extras[0]

    for i, h in enumerate(heights):
        cy1 = cur_top
        cy0 = cy1 - h + 1
        rects.append((sx0, sx1, cy0, cy1))
        if i < n - 1:
            cur_top = cy0 - 1 - cluster_gap_rows - extras[i + 1]

    return rects


# -------------------------
# 11) Strict cluster grid helpers
# -------------------------

def build_strict_cluster_grid(
    sx0: int,
    sx1: int,
    sy0: int,
    sy1: int,
    xs_u: List[int],
    ys_u: List[int],
    *,
    point_step: int = POINT_STEP,
    max_point_cols: int = MAX_CLUSTER_POINT_COLS,
    x_align: str = "center",
) -> Tuple[List[int], List[int]]:
    """
    Extract the candidate x/y coordinates for C4 placement within a cluster
    bounding box, subsampled by `point_step` and capped at `max_point_cols` columns.

    `x_align` controls which columns are selected when the strip is wider than
    the cluster's column count:
      "left"   – always leftmost columns
      "right"  – always rightmost columns
      "center" – centred (original behaviour)
      "random" – random starting offset, so clusters of the same width appear at
                 different x positions within the strip (breaks visual alignment)

    Returns (xs, ys) as lists of actual layout coordinates (not indices).
    """
    xs = [xs_u[i] for i in range(sx0, sx1 + 1, point_step)]   # X: every 2nd → effective pitch = 2×base
    ys = [ys_u[j] for j in range(sy0, sy1 + 1, 1)]            # Y: every row  → step = base pitch

    if not xs:
        xs = [xs_u[sx0]]
    if not ys:
        ys = [ys_u[sy0]]

    if len(xs) > max_point_cols:
        n_avail = len(xs)
        if x_align == "random":
            start = random.randint(0, n_avail - max_point_cols)
        elif x_align == "left":
            start = 0
        elif x_align == "right":
            start = n_avail - max_point_cols
        else:  # "center"
            mid = n_avail // 2
            start = max(0, mid - (max_point_cols // 2))
        xs = xs[start:start + max_point_cols]
        if len(xs) < max_point_cols:
            xs = xs[-max_point_cols:]

    return xs, ys


def validate_cluster_grid_capacity(
    xs: List[int],
    ys: List[int],
    k: int,
    group: str,
    part_id: int,
    strip_id: int,
) -> None:
    """Raise RuntimeError if the cluster grid has fewer candidate points than needed."""
    cap = len(xs) * len(ys)
    if cap < k:
        raise RuntimeError(
            f"Strict cluster grid capacity shortage: "
            f"partition={part_id}, strip={strip_id}, group={group}, "
            f"k={k}, capacity={cap}, cols={len(xs)}, rows={len(ys)}"
        )


# -------------------------
# 12) Placement
# -------------------------

def place_micros_grouped(
    cfg: ProblemCfg,
    *,
    num_partitions: int = 4,
    sub_partition_width_in_pitch: int = 8,
    cluster_gap_x_pitch: int = 1,
    cluster_gap_y_rows: int = 1,
    cluster_height_slack: float = 1.15,
    mix_ratio: float = 0.15,
    mix_top_k: int = 2,
    cluster_shuffle_ratio: float = 0.20,
    seed_override: Optional[int] = None,
    spatial_mode: str = "random",
) -> Tuple[List[Dict[str, Any]], Dict[str, List[int]], List[int]]:
    """
    Main placement routine.  Generates micro coordinates for all
    net groups according to the following pipeline:

    1. Compute the usable grid from the central `center_ratio` of the layout.
    2. Derive cluster size bounds as [1% , 4%] of the total signal count.
    3. Divide the grid into `num_partitions` vertical partitions and assign
       each net group to partitions (mode controlled by `spatial_mode`).
       - "random":     probabilistic assignment biased by group size.
       - "structured": fixed T-shape template:
             rank-0 (largest) → center-weighted across all partitions,
             rank-1 (2nd)     → extreme (leftmost + rightmost) only,
             rank-2 (3rd)     → inner partitions only,
             rank-3+ (rest)   → spread evenly.
    4. Apply cross-partition mixing: the top `mix_top_k` groups exchange
       `mix_ratio` of their home-partition signals with each other, so that
       the dominant groups spatially overlap instead of being fully isolated.
    5. Within each partition, slice further into narrow strips of width
       `sub_partition_width_in_pitch` pitch units.
    6. For each group in each partition, sample cluster sizes from a triangular
       distribution biased toward larger clusters for larger groups.
    7. Fit clusters into strips (largest first), then lay out micro
       coordinates using a randomly chosen zigzag pattern
       (zigzag_right_high / zigzag_left_high).
    8. Deduplicate against a global set of used coordinates.

    Args:
        mix_ratio:    fraction of each top group's home-partition signals that
                      bleed into the other top groups' home partitions (0 = off).
        mix_top_k:    number of largest groups to apply cross-partition mixing.
        seed_override: if provided, overrides cfg.generator.seed (used for batch generation).
        spatial_mode: "random" (default) or "structured" (T-shape template).

    Returns:
        micros          – list of dicts with keys: group, micro_id, micro_x, micro_y
        group_partition_ids – dict mapping group name to the partition IDs it uses
        all_cluster_sizes   – flat list of every cluster size sampled during generation
    """
    seed = seed_override if seed_override is not None else cfg.generator.seed
    random.seed(seed)

    pitch = get_base_pitch(cfg)

    # 5% margin on all four sides: signal bumps are restricted to the inner
    # 90% of the layout.  Dummy bumps still cover the entire grid.
    _MARGIN = 0.05
    _mx = round(cfg.layout.W * _MARGIN)
    _my = round(cfg.layout.H * _MARGIN)
    ux_min = align_up  (cfg.layout.x_min + _mx, pitch)
    ux_max = align_down(cfg.layout.x_max - _mx, pitch)
    uy_min = align_up  (cfg.layout.y_min + _my, pitch)
    uy_max = align_down(cfg.layout.y_max - _my, pitch)
    xs_u = list(range(ux_min, ux_max + 1, pitch))
    ys_u = list(range(uy_min, uy_max + 1, pitch))
    NX, NY = len(xs_u), len(ys_u)
    print(f"[Margin5%] usable grid: {NX} cols x {NY} rows "
          f"(x={ux_min}..{ux_max}, y={uy_min}..{uy_max})")

    counts = {g: int(v) for g, v in cfg.signals.signal.items() if int(v) > 0}
    groups_order = sorted(counts.keys(), key=lambda g: counts[g], reverse=True)

    total = sum(counts.values())
    if total > NX * NY:
        raise RuntimeError(
            f"Too many signals for the layout grid: "
            f"total signals={total}, grid cells={NX * NY} ({NX} cols × {NY} rows). "
            f"Reduce the number of signals, increase the layout size, "
            f"or reduce spec.sc.micro."
        )

    # Cluster size bounds from signal.cluster_u / cluster_v (fractions of total N).
    u = cfg.signals.c_min
    v = cfg.signals.c_max
    cluster_min = max(1, int(total * u))
    cluster_max = max(cluster_min + 1, int(total * v))
    # Used to compute per-group mode m_k
    count_min = min(counts.values())
    count_max = max(counts.values())

    print(f"[ClusterRange] u={u}, v={v}, min={cluster_min}, max={cluster_max} (total={total})")

    partitions = build_vertical_partitions(NX, NY, num_parts=num_partitions)
    boxes = list(sorted(partitions.keys()))

    if spatial_mode == "structured":
        print(f"[SpatialMode] structured (T-shape template)")
        alloc = allocate_groups_structured(counts, partitions)
    else:
        print(f"[SpatialMode] random (probabilistic partition assignment)")
        alloc = allocate_groups_to_partitions(counts, partitions)

    # Cross-partition mixing: let top groups partially share each other's zones
    apply_cross_partition_mixing(
        alloc, counts, partitions,
        mix_ratio=mix_ratio,
        mix_top_k=mix_top_k,
    )

    print("\n[Partition Allocation Result]")
    for p in sorted(partitions.keys()):
        print(f"Partition {p}")
        has_any = False
        for g in sorted(alloc.keys()):
            n = alloc[g].get(p, 0)
            if n > 0:
                has_any = True
                print(f"  {g} -> {n}")
        if not has_any:
            print("  (empty)")

    group_partition_ids: Dict[str, List[int]] = {}
    for g in counts.keys():
        used = [b for b in boxes if alloc[g].get(b, 0) > 0]
        group_partition_ids[g] = sorted(used)

    global_used: set[Tuple[int, int]] = set()
    micros: List[Dict[str, Any]] = []
    per_group_counter: Dict[str, int] = {g: 0 for g in cfg.signals.signal.keys()}
    all_cluster_sizes: List[int] = []  # tracks every cluster size sampled

    sub_w_cells   = max(1, int(sub_partition_width_in_pitch))
    sub_gap_cells = max(0, int(cluster_gap_x_pitch))

    # Maximum columns a cluster can use, derived from strip width and point step.
    # A random num_cols in [1, _strip_max_cols] is sampled per cluster so that
    # clusters have variable widths (and thus different heights), breaking the
    # rigid visual alignment that arises when every cluster uses max columns.
    _strip_max_cols = max(1, sub_w_cells // POINT_STEP)

    partition_order = boxes[:]
    # Partition processing order is fixed (no directional shuffle).
    # Spatial randomness is introduced via per-cluster num_cols + x_align="random"
    # in build_strict_cluster_grid, and the post-placement cluster shuffle below.
    all_clusters: List[Tuple[str, List[Tuple[int, int]]]] = []

    for part_id in partition_order:
        print(f"\n[Start Placement] Partition {part_id}")

        groups_in_part = [(g, int(alloc[g][part_id])) for g in groups_order if int(alloc[g][part_id]) > 0]
        if not groups_in_part:
            print("  (no assigned groups)")
            continue

        for g, n in groups_in_part:
            print(f"  {g} assigned {n} signals")

        part_rect = partitions[part_id]
        strips = build_subpartitions_in_partition(
            part_rect,
            sub_w_cells=sub_w_cells,
            sub_gap_cells=sub_gap_cells,
        )
        print(f"  sub-partitions in partition {part_id}: {len(strips)}")

        strip_remain_height: Dict[int, int] = {}
        for si, (sx0, sx1, sy0, sy1) in enumerate(strips):
            strip_remain_height[si] = sy1 - sy0 + 1

        # (group, size, num_cols) per strip
        strip_to_group_clusters: Dict[int, List[Tuple[str, int, int]]] = {i: [] for i in range(len(strips))}

        for g, n_gc in groups_in_part:
            print(f"    -> building clusters for {g} ({n_gc})")
            # m_k = u + (v - u) * (G_k - min G) / (max G - min G)
            # Triangular(N*u, N*v, N*m_k) → bias = (m_k - u) / (v - u) = ratio
            ratio = (counts[g] - count_min) / (count_max - count_min) if count_max > count_min else 0.5
            cluster_sizes = build_clusters_range_sample(n_gc, cluster_min, cluster_max, bias=ratio)
            all_cluster_sizes.extend(cluster_sizes)

            # Randomly assign each cluster a column count in [_min_cols, _strip_max_cols].
            # This gives clusters variable widths (narrower clusters are taller),
            # naturally breaking the rigid vertical column alignment.
            # Minimum 2 columns: zigzag patterns alternate N and N-1 cols per row,
            # so 1 col would produce a 0-col odd row — degenerate.
            # Upper bound = _strip_max_cols (default 4).
            _min_cols = max(2, _strip_max_cols // 2)
            cluster_num_cols = [random.randint(_min_cols, _strip_max_cols) for _ in cluster_sizes]

            # Pair (size, num_cols) and sort by size descending for allocation
            cluster_specs: List[Tuple[int, int]] = sorted(
                zip(cluster_sizes, cluster_num_cols),
                key=lambda x: x[0],
                reverse=True,
            )

            base_strip_count = desired_partition_count(counts[g], total)
            min_needed_by_volume = max(1, int(math.ceil(n_gc / 48.0)))
            desired_strip_count = min(
                len(strips),
                max(base_strip_count, min_needed_by_volume),
            )

            strip_rank = list(range(len(strips)))
            random.shuffle(strip_rank)
            strip_rank.sort(key=lambda si: (-strip_remain_height[si], si))
            chosen_strip_ids = strip_rank[:desired_strip_count]

            cluster_assign = allocate_clusters_to_strips_with_global_remain(
                cluster_specs=cluster_specs,
                chosen_strip_ids=chosen_strip_ids,
                strips=strips,
                strip_remain_height=strip_remain_height,
                cluster_height_slack=cluster_height_slack,
                cluster_gap_rows=cluster_gap_y_rows,
            )

            for global_si, k, nc in cluster_assign:
                strip_to_group_clusters[global_si].append((g, k, nc))

        # (group, size, num_cols, rect)
        strip_cluster_rects: Dict[int, List[Tuple[str, int, int, Tuple[int, int, int, int]]]] = {}

        for si, items in strip_to_group_clusters.items():
            if not items:
                strip_cluster_rects[si] = []
                continue

            items_sorted = items[:]
            items_sorted.sort(key=lambda x: (x[0], -x[1]))
            specs = [(k, nc) for (_, k, nc) in items_sorted]

            rects = build_cluster_rects_in_strip(
                cluster_specs_in_order=specs,
                strip_rect=strips[si],
                cluster_gap_rows=cluster_gap_y_rows,
                cluster_height_slack=cluster_height_slack,
            )

            packed_items: List[Tuple[str, int, int, Tuple[int, int, int, int]]] = []
            for (g, k, nc), rect in zip(items_sorted, rects):
                packed_items.append((g, k, nc, rect))
            strip_cluster_rects[si] = packed_items

        for si in range(len(strips)):
            items = strip_cluster_rects.get(si, [])
            if not items:
                continue

            for g, k, nc, rect in items:
                sx0, sx1, sy0, sy1 = rect

                shape = sample_cluster_shape()

                xs, ys = build_strict_cluster_grid(
                    sx0, sx1, sy0, sy1,
                    xs_u, ys_u,
                    point_step=POINT_STEP,
                    max_point_cols=nc,       # per-cluster column count
                    x_align="random",        # random x offset within strip
                )

                validate_cluster_grid_capacity(xs, ys, k, g, part_id, si)

                pts = generate_cluster_points(shape, xs, ys, k)

                chosen_pts: List[Tuple[int, int]] = []
                for x, y in pts:
                    if (x, y) in global_used:
                        continue
                    chosen_pts.append((x, y))
                    if len(chosen_pts) >= k:
                        break

                if len(chosen_pts) < k:
                    raise RuntimeError(
                        f"Not enough free points under strict spacing/width rule. "
                        f"partition={part_id}, strip={si}, group={g}, k={k}, "
                        f"shape={shape}, cols={len(xs)}, rows={len(ys)}"
                    )

                for x, y in chosen_pts:
                    global_used.add((x, y))
                all_clusters.append((g, chosen_pts))

    # ── Post-placement cluster shuffle ────────────────────────────────────────
    # Randomly select ~cluster_shuffle_ratio of all placed clusters and swap
    # their coordinate sets pairwise (no directional bias).
    # Only clusters of equal size are paired so group signal counts are preserved.
    _n_total = len(all_clusters)
    _n_sel   = max(0, round(_n_total * cluster_shuffle_ratio))
    if _n_sel >= 2:
        _order = list(range(_n_total))
        random.shuffle(_order)
        _selected = _order[:_n_sel]

        # Group selected indices by cluster size
        _by_size: Dict[int, List[int]] = {}
        for _idx in _selected:
            _sz = len(all_clusters[_idx][1])
            _by_size.setdefault(_sz, []).append(_idx)

        _swapped = 0
        for _idxs in _by_size.values():
            random.shuffle(_idxs)
            for _i in range(0, len(_idxs) - 1, 2):
                _a, _b = _idxs[_i], _idxs[_i + 1]
                _g_a, _pts_a = all_clusters[_a]
                _g_b, _pts_b = all_clusters[_b]
                all_clusters[_a] = (_g_a, _pts_b)  # group label stays, coords swap
                all_clusters[_b] = (_g_b, _pts_a)
                _swapped += 1

        print(
            f"\n[ClusterShuffle] total={_n_total}, selected={_n_sel} "
            f"({cluster_shuffle_ratio*100:.0f}%), swapped={_swapped} pairs"
        )
    else:
        print(f"\n[ClusterShuffle] skipped (clusters={_n_total}, selected={_n_sel})")

    # ── Flatten all_clusters into micros ─────────────────────────────────
    for _g, _pts in all_clusters:
        for _x, _y in _pts:
            _bid = per_group_counter[_g]
            per_group_counter[_g] += 1
            micros.append({
                "group":         _g,
                "micro_id": f"{_bid+1:04d}",
                "micro_x":  _x,
                "micro_y":  _y,
            })

    # ── Dummy bump generation ──────────────────────────────────────────────────
    # Like C4 dummy bumps, dummy bumps cover the ENTIRE layout (not just the
    # usable area).  Signal bumps are placed only in the usable center region;
    # all other grid positions — including the outer margins — become dummy bumps.
    xs_full = list(range(cfg.layout.x_min, cfg.layout.x_max + 1, pitch))
    ys_full = list(range(cfg.layout.y_min, cfg.layout.y_max + 1, pitch))

    placed_positions: set[Tuple[int, int]] = {
        (b["micro_x"], b["micro_y"]) for b in micros
    }
    total_grid = len(xs_full) * len(ys_full)
    phi = cfg.placement.phi

    empty_positions = [(x, y) for y in ys_full for x in xs_full if (x, y) not in placed_positions]
    # phi: 1.0 = 빈 자리를 전부 채움(기존 동작), 0.0 = 전혀 채우지 않음(마스킹 없음).
    # 자리마다 확률적으로 채우는 게 아니라, 빈 자리 중 정확히 phi*N개를 무작위로 뽑아서 채운다
    # (확률 샘플링은 매 실행마다 실제 채워지는 개수가 들썩여서 난이도 비교가 흔들리기 때문).
    num_to_fill = min(len(empty_positions), round(len(empty_positions) * phi))
    filled_positions = set(random.sample(empty_positions, num_to_fill))

    dummy_cnt = 0
    for (x, y) in empty_positions:
        if (x, y) in filled_positions:
            dummy_cnt += 1
            micros.append({
                "group":         "dummy",
                "micro_id": f"null_{dummy_cnt:05d}",
                "micro_x":  x,
                "micro_y":  y,
            })

    print(
        f"\n[DummyBumps] grid={total_grid:,} (full layout), "
        f"real={len(placed_positions)}, dummy={dummy_cnt} "
        f"(phi={phi}, empty={len(empty_positions)}, skipped={len(empty_positions) - dummy_cnt})"
    )

    return micros, group_partition_ids, all_cluster_sizes


# -------------------------
# 13) Generation statistics
# -------------------------

def print_generation_stats(
    micros: List[Dict[str, Any]],
    cluster_sizes: List[int],
    cfg: ProblemCfg,
) -> None:
    """Print a summary report of the generated instance (dummy bumps excluded)."""
    import statistics as _stats

    # Separate real bumps from dummy bumps
    real_bumps = [b for b in micros if b["group"] != "dummy"]
    null_count = len(micros) - len(real_bumps)

    print("\n" + "=" * 52)
    print("[Generation Statistics]")

    # Per-group counts (real bumps only)
    group_counts: Dict[str, int] = {}
    for b in real_bumps:
        group_counts[b["group"]] = group_counts.get(b["group"], 0) + 1
    print(f"  Real bumps  : {len(real_bumps)}")
    print(f"  Dummy bumps  : {null_count}")
    print(f"  Total in CSV: {len(micros)}")
    for g in sorted(group_counts):
        expected = cfg.signals.signal.get(g, 0)
        print(f"  {g}          : {group_counts[g]}  (expected {expected})")

    # Cluster size distribution
    if cluster_sizes:
        stdev = _stats.stdev(cluster_sizes) if len(cluster_sizes) > 1 else 0.0
        print(f"\n  Clusters    : {len(cluster_sizes)}")
        print(f"  Cluster size: min={min(cluster_sizes)}, max={max(cluster_sizes)}, "
              f"mean={_stats.mean(cluster_sizes):.1f}, stdev={stdev:.1f}")

    print("=" * 52)

    print("=" * 52)


# -------------------------
# 15) Plotting
# -------------------------

def plot_all(
    cfg: ProblemCfg,
    micros: List[Dict[str, Any]],
    save_path: str | None = None,
) -> None:
    """Publication-quality scatter plot of all generated bumps, colour-coded by group.

    Visual design:
    - Dummy bumps: very faint gray dots (s=2, alpha=0.12) as background grid context.
    - Signal bumps: large, opaque markers with thin black outlines for clear group separation.
    - Layout boundary: solid black rectangle.
    - Usable (central) region: dashed gray rectangle.
    - Colorblind-friendly palette with 5 distinct hues.
    """
    import matplotlib.patches as mpatches
    import matplotlib.ticker as mticker

    dummy_bumps = [b for b in micros if b["group"] == "dummy"]
    real_bumps  = [b for b in micros if b["group"] != "dummy"]

    # ── Colorblind-friendly, print-safe palette ────────────────────────────
    _PALETTE = [
        "#1f77b4",   # blue       G1
        "#d62728",   # red        G2
        "#2ca02c",   # green      G3
        "#ff7f0e",   # orange     G4
        "#9467bd",   # purple     G5
        "#8c564b",   # brown      G6
        "#e377c2",   # pink       G7
        "#17becf",   # cyan       G8
    ]
    all_groups   = sorted({b["group"] for b in real_bumps}) if real_bumps else []
    group2color  = {g: _PALETTE[i % len(_PALETTE)] for i, g in enumerate(all_groups)}

    # ── Figure setup ──────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(10, 7))
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    if not real_bumps and not dummy_bumps:
        ax.set_xlim(cfg.layout.x_min, cfg.layout.x_max)
        ax.set_ylim(cfg.layout.y_min, cfg.layout.y_max)
        ax.set_aspect("equal", adjustable="box")
        ax.set_title("No generated micro bumps")
        if save_path:
            os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
            fig.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.show()
        return

    # ── Layer 0: layout boundary (solid black box) ────────────────────────
    lx0, lx1 = cfg.layout.x_min, cfg.layout.x_max
    ly0, ly1 = cfg.layout.y_min, cfg.layout.y_max
    ax.add_patch(mpatches.Rectangle(
        (lx0, ly0), lx1 - lx0, ly1 - ly0,
        linewidth=1.2, edgecolor="black", facecolor="none", zorder=5,
    ))

    # ── Layer 2: dummy bumps — faint grid context ────────────────────────
    if dummy_bumps:
        dx = [b["micro_x"] for b in dummy_bumps]
        dy = [b["micro_y"] for b in dummy_bumps]
        ax.scatter(
            dx, dy,
            s=4, c="#aaaaaa", alpha=0.45,
            linewidths=0, edgecolors="none",
            zorder=1, rasterized=True,
        )

    # ── Layer 3: signal bumps — same marker size as dummy bumps ──────────
    for g in all_groups:
        grp_bumps = [b for b in real_bumps if b["group"] == g]
        gx = [b["micro_x"] for b in grp_bumps]
        gy = [b["micro_y"] for b in grp_bumps]
        ax.scatter(
            gx, gy,
            s=4,
            c=group2color[g],
            alpha=1.0,
            linewidths=0,
            edgecolors="none",
            zorder=3,
            label=f"{g}  (n={len(grp_bumps):,})",
        )

    # ── Axes ──────────────────────────────────────────────────────────────
    ax.set_xlim(lx0 - 100, lx1 + 100)
    ax.set_ylim(ly0 - 100, ly1 + 100)
    ax.set_aspect("equal", adjustable="box")

    ax.set_xlabel("X coordinate (grid units)", fontsize=13)
    ax.set_ylabel("Y coordinate (grid units)", fontsize=13)
    ax.tick_params(axis="both", labelsize=11)
    ax.xaxis.set_major_locator(mticker.MultipleLocator(1000))
    ax.yaxis.set_major_locator(mticker.MultipleLocator(1000))

    # light grid for spatial reference
    ax.grid(True, linestyle=":", linewidth=0.4, color="#cccccc", zorder=0)

    # ── Legend: signal groups + dummy (no counts) ────────────────────────
    handles, _ = ax.get_legend_handles_labels()
    if dummy_bumps:
        handles.append(plt.Line2D(
            [0], [0], marker="o", color="w",
            label="Dummy",
            markerfacecolor="#bbbbbb", markeredgecolor="none", markersize=7,
        ))
    ax.legend(
        handles=handles,
        title="Group",
        loc="upper left",
        bbox_to_anchor=(1.01, 1),
        borderaxespad=0,
        fontsize=11,
        title_fontsize=11,
        framealpha=0.9,
        edgecolor="#aaaaaa",
        markerscale=1.8,
    )

    # ── Save / show ───────────────────────────────────────────────────────
    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
        fig.savefig(save_path, dpi=300, bbox_inches="tight",
                    facecolor="white", edgecolor="none")
        print(f"[OK] saved -> {save_path}")

    plt.show()

def plot_C4_candidates(
    cfg: ProblemCfg,
    save_path: str | None = None,
) -> None:
    """Plot C4 candidate grid only (colored by grid position, no placed bumps)."""
    pitch  = int(cfg.spec.sc["C4"])
    lay    = cfg.layout
    x_min, x_max = lay.x_min, lay.x_max
    y_min, y_max = lay.y_min, lay.y_max

    # Build candidate grid
    cand_xs: List[int] = []
    cand_ys: List[int] = []
    y = y_max
    while y >= y_min:
        x = x_min
        while x <= x_max:
            cand_xs.append(x)
            cand_ys.append(y)
            x += pitch
        y -= pitch

    fig, ax = plt.subplots(figsize=(10, 10))

    ax.scatter(
        cand_xs, cand_ys,
        s=20, c="#4a90d9", alpha=0.7,
        linewidths=0, edgecolors="none",
        zorder=1,
    )

    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("X-coordinate", fontsize=13)
    ax.set_ylabel("Y-coordinate", fontsize=13)
    ax.tick_params(axis="both", labelsize=11)
    ax.set_title(
        f"C4 bump candidate grid | pitch={pitch} | total={len(cand_xs)}",
        fontsize=15, pad=14,
    )
    ax.grid(False)

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path, dpi=300, bbox_inches="tight")

    plt.show()


# -------------------------
# 14) CSV
# -------------------------

def save_micro_coordinate_csv(
    out_dir: str,
    micros: List[Dict[str, Any]],
    filename: str = "micro_coordinate.csv",
) -> str:
    """Write micro coordinates to a CSV file with columns: group, micro_id, micro_x, micro_y."""
    import csv

    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, filename)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["group", "micro_id", "micro_x", "micro_y"])
        for b in micros:
            w.writerow([b["group"], b["micro_id"], b["micro_x"], b["micro_y"]])
    return path


def save_C4_candidate_csv(
    out_dir: str,
    cfg: "ProblemCfg",
    filename: str = "C4_candidate.csv",
) -> str:
    """Generate and save the predefined C4 candidate grid. CSV columns: candidate_id, x, y"""
    import csv

    pitch  = int(cfg.spec.sc["C4"])
    lay    = cfg.layout
    x_min, x_max = lay.x_min, lay.x_max
    y_min, y_max = lay.y_min, lay.y_max

    candidates: List[Tuple[int, int]] = []
    y = y_max
    while y >= y_min:
        x = x_min
        while x <= x_max:
            candidates.append((x, y))
            x += pitch
        y -= pitch

    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, filename)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["candidate_id", "C4_x", "C4_y"])
        for cid, (x, y) in enumerate(candidates):
            w.writerow([cid, x, y])
    return path


# -------------------------
# 15) CLI helpers
# -------------------------



# -------------------------
# 16) Main
# -------------------------

def main():
    ap = argparse.ArgumentParser(
        description="Generate synthetic micro placement benchmark instances."
    )
    ap.add_argument("-c", "--config", required=True, help="YAML config path (e.g., input.yaml)")
    ap.add_argument("--no-plot", action="store_true")
    ap.add_argument("--num-instances", type=int, default=1,
                    help="number of instances to generate with consecutive seeds (default: 1)")

    args = ap.parse_args()

    if args.num_instances < 1:
        ap.error(f"--num-instances must be at least 1, got {args.num_instances}.")

    cfg = load_cfg(args.config)
    base_seed = cfg.generator.seed
    num_instances = int(args.num_instances)
    batch = num_instances > 1

    # Convert cluster gap to pitch steps.
    # g_x / g_y are in sc.micro units (= micro-bump pitch = 48 grid units each).
    # 1 step = 1 × sc.micro grid units of gap between adjacent cluster boundaries.
    _pitch = int(cfg.spec.sc["micro"])  # 48 grid units per step
    _gx_um = cfg.generator.g_x
    _gy_um = cfg.generator.g_y

    # Validate: cluster_gap must be non-negative.
    if _gx_um < 0:
        ap.error(f"g_x = {_gx_um} must be non-negative.")
    if _gy_um < 0:
        ap.error(f"g_y = {_gy_um} must be non-negative.")

    # Convert to pitch steps (ceil so the minimum gap is always satisfied).
    _gx_steps = math.ceil(_gx_um / _pitch) if _gx_um > 0 else 0
    _gy_steps = math.ceil(_gy_um / _pitch) if _gy_um > 0 else 0

    placement_kwargs = dict(
        num_partitions=4,
        sub_partition_width_in_pitch=8,
        cluster_gap_x_pitch=_gx_steps,
        cluster_gap_y_rows=_gy_steps,
        cluster_height_slack=1.15,
        mix_ratio=0.15,
        mix_top_k=2,
        cluster_shuffle_ratio=0.20,
        spatial_mode="random",
    )

    for i in range(num_instances):
        seed = base_seed + i
        out_dir = os.path.join(cfg.output.dir, f"instance_{i + 1}") if batch else cfg.output.dir

        print(f"\n{'='*52}")
        if batch:
            print(f"[Instance {i+1}/{num_instances}]  seed={seed}  -> {out_dir}")
        print(f"[Config] partitions=4, gap_x={_gx_um}μm ({_gx_steps} pitch steps), gap_y={_gy_um}μm ({_gy_steps} pitch steps)")
        print(f"[ClusterShape] max_point_cols={MAX_CLUSTER_POINT_COLS}, point_step={POINT_STEP}")
        print("[PatternProb] zigzag_right_high=0.50, zigzag_left_high=0.50")

        micros, group_partition_ids, cluster_sizes = place_micros_grouped(
            cfg,
            **placement_kwargs,
            seed_override=seed,
        )

        for g in sorted(group_partition_ids.keys()):
            if group_partition_ids[g]:
                print(f"  {g} -> partitions {group_partition_ids[g]}")

        print_generation_stats(micros, cluster_sizes, cfg)

        os.makedirs(out_dir, exist_ok=True)
        csv_path = save_micro_coordinate_csv(out_dir, micros)
        print(f"[OK] wrote: {csv_path}")

        cand_path = save_C4_candidate_csv(out_dir, cfg)
        print(f"[OK] wrote: {cand_path}")

        if not args.no_plot:
            fig_path = os.path.join(out_dir, "micro_coordinate.png")
            plot_all(cfg, micros, save_path=fig_path)
            print(f"[OK] wrote: {fig_path}")

            cand_fig_path = os.path.join(out_dir, "C4_candidate.png")
            plot_C4_candidates(cfg, save_path=cand_fig_path)
            print(f"[OK] wrote: {cand_fig_path}")


if __name__ == "__main__":
    main()