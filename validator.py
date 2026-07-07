# validator.py
#
# Validates routing output JSON for placement and routing feasibility,
# and measures routing performance.
#
# Usage:
#   python validator.py -c input.yaml -r result.json

from __future__ import annotations

import sys
sys.stdout.reconfigure(encoding='utf-8')
sys.stderr.reconfigure(encoding='utf-8')

import argparse
import csv
import glob
import json
import math
import os
from dataclasses import dataclass, field
from typing import Any, Dict, FrozenSet, List, Optional, Tuple

import numpy as np
import yaml


# ─────────────────────────────────────────────────────────────────────────────
# 1. Config
# ─────────────────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class ValidatorCfg:
    L: int                  # number of routing layers
    pitch: int              # sc.micro value = micro center-to-center grid pitch
    micro_radius: float     # d_micro / 2
    C4_radius: float        # d_C4 / 2
    via_radius: float       # d_via / 2
    # spacing: flat dict of all center-to-center distances, keyed by table notation.
    #   sc_micro, sc_C4              → bump-to-bump (from spec.sc.*)
    #   sc_via_micro, sc_via_C4, sc_via_via → via-to-component (from spec.sc_via.*)
    #   sc_wire_micro, sc_wire_C4, sc_wire_via, sc_wire_wire → wire-to-component (from spec.sc_wire.*)
    spacing: Dict[str, float]
    W: int                  # layout width
    H: int                  # layout height

    @property
    def x_min(self) -> int: return 0
    @property
    def x_max(self) -> int: return self.W
    @property
    def y_min(self) -> int: return 0
    @property
    def y_max(self) -> int: return self.H
    q: float                # q: density-check window side length (in C4-pitch units)
    p: int                  # p: max C4 bumps allowed within any q×q window
    delta: float            # Δ: unit grid size (= wire width)
    # rho_L: per-layer routing length ratio constraint (key: 'm1', 'm2', ...).
    # None → unconstrained; otherwise layer_length / total_length must be ≤ value
    # (0.0 is a real constraint now: that layer must carry ~0% of the length).
    rho: Dict[str, Optional[float]]
    K: int                       # K: number of signal groups (G1 … G{K})
    total_signals: int           # total net count = sum of all group sizes
    # sigma_k: per-group routing length deviation limit: (max - min) / min <= limit.
    sigma: Dict[str, Optional[float]]      # key: 'G1', 'G2', ...


def load_validator_cfg(yaml_path: str) -> ValidatorCfg:
    """Load the subset of input.yaml needed for validation."""
    with open(yaml_path, 'r', encoding='utf-8', errors='replace') as f:
        raw = yaml.safe_load(f)

    lay  = raw['layout']
    spec = raw['spec']
    plc  = raw['placement']   # q / p for density check
    rout = raw['routing']
    sig  = raw['signal']

    # Build flat spacing dict from three nested sections (sc, sc_via, sc_wire).
    spacing: Dict[str, float] = {}
    for k, v in spec.get('sc', {}).items():
        spacing[f'sc_{k}'] = float(v)          # sc_micro, sc_C4
    for k, v in spec.get('sc_via', {}).items():
        spacing[f'sc_via_{k}'] = float(v)      # sc_via_micro, sc_via_C4, sc_via_via
    for k, v in spec.get('sc_wire', {}).items():
        spacing[f'sc_wire_{k}'] = float(v)     # sc_wire_micro, sc_wire_C4, sc_wire_via, sc_wire_wire

    # ── Overlap guard: spacing must be ≥ sum of component radii ─────────────
    _d_micro = float(spec['d_micro'])
    _d_C4    = float(spec['d_C4'])
    _d_via   = float(spec['d_via'])
    _delta   = float(spec['delta'])
    _min_c2c = {
        'sc_micro':      _d_micro,
        'sc_C4':         _d_C4,
        'sc_via_micro':  (_d_via + _d_micro) / 2,
        'sc_via_C4':     (_d_via + _d_C4)    / 2,
        'sc_via_via':    _d_via,
        'sc_wire_micro': (_delta + _d_micro)  / 2,
        'sc_wire_C4':    (_delta + _d_C4)     / 2,
        'sc_wire_via':   (_delta + _d_via)    / 2,
        'sc_wire_wire':  _delta,
    }
    for _key, _min in _min_c2c.items():
        _val = spacing.get(_key)
        if _val is not None and _val < _min:
            raise ValueError(
                f"spec.{_key} = {_val} < {_min} (sum of component radii) — "
                f"components would physically overlap."
            )

    # Normalize rho keys to lowercase ('M1' → 'm1') to match RoutedNet layer names.
    # None (yaml null) means "unconstrained" and is preserved as-is.
    rho = {
        k.lower(): (float(v) if v is not None else None)
        for k, v in rout.get('rho', {}).items()
    }

    sigma = {
        k: (float(v) if v is not None else None)
        for k, v in rout.get('sigma', {}).items()
    }

    K = int(sig['K'])
    return ValidatorCfg(
        L             = int(lay['L']),
        pitch         = int(spec['sc']['micro']),
        micro_radius  = float(spec['d_micro']) / 2,
        C4_radius     = float(spec['d_C4'])    / 2,
        via_radius    = float(spec['d_via'])   / 2,
        spacing       = spacing,
        W             = int(lay['W']),
        H             = int(lay['H']),
        q             = float(plc['q']),
        p             = int(plc['p']),
        delta         = float(spec['delta']),
        rho           = rho,
        K             = K,
        total_signals = sum(int(sig[f'G{i}']) for i in range(1, K + 1)),
        sigma         = sigma,
    )


# ─────────────────────────────────────────────────────────────────────────────
# 2. Bump candidate loader
# ─────────────────────────────────────────────────────────────────────────────

def load_C4_candidates(csv_path: str) -> List[Tuple[int, int]]:
    """Load C4 candidate grid from C4_candidate.csv.
    Returns a list of (bump_x, bump_y) tuples."""
    candidates: List[Tuple[int, int]] = []
    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            candidates.append((int(row['C4_x']), int(row['C4_y'])))
    return candidates


def load_micro_coordinates(csv_path: str) -> Dict[str, Tuple[int, int]]:
    """Load micro_coordinate.csv.
    Returns a dict mapping netname (e.g. 'G1_0001') to (micro_x, micro_y)."""
    micro_map: Dict[str, Tuple[int, int]] = {}
    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            netname = f"{row['group']}_{row['micro_id']}"
            micro_map[netname] = (int(row['micro_x']), int(row['micro_y']))
    return micro_map


# ─────────────────────────────────────────────────────────────────────────────
# 3. Routing JSON format & loader
# ─────────────────────────────────────────────────────────────────────────────
#
# Expected JSON format:
# [
#   {
#     "netname": "G1_0001",
#     "m1": [[x1,y1], [x2,y2], [x3,y3]],   <- C4 → via12 (waypoints only)
#     "m2": [[x3,y3], [x4,y4]],             <- via12 → via23
#     ...
#     "mL": [[..], [xN,yN]]                 <- via(L-1,L) → micro
#   },
#   ...
# ]
#
# Rules:
#   - All L layers must be present for every net (no empty layers).
#   - Each layer must have >= 2 points.
#   - Last point of mN must equal first point of m(N+1)  (via junction).
#   - m1[0]  = C4 position.
#   - mL[-1] = micro position.

Coord = Tuple[float, float]


@dataclass
class RoutedNet:
    netname: str
    layers: Dict[str, List[Coord]]   # key: 'm1', 'm2', ..., 'mL'

    # ── Derived properties ────────────────────────────────────────────────

    def layer_names(self) -> List[str]:
        """Layer names sorted numerically: m1, m2, ..., mL."""
        return sorted(self.layers.keys(), key=lambda s: int(s[1:]))

    @property
    def micro(self) -> Coord:
        """mL last point = micro position (top layer)."""
        return self.layers[self.layer_names()[-1]][-1]

    @property
    def C4(self) -> Coord:
        """m1 first point = C4 position (bottom layer)."""
        return self.layers['m1'][0]

    def via_positions(self) -> List[Tuple[str, Coord]]:
        """
        Return [(via_name, (x, y)), ...] for every inter-layer junction.
        The via between mN and m(N+1) is at the last point of mN.
        """
        lns = self.layer_names()
        return [
            (f'via{i+1}{i+2}', self.layers[lns[i]][-1])
            for i in range(len(lns) - 1)
        ]


def load_routing_json(path: str, num_layers: int) -> List[RoutedNet]:
    """Parse and validate the routing output JSON file."""
    with open(path, 'r', encoding='utf-8') as f:
        raw = json.load(f)

    if not isinstance(raw, list):
        raise ValueError("Routing JSON top level must be a list.")

    layer_names = [f'm{i}' for i in range(1, num_layers + 1)]
    nets: List[RoutedNet] = []

    for idx, entry in enumerate(raw):
        netname = entry.get('netname', f'net_{idx}')

        # Check all layers are present with sufficient points
        layers: Dict[str, List[Coord]] = {}
        for ln in layer_names:
            if ln not in entry:
                raise ValueError(f"[{netname}] Missing layer '{ln}'.")
            pts = entry[ln]
            if not isinstance(pts, list) or len(pts) < 2:
                raise ValueError(
                    f"[{netname}][{ln}] Must have at least 2 waypoints, got {len(pts)}."
                )
            layers[ln] = [(float(p[0]), float(p[1])) for p in pts]

        nets.append(RoutedNet(netname=netname, layers=layers))

    return nets


# ─────────────────────────────────────────────────────────────────────────────
# 3. Component extraction
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class Component:
    kind: str                # 'micro' | 'via' | 'C4'
    x: float
    y: float
    radius: float
    netname: str
    label: str               # e.g. 'micro(G1_0001)', 'via12(G1_0001)'
    layers: FrozenSet[int]   # physical layer(s) this component occupies
    #   micro  → {1}
    #   via_N,N+1 → {N, N+1}  (footprint spans both adjacent layers)
    #   C4 → {num_layers}


def extract_components(nets: List[RoutedNet], cfg: ValidatorCfg) -> List[Component]:
    """
    Extract all physical circular components from routed nets:
      - micro          : mL last point              → layer num_layers (top)
      - C4             : m1 first point             → layer 1 (bottom)
      - Via between mN and m(N+1) : junction point  → layers {N, N+1}
    """
    L = cfg.L
    comps: List[Component] = []
    for net in nets:
        tx, ty = net.micro
        comps.append(Component('micro', tx, ty, cfg.micro_radius,
                               net.netname, f'micro({net.netname})',
                               frozenset({L})))

        bx, by = net.C4
        comps.append(Component('C4', bx, by, cfg.C4_radius,
                               net.netname, f'C4({net.netname})',
                               frozenset({1})))

        for i, (via_name, (vx, vy)) in enumerate(net.via_positions()):
            # via between layer i+1 and i+2
            comps.append(Component('via', vx, vy, cfg.via_radius,
                                   net.netname, f'{via_name}({net.netname})',
                                   frozenset({i + 1, i + 2})))
    return comps


# ─────────────────────────────────────────────────────────────────────────────
# 4. Placement feasibility checker
# ─────────────────────────────────────────────────────────────────────────────

# Mapping from component kind pair → spacing key in cfg.spacing.
# All values are center-to-center minimum distances.
_SPACING_KEY: Dict[FrozenSet[str], str] = {
    frozenset({'micro', 'micro'}): 'sc_micro',
    frozenset({'micro', 'via'}):   'sc_via_micro',
    frozenset({'micro', 'C4'}):    'sc_micro_C4',    # not in yaml → falls to fallback
    frozenset({'via',   'via'}):   'sc_via_via',
    frozenset({'via',   'C4'}):    'sc_via_C4',
    frozenset({'C4',    'C4'}):    'sc_C4',
}


@dataclass
class Violation:
    category: str       # 'spacing' | 'density' | 'candidate' | 'bounds'
    description: str


def _dist(ci: Component, cj: Component) -> float:
    return math.sqrt((ci.x - cj.x) ** 2 + (ci.y - cj.y) ** 2)


def check_placement_feasibility(
    components: List[Component],
    cfg: ValidatorCfg,
    candidates: List[Tuple[int, int]],
    nets: Optional[List[Any]] = None,
    micro_map: Optional[Dict[str, Tuple[int, int]]] = None,
) -> List[Violation]:
    """
    Placement feasibility checks:

    1. micro coordinate & net completeness
       Every netname in micro_map must appear in the routing output with a matching
       micro coordinate, and vice versa.

    2. Minimum spacing (subsumes non-overlap)
       - If a spacing rule exists for the pair → check that rule.
         All spacing values are center-to-center: dist >= cfg.spacing[key].
       - If no spacing rule exists for the pair → fallback: check non-overlap only
         (center_dist >= sum_of_radii), since we have no tighter constraint to apply.

    3. Bump candidate validation
       Each C4 must be located at one of the predefined candidate grid positions.

    4. Window density (anchor = candidate grid positions)
       For any window × window area (coordinate units), C4 center count must be
       ≤ max_count. Parameters read from yaml: placement.window / placement.max_count.
    """
    violations: List[Violation] = []
    n = len(components)

    # ── Check 1: micro coordinate & net completeness ────────────────────────
    if nets is not None and micro_map is not None:
        routed_names = {net.netname for net in nets}
        # dummy micro bumps are not routing targets, only obstacles -> excluded from completeness check
        csv_names    = {name for name in micro_map if not name.startswith('dummy_')}

        # Nets in CSV but missing from routing output
        for name in sorted(csv_names - routed_names):
            violations.append(Violation(
                category='micro_match',
                description=f"'{name}' in micro_coordinate.csv but missing from routing output."
            ))

        # Nets in routing output but missing from CSV
        for name in sorted(routed_names - csv_names):
            violations.append(Violation(
                category='micro_match',
                description=f"'{name}' in routing output but missing from micro_coordinate.csv."
            ))

        # Nets present in both: check micro coordinate matches
        for net in nets:
            if net.netname not in micro_map:
                continue
            expected = micro_map[net.netname]
            actual   = (int(net.micro[0]), int(net.micro[1]))
            if actual != expected:
                violations.append(Violation(
                    category='micro_match',
                    description=(
                        f"[{net.netname}] micro coordinate mismatch: "
                        f"routing has {actual}, micro_coordinate.csv has {expected}."
                    )
                ))

    # ── Check 2: pairwise spacing (same-layer pairs only) ────────────────
    # Rule: minimum spacing applies uniformly to every component pair,
    #       regardless of whether the two components belong to the same
    #       net/signal (no same-net exception).
    for i in range(n):
        for j in range(i + 1, n):
            ci, cj = components[i], components[j]

            # Skip pairs that share no physical layer
            if not (ci.layers & cj.layers):
                continue

            dist = _dist(ci, cj)

            # All spacing values in cfg.spacing are center-to-center distances.
            key = _SPACING_KEY.get(frozenset({ci.kind, cj.kind}))

            if key in cfg.spacing:
                # Center-to-center check for all component pairs
                if dist < cfg.spacing[key]:
                    violations.append(Violation(
                        category='spacing',
                        description=(
                            f"{ci.label} ↔ {cj.label} [{key}]: "
                            f"center_dist={dist:.2f} < min={cfg.spacing[key]:.2f}"
                        )
                    ))
            else:
                # No spacing rule → fallback: non-overlap only (center_dist >= sum of radii)
                sum_radii = ci.radius + cj.radius
                if dist < sum_radii:
                    violations.append(Violation(
                        category='spacing',
                        description=(
                            f"{ci.label} ↔ {cj.label} [no rule, overlap]: "
                            f"center_dist={dist:.2f} < sum_radii={sum_radii:.2f}"
                        )
                    ))

    C4_pts = [(c.x, c.y) for c in components if c.kind == 'C4']

    # ── Check 2: C4 candidate validation ───────────────────────────────
    candidate_set = set(candidates)
    for c in components:
        if c.kind != 'C4':
            continue
        if (int(c.x), int(c.y)) not in candidate_set:
            violations.append(Violation(
                category='candidate',
                description=(
                    f"{c.label}: position ({int(c.x)}, {int(c.y)}) "
                    f"is not a valid C4 candidate grid point."
                )
            ))

    # ── Check 3: window density for bumps ────────────────────────────────
    # For any window × window area (in coordinate units), the number of C4
    # center points inside must be ≤ max_count.
    #
    # Anchor enumeration: use candidate grid positions when available
    # (each candidate point is a potential window left-bottom corner).
    # Fallback: O(N²) combinations of (bump_i.x, bump_j.y).
    C4_pitch = cfg.spacing.get('sc_C4', 1.0)
    W         = cfg.q * C4_pitch   # window in C4-pitch units → coordinate units
    max_count = cfg.p

    anchors = candidates

    n_violations = 0
    max_density  = 0
    worst_origin: Optional[Tuple[float, float]] = None

    for (left, bottom) in anchors:
        right = left + W
        top   = bottom + W
        cnt = sum(
            1 for (px, py) in C4_pts
            if left <= px < right and bottom <= py < top
        )
        if cnt > max_density:
            max_density  = cnt
            worst_origin = (left, bottom)
        if cnt > max_count:
            n_violations += 1

    if n_violations > 0:
        violations.append(Violation(
            category='density',
            description=(
                f"{n_violations} window(s) of size {W}×{W} exceed max_count={max_count}. "
                f"Peak density={max_density} at origin {worst_origin}."
            )
        ))

    # ── Check 3: Layout bounds for all components ─────────────────────────
    for c in components:
        if not (cfg.x_min <= c.x <= cfg.x_max and cfg.y_min <= c.y <= cfg.y_max):
            violations.append(Violation(
                category='bounds',
                description=(
                    f"{c.label}: position ({c.x}, {c.y}) outside layout bounds "
                    f"x=[0, {cfg.W}], y=[0, {cfg.H}]."
                )
            ))

    return violations


# ─────────────────────────────────────────────────────────────────────────────
# 5. Report printer
# ─────────────────────────────────────────────────────────────────────────────

_MAX_SHOW = 10   # max violations to print per category


def _print_section(label: str, items: List[Violation]) -> None:
    if items:
        print(f"\n  FAIL {label} ({len(items)} violation(s)):")
        for v in items[:_MAX_SHOW]:
            print(f"      {v.description}")
        if len(items) > _MAX_SHOW:
            print(f"      ... and {len(items) - _MAX_SHOW} more.")
    else:
        print(f"\n  PASS {label}")


def print_placement_report(violations: List[Violation]) -> bool:
    """Print a structured placement feasibility report. Returns True if feasible."""
    tsv_match = [v for v in violations if v.category == 'micro_match']
    spacing   = [v for v in violations if v.category == 'spacing']
    candidate = [v for v in violations if v.category == 'candidate']
    density   = [v for v in violations if v.category == 'density']
    bounds    = [v for v in violations if v.category == 'bounds']

    print("\n" + "=" * 60)
    print("[Placement Feasibility]")
    _print_section("micro coordinate & net completeness", tsv_match)
    _print_section("Layout bounds",                     bounds)
    _print_section("Bump candidate grid",               candidate)
    _print_section("Minimum spacing",                   spacing)
    _print_section("Window density",                    density)

    feasible = len(violations) == 0
    total    = len(violations)
    print(f"\n  → {'FEASIBLE' if feasible else f'INFEASIBLE  ({total} violation(s))'}")
    print("=" * 60)
    return feasible


# ─────────────────────────────────────────────────────────────────────────────
# 6. Routing feasibility checker
# ─────────────────────────────────────────────────────────────────────────────

def _octant(dx: float, dy: float) -> int:
    """Quantize direction vector (dx, dy) to one of 8 octants (0–7, step = 45°)."""
    return round(math.atan2(dy, dx) / (math.pi / 4)) % 8


def _proper_intersect(
    p1: Tuple[float, float], p2: Tuple[float, float],
    p3: Tuple[float, float], p4: Tuple[float, float],
) -> bool:
    """
    Return True iff segment p1-p2 and segment p3-p4 properly cross each other.

    "Properly" means the intersection is in the interior of both segments
    (shared endpoints are NOT counted as crossings so that adjacent segments
    sharing a waypoint do not false-positive).

    Uses the cross-product (2D) orientation test.
    """
    def _cross(o, a, b):
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

    def _on_seg(p, q, r):
        """True if point q lies strictly between p and r (collinear case)."""
        return (min(p[0], r[0]) < q[0] < max(p[0], r[0]) or
                min(p[1], r[1]) < q[1] < max(p[1], r[1]))

    d1 = _cross(p3, p4, p1)
    d2 = _cross(p3, p4, p2)
    d3 = _cross(p1, p2, p3)
    d4 = _cross(p1, p2, p4)

    if ((d1 > 0 and d2 < 0) or (d1 < 0 and d2 > 0)) and \
       ((d3 > 0 and d4 < 0) or (d3 < 0 and d4 > 0)):
        return True

    # Collinear cases: interior overlap (not just a shared endpoint)
    if d1 == 0 and _on_seg(p3, p1, p4):
        return True
    if d2 == 0 and _on_seg(p3, p2, p4):
        return True
    if d3 == 0 and _on_seg(p1, p3, p2):
        return True
    if d4 == 0 and _on_seg(p1, p4, p2):
        return True

    return False


def check_routing_feasibility(
    nets: List[RoutedNet],
    cfg: ValidatorCfg,
) -> List[Violation]:
    """
    Three routing feasibility checks applied to every net:

    1. Connectivity
       The route must be unbroken from C4 (m1[0]) to micro (mL[-1]).
       Specifically, the last point of layer mN must equal the first point of m(N+1).

    2. Turn angle
       At every interior waypoint, the change in direction must be exactly ±45°
       (one octant step on the octagonal grid).
       Interior angle at the bend must be 135° (left turn) or 225° (right turn).
       Forbidden: 90°, 180° (U-turn), 270°, and any other multiple of 45° ≥ 90°.

    3. Minimum advance before turn
       Every segment between consecutive waypoints must span at least 1 full grid step
       (max(|dx|, |dy|) >= pitch), so that at least one grid cell is traversed
       before any direction change.
    """
    violations: List[Violation] = []
    pitch = cfg.delta

    for net in nets:
        lns = net.layer_names()

        # ── Check 1: Connectivity ──────────────────────────────────────────
        for i, ln in enumerate(lns[:-1]):
            next_ln = lns[i + 1]
            end   = net.layers[ln][-1]
            start = net.layers[next_ln][0]
            if end != start:
                violations.append(Violation(
                    category='connectivity',
                    description=(
                        f"[{net.netname}] Route break between {ln} and {next_ln}: "
                        f"end={end} ≠ start={start}"
                    )
                ))

        # ── Checks 2, 3 & 4: per layer ────────────────────────────────────
        for ln in lns:
            pts   = net.layers[ln]
            n_pts = len(pts)
            if n_pts < 2:
                continue

            # ── Check 4: Layout bounds for every waypoint ─────────────────
            for k, (wx, wy) in enumerate(pts):
                if not (cfg.x_min <= wx <= cfg.x_max and cfg.y_min <= wy <= cfg.y_max):
                    violations.append(Violation(
                        category='bounds',
                        description=(
                            f"[{net.netname}][{ln}] Waypoint {k} ({wx}, {wy}) "
                            f"outside layout bounds "
                            f"x=[0, {cfg.W}], y=[0, {cfg.H}]."
                        )
                    ))

            # Build segment list: (dx, dy, start_waypoint_index)
            segs: List[Tuple[float, float, int]] = []
            for k in range(n_pts - 1):
                dx = pts[k + 1][0] - pts[k][0]
                dy = pts[k + 1][1] - pts[k][1]
                if dx == 0 and dy == 0:
                    violations.append(Violation(
                        category='routing',
                        description=(
                            f"[{net.netname}][{ln}] Duplicate consecutive waypoints "
                            f"at index {k}."
                        )
                    ))
                    continue
                segs.append((dx, dy, k))

            for s_idx, (dx, dy, pt_idx) in enumerate(segs):

                # ── Check: octagonal direction ────────────────────────────
                # Each segment must be horizontal (dy=0), vertical (dx=0),
                # or 45° diagonal (|dx| == |dy|).
                if not (dx == 0 or dy == 0 or abs(abs(dx) - abs(dy)) < 1e-9):
                    violations.append(Violation(
                        category='turn_angle',
                        description=(
                            f"[{net.netname}][{ln}] Segment {pt_idx}→{pt_idx+1}: "
                            f"non-octagonal direction (dx={dx:.1f}, dy={dy:.1f}). "
                            f"Must be horizontal, vertical, or 45-degree diagonal."
                        )
                    ))

                # ── Check 3: Minimum 1 grid step per segment ──────────────
                steps = max(abs(dx), abs(dy)) / pitch
                if steps < 1 - 1e-9:
                    violations.append(Violation(
                        category='min_advance',
                        description=(
                            f"[{net.netname}][{ln}] Segment {pt_idx}→{pt_idx + 1}: "
                            f"{steps:.2f} grid step(s) (minimum 1 required before "
                            f"any turn)."
                        )
                    ))

                # ── Check 2: Turn angle at waypoint pt_idx ────────────────
                # The turn is at the shared waypoint between segs[s_idx-1] and segs[s_idx].
                if s_idx == 0:
                    continue  # first segment has no incoming direction
                dx_prev, dy_prev, _ = segs[s_idx - 1]
                oct_in  = _octant(dx_prev, dy_prev)
                oct_out = _octant(dx, dy)
                turn    = (oct_out - oct_in) % 8
                # turn==0: same direction → redundant waypoint, format error
                # turn==1: +45° → interior 135° ✓
                # turn==7: -45° → interior 225° ✓
                # any other value: invalid turn angle
                if turn == 0:
                    violations.append(Violation(
                        category='routing',
                        description=(
                            f"[{net.netname}][{ln}] Redundant waypoint at index {pt_idx}: "
                            f"no direction change (waypoints must only appear at turns)."
                        )
                    ))
                elif turn not in (1, 7):
                    violations.append(Violation(
                        category='turn_angle',
                        description=(
                            f"[{net.netname}][{ln}] Invalid turn at waypoint {pt_idx}: "
                            f"direction change = {turn * 45}° "
                            f"(only ±45° allowed; interior angle must be 135° or 225°)."
                        )
                    ))

        # ── Check 5: Wire self-crossing (same net, same layer) ────────────
        for ln in lns:
            pts   = net.layers[ln]
            n_pts = len(pts)
            if n_pts < 2:
                continue

            # Build list of segments as coordinate pairs
            seg_pts = [
                (pts[k], pts[k + 1])
                for k in range(n_pts - 1)
                if not (pts[k][0] == pts[k + 1][0] and pts[k][1] == pts[k + 1][1])
            ]

            # Check non-adjacent pairs (j >= i+2) for proper intersection
            for i in range(len(seg_pts)):
                for j in range(i + 2, len(seg_pts)):
                    p1, p2 = seg_pts[i]
                    p3, p4 = seg_pts[j]
                    if _proper_intersect(p1, p2, p3, p4):
                        violations.append(Violation(
                            category='routing',
                            description=(
                                f"[{net.netname}][{ln}] Self-crossing: "
                                f"segment {i}→{i+1} crosses segment {j}→{j+1}."
                            )
                        ))

    # ── Check 6: Layer usage ratio ─────────────────────────────────────────
    # For each layer with a non-zero ratio constraint, the layer's total routing
    # length must not exceed ratio × total_routing_length.
    if cfg.rho:
        # Accumulate routing length per layer across all nets.
        layer_lengths: Dict[str, float] = {}
        for net in nets:
            for ln in net.layer_names():
                pts = net.layers[ln]
                seg_len = sum(
                    math.sqrt((pts[k+1][0]-pts[k][0])**2 + (pts[k+1][1]-pts[k][1])**2)
                    for k in range(len(pts) - 1)
                )
                layer_lengths[ln] = layer_lengths.get(ln, 0.0) + seg_len

        total_length = sum(layer_lengths.values())

        if total_length > 0:
            for ln, ratio in cfg.rho.items():
                if ratio is None:
                    continue   # unconstrained layer
                actual_len   = layer_lengths.get(ln, 0.0)
                actual_ratio = actual_len / total_length
                if actual_ratio > ratio + 1e-9:
                    violations.append(Violation(
                        category='layer_usage',
                        description=(
                            f"Layer {ln.upper()}: usage ratio = {actual_ratio:.4f} "
                            f"(limit = {ratio:.4f}). "
                            f"Length = {actual_len:.2f} / total = {total_length:.2f}."
                        )
                    ))

    return violations


def print_routing_report(violations: List[Violation]) -> bool:
    """Print a structured routing feasibility report. Returns True if feasible."""
    connectivity     = [v for v in violations if v.category == 'connectivity']
    bounds           = [v for v in violations if v.category == 'bounds']
    turn_angle       = [v for v in violations if v.category == 'turn_angle']
    min_advance      = [v for v in violations if v.category == 'min_advance']
    routing_spacing  = [v for v in violations if v.category == 'routing_spacing']
    layer_usage      = [v for v in violations if v.category == 'layer_usage']
    group_deviation  = [v for v in violations if v.category == 'group_deviation']
    other            = [v for v in violations if v.category == 'routing']

    print("\n" + "=" * 60)
    print("[Routing Feasibility]")
    _print_section("Connectivity (C4 → micro)",           connectivity)
    _print_section("Layout bounds",                     bounds)
    _print_section("Turn angles (±45° only)",           turn_angle)
    _print_section("Min advance (≥1 step before turn)", min_advance)
    _print_section("Routing spacing",                   routing_spacing)
    _print_section("Layer usage ratio",                 layer_usage)
    _print_section("Group length deviation",            group_deviation)
    if other:
        _print_section("Other routing issues", other)

    feasible = len(violations) == 0
    total    = len(violations)
    print(f"\n  → {'FEASIBLE' if feasible else f'INFEASIBLE  ({total} violation(s))'}")
    print("=" * 60)
    return feasible


# ─────────────────────────────────────────────────────────────────────────────
# 7. Routing spacing checker (line-to-component and line-to-line)
# ─────────────────────────────────────────────────────────────────────────────

def _seg_dist_grid(
    ax: float, ay: float, bx: float, by: float,
    ix0: int, ix1: int, iy0: int, iy1: int,
    x_min: float, y_min: float, gs: float,
) -> np.ndarray:
    """
    Return a 2D float array (shape: ix1-ix0+1, iy1-iy0+1) where each element
    is the Euclidean distance from that grid cell's center to segment AB.
    """
    cx = x_min + np.arange(ix0, ix1 + 1) * gs
    cy = y_min + np.arange(iy0, iy1 + 1) * gs
    CX, CY = np.meshgrid(cx, cy, indexing='ij')
    dx, dy = bx - ax, by - ay
    len_sq = dx * dx + dy * dy
    if len_sq == 0:
        return np.sqrt((CX - ax) ** 2 + (CY - ay) ** 2)
    t  = np.clip(((CX - ax) * dx + (CY - ay) * dy) / len_sq, 0.0, 1.0)
    PX = ax + t * dx
    PY = ay + t * dy
    return np.sqrt((CX - PX) ** 2 + (CY - PY) ** 2)


def _paint_circle(
    grid: np.ndarray,
    cx: float, cy: float, radius: float,
    x_min: float, y_min: float, gs: float,
) -> None:
    """Paint all grid cells whose centers lie within `radius` of (cx, cy)."""
    NX, NY = grid.shape
    margin = int(radius / gs) + 1
    ix_c   = int((cx - x_min) / gs)
    iy_c   = int((cy - y_min) / gs)
    ix0    = max(0, ix_c - margin)
    ix1    = min(NX - 1, ix_c + margin)
    iy0    = max(0, iy_c - margin)
    iy1    = min(NY - 1, iy_c + margin)
    cell_x = x_min + np.arange(ix0, ix1 + 1) * gs
    cell_y = y_min + np.arange(iy0, iy1 + 1) * gs
    XX, YY = np.meshgrid(cell_x, cell_y, indexing='ij')
    grid[ix0:ix1 + 1, iy0:iy1 + 1] |= (np.sqrt((XX - cx) ** 2 + (YY - cy) ** 2) <= radius)


def check_routing_spacing(
    nets: List[RoutedNet],
    components: List[Component],
    cfg: ValidatorCfg,
    candidates: Optional[List[Tuple[int, int]]] = None,
    micro_positions: Optional[List[Tuple[int, int]]] = None,
) -> List[Violation]:
    """
    Routing spacing check using per-type occupancy grids, one set per layer.

    Component grids (initialized once, static):
      micro_grid[l] — cells within micro_radius of a micro on layer l
                           This includes both routed micro bumps (with nets) AND
                           dummy micro bumps (unselected positions from micro CSV).
      via_grid[l]        — cells within via_radius of a via on layer l
      C4_grid[l]    — cells within C4_radius of a C4 on layer l
                           This includes both selected C4 bumps (with nets) AND
                           dummy bumps (unselected candidates masked at C4_radius).

    Line grid (updated sequentially as nets are processed):
      wire_grid[l] — cells within wire_width/2 of routing wires placed so far

    For each routing segment, the distance D from every cell in the bounding
    box to the segment centerline is computed. A violation is raised when:
      micro_grid[cell] and D < wire_width/2 + micro_wire → micro_wire violation
        (applies to both real micro bumps and dummy micro bumps)
      via_grid[cell]        and D < wire_width/2 + via_wire        → via_wire violation
      C4_grid[cell]    and D < wire_width/2 + C4_wire    → C4_wire violation
        (applies to both real C4 bumps and dummy C4 bumps)
      wire_grid[cell]       and D < wire_width/2 + wire_wire       → wire_wire violation
        (wire_grid already encodes wire_width/2 of the existing wire)

    Same-net segments are not checked against each other (a net's segments are
    painted onto wire_grid only after all its segments on that layer are checked).
    Checks are performed per-layer (same-layer pairs only).
    """
    gs   = cfg.delta
    mw2  = cfg.delta / 2.0
    L    = cfg.L
    NXg  = int((cfg.x_max - cfg.x_min) / gs) + 1
    NYg  = int((cfg.y_max - cfg.y_min) / gs) + 1

    # ── Initialize component grids ────────────────────────────────────────
    micro_grids  = [np.zeros((NXg, NYg), dtype=bool) for _ in range(L)]
    via_grids  = [np.zeros((NXg, NYg), dtype=bool) for _ in range(L)]
    C4_grids = [np.zeros((NXg, NYg), dtype=bool) for _ in range(L)]
    wire_grids = [np.zeros((NXg, NYg), dtype=bool) for _ in range(L)]

    for c in components:
        for layer_1b in c.layers:
            li = layer_1b - 1
            if li < 0 or li >= L:
                continue
            if c.kind == 'micro':
                _paint_circle(micro_grids[li],  c.x, c.y, c.radius, cfg.x_min, cfg.y_min, gs)
            elif c.kind == 'via':
                _paint_circle(via_grids[li],  c.x, c.y, c.radius, cfg.x_min, cfg.y_min, gs)
            elif c.kind == 'C4':
                _paint_circle(C4_grids[li], c.x, c.y, c.radius, cfg.x_min, cfg.y_min, gs)

    # ── Dummy micro bump masking: unrouted micro bump positions → top layer ──────
    # Every position in the micro CSV is physically occupied by either a
    # routed signal micro bump or a dummy micro bump of the same size.
    # Both must be respected as obstacles for wires (micro_wire spacing).
    if micro_positions:
        selected_micro_pos = {(int(c.x), int(c.y)) for c in components if c.kind == 'micro'}
        li_last = L - 1   # micro bumps reside on the top (last) routing layer
        for (mx, my) in micro_positions:
            if (mx, my) not in selected_micro_pos:
                _paint_circle(micro_grids[li_last], float(mx), float(my),
                              cfg.micro_radius, cfg.x_min, cfg.y_min, gs)

    # ── Dummy C4 bump masking: unselected C4 candidates → painted on bottom layer ──
    # Every C4 candidate position is physically occupied by either a routed C4 bump
    # or a dummy bump of the same size.  Both must be respected as obstacles for wires.
    if candidates:
        selected_C4_pos = {(int(c.x), int(c.y)) for c in components if c.kind == 'C4'}
        li_first = 0   # C4 bumps reside on the bottom (first) routing layer
        for (cx, cy) in candidates:
            if (cx, cy) not in selected_C4_pos:
                _paint_circle(C4_grids[li_first], float(cx), float(cy),
                              cfg.C4_radius, cfg.x_min, cfg.y_min, gs)

    # ── Spacing thresholds ────────────────────────────────────────────────
    # All spacing values are center-to-center distances.
    # Component grids are painted at the component radius.
    # A wire segment (half-width mw2) violates spacing when:
    #   dist(wire_centerline, comp_center) < sp[key]
    # In the grid check: component cell is painted if within comp_radius of comp_center;
    #   violation fires if D (dist from wire centerline to cell) < thr.
    #   Geometrically: fires when dist(wire_cl, comp_center) < comp_radius + thr.
    #   So set thr = sp[key] - comp_radius to enforce center-to-center >= sp[key].
    # For wire-wire: wire_grid is painted within mw2 of centerline;
    #   thr = sp['wire_wire'] - mw2 so fires when dist < mw2 + thr = sp['wire_wire'].
    sp         = cfg.spacing
    thr_micro  = sp.get('sc_wire_micro', 0.0) - cfg.micro_radius
    thr_via    = sp.get('sc_wire_via',   0.0) - cfg.via_radius
    thr_C4     = sp.get('sc_wire_C4',    0.0) - cfg.C4_radius
    thr_wire   = sp.get('sc_wire_wire',  0.0) - mw2
    max_thr    = max(thr_micro, thr_via, thr_C4, thr_wire)

    violations: List[Violation] = []

    # ── Process nets sequentially ─────────────────────────────────────────
    for net in nets:
        # Build same-net exclusion masks (own micro/via/C4 must not trigger
        # spacing violations against that net's own wire segments).
        own_micro_mask  = [np.zeros((NXg, NYg), dtype=bool) for _ in range(L)]
        own_via_mask  = [np.zeros((NXg, NYg), dtype=bool) for _ in range(L)]
        own_C4_mask = [np.zeros((NXg, NYg), dtype=bool) for _ in range(L)]
        own_wire_mask = [np.zeros((NXg, NYg), dtype=bool) for _ in range(L)]
        for c in components:
            if c.netname != net.netname:
                continue
            for layer_1b in c.layers:
                li_c = layer_1b - 1
                if 0 <= li_c < L:
                    if c.kind == 'micro':
                        _paint_circle(own_micro_mask[li_c],  c.x, c.y, c.radius,
                                      cfg.x_min, cfg.y_min, gs)
                    elif c.kind == 'via':
                        _paint_circle(own_via_mask[li_c],  c.x, c.y, c.radius,
                                      cfg.x_min, cfg.y_min, gs)
                    elif c.kind == 'C4':
                        _paint_circle(own_C4_mask[li_c], c.x, c.y, c.radius,
                                      cfg.x_min, cfg.y_min, gs)
        # Pre-paint all wire segments of this net into own_wire_mask
        for ln2 in net.layer_names():
            li2  = int(ln2[1:]) - 1
            pts2 = net.layers[ln2]
            for k in range(len(pts2) - 1):
                ax2, ay2 = pts2[k]
                bx2, by2 = pts2[k + 1]
                if ax2 == bx2 and ay2 == by2:
                    continue
                pad2 = int(mw2 / gs) + 1
                ix0_ = max(0,       int((min(ax2, bx2) - cfg.x_min) / gs) - pad2)
                ix1_ = min(NXg - 1, int((max(ax2, bx2) - cfg.x_min) / gs) + pad2)
                iy0_ = max(0,       int((min(ay2, by2) - cfg.y_min) / gs) - pad2)
                iy1_ = min(NYg - 1, int((max(ay2, by2) - cfg.y_min) / gs) + pad2)
                if ix1_ < ix0_ or iy1_ < iy0_:
                    continue  # segment lies entirely outside layout bounds; caught separately by the bounds check
                D2   = _seg_dist_grid(ax2, ay2, bx2, by2, ix0_, ix1_, iy0_, iy1_,
                                      cfg.x_min, cfg.y_min, gs)
                own_wire_mask[li2][ix0_:ix1_+1, iy0_:iy1_+1] |= (D2 <= mw2)

        for ln in net.layer_names():
            li  = int(ln[1:]) - 1
            pts = net.layers[ln]
            if len(pts) < 2:
                continue

            segs = [
                (pts[k][0], pts[k][1], pts[k+1][0], pts[k+1][1], k)
                for k in range(len(pts) - 1)
                if not (pts[k][0] == pts[k+1][0] and pts[k][1] == pts[k+1][1])
            ]

            # Check all segments of this net against existing grids
            for ax, ay, bx, by, seg_idx in segs:
                pad = int(max_thr / gs) + 1
                ix0 = max(0,       int((min(ax, bx) - cfg.x_min) / gs) - pad)
                ix1 = min(NXg - 1, int((max(ax, bx) - cfg.x_min) / gs) + pad)
                iy0 = max(0,       int((min(ay, by) - cfg.y_min) / gs) - pad)
                iy1 = min(NYg - 1, int((max(ay, by) - cfg.y_min) / gs) + pad)
                if ix1 < ix0 or iy1 < iy0:
                    continue  # segment lies entirely outside layout bounds; caught separately by the bounds check

                D = _seg_dist_grid(ax, ay, bx, by, ix0, ix1, iy0, iy1,
                                   cfg.x_min, cfg.y_min, gs)

                def _chk(grid: np.ndarray, own: np.ndarray,
                         thr: float, rule: str) -> None:
                    foreign = grid[ix0:ix1+1, iy0:iy1+1] & ~own[ix0:ix1+1, iy0:iy1+1]
                    if np.any(foreign & (D < thr)):
                        violations.append(Violation(
                            category='routing_spacing',
                            description=(
                                f"[{net.netname}][{ln}] segment {seg_idx}→{seg_idx+1}: "
                                f"{rule} spacing violated."
                            )
                        ))

                _chk(micro_grids[li],  own_micro_mask[li],  thr_micro,  'sc_wire_micro')
                _chk(via_grids[li],  own_via_mask[li],  thr_via,  'sc_wire_via')
                _chk(C4_grids[li], own_C4_mask[li], thr_C4, 'sc_wire_C4')
                _chk(wire_grids[li], own_wire_mask[li], thr_wire, 'sc_wire_wire')


            # Paint this net's segments onto wire_grid (after all checks)
            for ax, ay, bx, by, _ in segs:
                pad = int(mw2 / gs) + 1
                ix0 = max(0,       int((min(ax, bx) - cfg.x_min) / gs) - pad)
                ix1 = min(NXg - 1, int((max(ax, bx) - cfg.x_min) / gs) + pad)
                iy0 = max(0,       int((min(ay, by) - cfg.y_min) / gs) - pad)
                iy1 = min(NYg - 1, int((max(ay, by) - cfg.y_min) / gs) + pad)
                if ix1 < ix0 or iy1 < iy0:
                    continue  # segment lies entirely outside layout bounds; caught separately by the bounds check
                D   = _seg_dist_grid(ax, ay, bx, by, ix0, ix1, iy0, iy1,
                                     cfg.x_min, cfg.y_min, gs)
                wire_grids[li][ix0:ix1+1, iy0:iy1+1] |= (D <= mw2)

    return violations


# ─────────────────────────────────────────────────────────────────────────────
# 8. Routing metrics
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class NetMetrics:
    netname: str
    routing_length: float          # total Euclidean wire length across all layers
    bend_count: int                # number of ±45° direction changes across all layers
    layer_lengths: Dict[str, float] = None   # per-layer wire length (e.g. {'m1': ..., 'm2': ...})

    def __post_init__(self):
        if self.layer_lengths is None:
            self.layer_lengths = {}


def compute_routing_metrics(nets: List[RoutedNet]) -> List[NetMetrics]:
    """
    Compute per-net routing metrics:
      - routing_length: sum of Euclidean segment lengths over all layers.
      - bend_count: number of valid turns (±45°, i.e. octant change of 1 or 7)
                    across all layers. Redundant waypoints (turn==0) and invalid
                    turns are not counted — those are caught by the feasibility check.
      - layer_lengths: per-layer breakdown of routing_length.
    """
    metrics: List[NetMetrics] = []
    for net in nets:
        total_length  = 0.0
        total_bends   = 0
        layer_lengths: Dict[str, float] = {}

        for ln in net.layer_names():
            pts = net.layers[ln]
            ln_len = 0.0

            for k in range(len(pts) - 1):
                dx = pts[k + 1][0] - pts[k][0]
                dy = pts[k + 1][1] - pts[k][1]
                ln_len += math.sqrt(dx * dx + dy * dy)

            total_length += ln_len
            layer_lengths[ln] = ln_len

            # Bend count: ±45° direction changes at interior waypoints
            segs = [
                (pts[k + 1][0] - pts[k][0], pts[k + 1][1] - pts[k][1])
                for k in range(len(pts) - 1)
                if not (pts[k + 1][0] == pts[k][0] and pts[k + 1][1] == pts[k][1])
            ]
            for s in range(1, len(segs)):
                dx_prev, dy_prev = segs[s - 1]
                dx_cur,  dy_cur  = segs[s]
                turn = (_octant(dx_cur, dy_cur) - _octant(dx_prev, dy_prev)) % 8
                if turn in (1, 7):
                    total_bends += 1

        metrics.append(NetMetrics(
            netname        = net.netname,
            routing_length = total_length,
            bend_count     = total_bends,
            layer_lengths  = layer_lengths,
        ))
    return metrics


def check_group_deviation(
    metrics: List[NetMetrics],
    cfg: ValidatorCfg,
) -> List[Violation]:
    """
    For each group G1…G{num_group}, check that the routing length spread
    within the group satisfies:
        (max_length - min_length) / min_length  <=  group_deviation[group]

    Net-to-group mapping is derived from the netname prefix (e.g. 'G1_0001' → 'G1').
    Groups with fewer than 2 nets are skipped (deviation is undefined).
    """
    violations: List[Violation] = []

    # Build per-group length lists
    group_lengths: Dict[str, List[float]] = {
        f'G{i}': [] for i in range(1, cfg.K + 1)
    }
    for m in metrics:
        group = m.netname.split('_')[0]   # 'G1_0001' → 'G1'
        if group in group_lengths:
            group_lengths[group].append(m.routing_length)

    for group, lengths in group_lengths.items():
        if len(lengths) < 2:
            continue
        limit = cfg.sigma.get(group)
        if limit is None:
            continue   # no constraint defined for this group

        max_len  = max(lengths)
        min_len  = min(lengths)
        if min_len <= 0:
            continue   # degenerate case; skip

        deviation = (max_len - min_len) / min_len
        if deviation > limit + 1e-9:
            violations.append(Violation(
                category='group_deviation',
                description=(
                    f"{group}: (max - min) / min = {deviation:.4f} "
                    f"(limit = {limit:.4f}). "
                    f"max = {max_len:.2f}, min = {min_len:.2f}  "
                    f"[{len(lengths)} nets]."
                )
            ))

    return violations


# ─────────────────────────────────────────────────────────────────────────────
# 10. Visualization
# ─────────────────────────────────────────────────────────────────────────────

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

_LAYER_COLORS = [
    'royalblue', 'darkorange', 'green', 'red',
    'purple', 'brown', 'deeppink', 'gray', 'olive', 'cyan',
]

def _layer_color(layer_idx: int) -> str:
    return _LAYER_COLORS[layer_idx % len(_LAYER_COLORS)]

def _group_color(netname: str) -> tuple:
    group = netname.split('_')[0]
    try:
        idx = int(group[1:]) - 1
    except (ValueError, IndexError):
        idx = 0
    return plt.get_cmap('tab10')(idx % 10)

def _save_fig(fig: plt.Figure, out_path: str, label: str) -> None:
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    print(f"  Saved {label} -> '{out_path}'")
    plt.close(fig)

def plot_per_layer(nets: List[RoutedNet], num_layers: int, out_dir: str, base: str) -> None:
    for layer_idx in range(num_layers):
        ln = f'm{layer_idx + 1}'
        fig, ax = plt.subplots(figsize=(8, 7))
        ax.set_title(f'Layer {layer_idx + 1}', fontsize=13)
        ax.grid(True, linestyle='--', linewidth=0.3, alpha=0.5)
        ax.set_aspect('equal', adjustable='datalim')
        for net in nets:
            if ln not in net.layers:
                continue
            pts   = net.layers[ln]
            color = _group_color(net.netname)
            xs    = [p[0] for p in pts]
            ys    = [p[1] for p in pts]
            ax.plot(xs, ys, color=color, linewidth=0.8, alpha=0.7)
            # bump/via 끝점에만 마커 찍기 (중간 꺾이는 점은 와이어 형태일 뿐, 실제 컴포넌트가 아님)
            ax.scatter([xs[0], xs[-1]], [ys[0], ys[-1]], color=color, s=8, zorder=3)
        plt.tight_layout()
        _save_fig(fig, os.path.join(out_dir, f'{base}_layer{layer_idx + 1}.png'),
                  f'layer {layer_idx + 1}')

def plot_3d(nets: List[RoutedNet], num_layers: int, out_path: str) -> None:
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
    import numpy as np
    fig = plt.figure(figsize=(12, 9))
    ax  = fig.add_subplot(111, projection='3d')
    layer_names = [f'm{i}' for i in range(1, num_layers + 1)]
    all_x: List[float] = []
    all_y: List[float] = []
    for net in nets:
        for ln in layer_names:
            if ln in net.layers:
                all_x.extend(p[0] for p in net.layers[ln])
                all_y.extend(p[1] for p in net.layers[ln])
    x_range = [min(all_x), max(all_x)] if all_x else [0, 1]
    y_range = [min(all_y), max(all_y)] if all_y else [0, 1]
    for layer_idx, ln in enumerate(layer_names):
        z     = float(layer_idx)
        color = _layer_color(layer_idx)
        xx, yy = np.meshgrid(x_range, y_range)
        zz = np.full_like(xx, z, dtype=float)
        ax.plot_surface(xx, yy, zz, alpha=0.07, color=color, zorder=1)
        for net in nets:
            if ln not in net.layers:
                continue
            pts = net.layers[ln]
            xs  = [p[0] for p in pts]
            ys  = [p[1] for p in pts]
            zs  = [z] * len(pts)
            ax.plot(xs, ys, zs, color=color, linewidth=0.7, alpha=0.7)
    for net in nets:
        for i in range(num_layers - 1):
            ln = layer_names[i]
            if ln not in net.layers or len(net.layers[ln]) < 1:
                continue
            vx, vy = net.layers[ln][-1]
            ax.plot([vx, vx], [vy, vy], [float(i), float(i + 1)],
                    color='gray', linewidth=0.5, alpha=0.35)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Layer')
    ax.set_zticks(list(range(num_layers)))
    ax.set_zticklabels([f'M{i + 1}' for i in range(num_layers)])
    legend_handles = [
        mpatches.Patch(color=_layer_color(i), label=f'Layer {i + 1}')
        for i in range(num_layers)
    ]
    ax.legend(handles=legend_handles, loc='upper left', fontsize=8)
    plt.tight_layout()
    _save_fig(fig, out_path, '3D')

def plot_projection(nets: List[RoutedNet], num_layers: int, out_path: str) -> None:
    fig, ax = plt.subplots(figsize=(11, 10))
    ax.set_title('Top-down Projection (all layers)', fontsize=13)
    ax.set_aspect('equal', adjustable='datalim')
    ax.grid(True, linestyle='--', linewidth=0.3, alpha=0.5)
    for layer_idx in range(num_layers):
        ln    = f'm{layer_idx + 1}'
        color = _layer_color(layer_idx)
        for net in nets:
            if ln not in net.layers:
                continue
            pts = net.layers[ln]
            xs  = [p[0] for p in pts]
            ys  = [p[1] for p in pts]
            ax.plot(xs, ys, color=color, linewidth=0.6, alpha=0.6)
            # bump/via 끝점에만 마커 찍기 (중간 꺾이는 점은 와이어 형태일 뿐, 실제 컴포넌트가 아님)
            ax.scatter([xs[0], xs[-1]], [ys[0], ys[-1]], color=color, s=4, zorder=3)
    legend_handles = [
        mpatches.Patch(color=_layer_color(i), label=f'layer{i + 1}')
        for i in range(num_layers)
    ]
    ax.legend(handles=legend_handles, loc='upper right', fontsize=9)
    plt.tight_layout()
    _save_fig(fig, out_path, 'projection')


def save_metrics_csv(metrics: List[NetMetrics], out_path: str) -> None:
    """Save per-net routing metrics to a CSV file.

    Columns: netname, routing_length, bend_count, m1_length, m2_length, ...
    Summary rows appended after per-net rows:
      TOTAL          — sum across all nets
      GROUP_G{k}     — sum across nets of group Gk
      LAYER_M{l}     — total routing length on layer Ml
    """
    # Collect all layer names present
    all_layers = sorted(
        {ln for m in metrics for ln in m.layer_lengths},
        key=lambda s: int(s[1:])
    )
    layer_cols = [f'{ln}_length' for ln in all_layers]

    total_len   = sum(m.routing_length for m in metrics)
    total_bends = sum(m.bend_count     for m in metrics)

    # Per-group aggregation
    import collections as _col
    group_len   = _col.defaultdict(float)
    group_bends = _col.defaultdict(int)
    layer_total = _col.defaultdict(float)
    for m in metrics:
        g = m.netname.split('_')[0]
        group_len[g]   += m.routing_length
        group_bends[g] += m.bend_count
        for ln, v in m.layer_lengths.items():
            layer_total[ln] += v

    with open(out_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['netname', 'routing_length', 'bend_count'] + layer_cols)
        for m in metrics:
            row = [m.netname, f'{m.routing_length:.4f}', m.bend_count]
            row += [f'{m.layer_lengths.get(ln, 0.0):.4f}' for ln in all_layers]
            writer.writerow(row)
        # TOTAL row
        total_row = ['TOTAL', f'{total_len:.4f}', total_bends]
        total_row += [f'{layer_total.get(ln, 0.0):.4f}' for ln in all_layers]
        writer.writerow(total_row)
        # Per-group rows
        for g in sorted(group_len):
            row = [f'GROUP_{g}', f'{group_len[g]:.4f}', group_bends[g]]
            row += [''] * len(all_layers)
            writer.writerow(row)
        # Per-layer rows
        for ln in all_layers:
            row = [f'LAYER_{ln.upper()}', f'{layer_total[ln]:.4f}', '']
            row += [f'{layer_total.get(l, 0.0):.4f}' if l == ln else '' for l in all_layers]
            writer.writerow(row)

    print(f"\n[Metrics] Saved {len(metrics)} nets → '{out_path}'")
    print(f"          Total routing length = {total_len:.2f}, "
          f"Total bends = {total_bends}")


# ── Category display names (for the JSON "check" field) ──────────────────────
_CATEGORY_LABEL: Dict[str, str] = {
    'micro_match':     'micro coordinate & net completeness',
    'bounds':          'layout bounds',
    'candidate':       'bump candidate grid',
    'spacing':         'minimum spacing',
    'density':         'window density',
    'connectivity':    'connectivity (C4 → micro)',
    'turn_angle':      'turn angles (±45° only)',
    'min_advance':     'min advance (≥1 step before turn)',
    'routing_spacing': 'routing spacing',
    'layer_usage':     'layer usage ratio',
    'group_deviation': 'group length deviation',
    'routing':         'routing (other)',
}


def save_summary_json(
    placement_violations: List[Violation],
    routing_violations:   List[Violation],
    out_path: str,
) -> None:
    """
    Save all constraint violations to a single JSON file.

    Structure
    ---------
    {
      "feasible": bool,
      "total_violations": int,
      "placement": {
        "feasible": bool,
        "total_violations": int,
        "by_category": {
          "<category>": {
            "check": "<human-readable name>",
            "count": int,
            "violations": [ {"description": "..."}, ... ]
          },
          ...
        }
      },
      "routing": { ... same shape as placement ... }
    }
    """
    def _group(vs: List[Violation]) -> Dict[str, Any]:
        by_cat: Dict[str, List[str]] = {}
        for v in vs:
            by_cat.setdefault(v.category, []).append(v.description)
        return {
            cat: {
                "check":      _CATEGORY_LABEL.get(cat, cat),
                "count":      len(descs),
                "violations": [{"description": d} for d in descs],
            }
            for cat, descs in by_cat.items()
        }

    placement_ok = len(placement_violations) == 0
    routing_ok   = len(routing_violations)   == 0
    total        = len(placement_violations) + len(routing_violations)

    doc: Dict[str, Any] = {
        "feasible":         placement_ok and routing_ok,
        "total_violations": total,
        "placement": {
            "feasible":         placement_ok,
            "total_violations": len(placement_violations),
            "by_category":      _group(placement_violations),
        },
        "routing": {
            "feasible":         routing_ok,
            "total_violations": len(routing_violations),
            "by_category":      _group(routing_violations),
        },
    }

    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(doc, f, indent=2, ensure_ascii=False)

    print(f"\n[Summary] {total} violation(s) written → '{out_path}'"
          f"  (placement: {len(placement_violations)}, "
          f"routing: {len(routing_violations)})")


# ─────────────────────────────────────────────────────────────────────────────
# 9. CLI
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    ap = argparse.ArgumentParser(
        description="Validate routing output JSON (placement + routing feasibility)."
    )
    ap.add_argument('-r', '--routing', required=True, help="routing output JSON path")
    ap.add_argument('-c', '--config', default=None,
                     help="input.yaml path (default: <routing dir>/input_*.yaml, else <routing dir>/../input.yaml)")
    ap.add_argument('-d', '--data-dir', default=None,
                     help="directory containing C4_candidate.csv and micro_coordinate.csv "
                          "(default: same directory as routing JSON)")
    args = ap.parse_args()

    # ── Derive file paths from routing JSON location ──────────────────────
    out_dir              = os.path.dirname(os.path.abspath(args.routing))
    data_dir             = os.path.abspath(args.data_dir) if args.data_dir else out_dir
    if args.config:
        config_path = args.config
    else:
        local_yamls = glob.glob(os.path.join(out_dir, 'input_*.yaml'))
        config_path = local_yamls[0] if local_yamls else os.path.join(out_dir, '..', 'input.yaml')
    candidate_path       = os.path.join(data_dir, 'C4_candidate.csv')
    micro_coord_path     = os.path.join(data_dir, 'micro_coordinate.csv')
    metrics_csv          = os.path.join(out_dir, 'routing_metrics.csv')
    summary_json         = os.path.join(out_dir, 'summary.json')

    # ── Load config ───────────────────────────────────────────────────────
    cfg = load_validator_cfg(config_path)
    print(f"[Config] L={cfg.L}, pitch={cfg.pitch}")
    print(f"         micro_r={cfg.micro_radius}, C4_r={cfg.C4_radius}, via_r={cfg.via_radius}")
    print(f"         spacing={cfg.spacing}")

    # ── Load routing output ───────────────────────────────────────────────
    nets = load_routing_json(args.routing, cfg.L)
    print(f"\n[Routing] {len(nets)} nets loaded from '{args.routing}'")
    _routing_missing: List[Violation] = []
    if len(nets) != cfg.total_signals:
        missing = cfg.total_signals - len(nets)
        print(f"  [WARNING] Net count mismatch: output has {len(nets)} nets, "
              f"but input.yaml defines {cfg.total_signals} signals "
              f"(G1–G{cfg.K}). {missing} net(s) not routed.")
        _routing_missing.append(Violation(
            category='routing_incomplete',
            description=f"{missing} net(s) missing from routing output "
                        f"(expected {cfg.total_signals}, got {len(nets)})."
        ))

    # ── Extract components ────────────────────────────────────────────────
    components = extract_components(nets, cfg)
    micro_cnt = sum(1 for c in components if c.kind == 'micro')
    via_cnt        = sum(1 for c in components if c.kind == 'via')
    bump_cnt       = sum(1 for c in components if c.kind == 'C4')
    print(f"[Components] micro={micro_cnt}, via={via_cnt}, C4={bump_cnt}  "
          f"(total={len(components)})")

    # ── Load C4 candidates ──────────────────────────────────────────────
    candidates = load_C4_candidates(candidate_path)
    print(f"[Candidates] {len(candidates)} C4 candidate positions loaded "
          f"from '{candidate_path}'")

    # ── Load micro coordinates (real + dummy) ───────────────────────────────
    micro_map = load_micro_coordinates(micro_coord_path)
    micro_cnt_real  = sum(1 for k in micro_map if not k.startswith('dummy_'))
    micro_cnt_dummy = len(micro_map) - micro_cnt_real
    print(f"[micro map]    {len(micro_map)} entries loaded from '{micro_coord_path}' "
          f"(real={micro_cnt_real}, dummy={micro_cnt_dummy})")

    # ── Placement feasibility ─────────────────────────────────────────────
    placement_violations = check_placement_feasibility(
        components, cfg, candidates, nets=nets, micro_map=micro_map
    )
    placement_ok = print_placement_report(placement_violations)

    # ── Routing metrics → CSV ────────────────────────────────────────────
    metrics = compute_routing_metrics(nets)
    save_metrics_csv(metrics, metrics_csv)

    # ── Routing feasibility (including spacing and group deviation) ───────
    routing_violations   = check_routing_feasibility(nets, cfg)
    all_micro_positions = list(micro_map.values())
    spacing_violations   = check_routing_spacing(
        nets, components, cfg,
        candidates=candidates,
        micro_positions=all_micro_positions,
    )
    deviation_violations = check_group_deviation(metrics, cfg)
    all_routing_violations = _routing_missing + routing_violations + spacing_violations + deviation_violations
    routing_ok = print_routing_report(all_routing_violations)

    # ── Save summary JSON ─────────────────────────────────────────────────
    save_summary_json(placement_violations, all_routing_violations, summary_json)

    # ── Visualization ─────────────────────────────────────────────────────
    base = os.path.splitext(os.path.basename(args.routing))[0]
    plot_per_layer(nets, cfg.L, out_dir, base)
    plot_3d(nets, cfg.L,
            out_path=os.path.join(out_dir, f'{base}_3d.png'))
    plot_projection(nets, cfg.L,
                    out_path=os.path.join(out_dir, f'{base}_projection.png'))
    print(f"\n[Visualization] Done. Plots saved to '{out_dir}'")


if __name__ == '__main__':
    main()
