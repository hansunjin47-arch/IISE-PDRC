"""
MILP implementation of supplementary_formulation_revised.tex.

C4 candidates on the M1 layer are selected as sources; fixed micro bumps
on the ML layer are sinks. Grid edges between layers point upward.
The flow/depot structure is retained without subtour-elimination constraints.
No diagonal self-crossing constraints are added, matching the supplement.
Disconnected selected edges are reported separately in export diagnostics.

All spacing constraints use the supplement's grid-node neighborhoods,
with Euclidean physical distance strictly below the threshold (equality allowed).
They are explicit linear constraints; validator.py is an independent checker,
not a callback defining the feasible region. Raster validation need not be
equivalent to these node-neighborhood constraints.

The via variable z[n, node] counts incoming and outgoing vertical edges.
Rho bounds aggregate length over all nets; null rho means unconstrained.
Sigma uses (max-min)/min. The bend penalty eta must be positive.
Internal coordinates are (grid_x, grid_y, zero-based layer).
"""

from __future__ import annotations

import csv
import json
import math
import os
from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, List, Tuple

import yaml

import gurobipy as gp
from gurobipy import GRB

Node = Tuple[int, int, int]          # (ix, iy, layer)  layer 0 = M1
XY = Tuple[int, int]                 # grid index
DEPOT: Node = (-1, -1, -1)

DIRS: List[Tuple[int, int]] = [
    (1, 0), (1, 1), (0, 1), (-1, 1), (-1, 0), (-1, -1), (0, -1), (1, -1),
]


def _octant(d: Tuple[int, int]) -> int:
    return DIRS.index(d)


def _turn(d1: Tuple[int, int], d2: Tuple[int, int]) -> int:
    """Turn angle in degrees: 0 / 45 / 90 / 135 / 180."""
    t = (_octant(d2) - _octant(d1)) % 8
    return min(t, 8 - t) * 45


# --------------------------------------------------------------------------- #
# Instance
# --------------------------------------------------------------------------- #
@dataclass
class Instance:
    L: int
    W: int
    H: int
    delta: int
    d_micro: float
    d_C4: float
    d_via: float
    sc: Dict[str, float]             # sc_micro, sc_C4
    sc_via: Dict[str, float]         # micro, C4, via
    sc_wire: Dict[str, float]        # micro, C4, via, wire
    rho: Dict[str, float | None]     # m1..mL per-layer length ratio upper bound
    sigma: Dict[str, float | None]   # G1..GK group length-deviation limit
    q: float
    p: int
    nets: List[str]                          # signal net names (e.g. G1_0001)
    micro_of: Dict[str, XY]                  # net -> micro grid index
    all_micro: List[XY]                      # all micros incl. dummies (obstacles)
    candidates: List[XY]                     # C4 candidate grid indices
    eta: float = 2 ** -4                     # bend-count penalty weight (default 0.0625)
    config_path: str = ""                    # source configuration for run provenance

    @property
    def nx(self) -> int:
        return self.W // self.delta + 1

    @property
    def ny(self) -> int:
        return self.H // self.delta + 1

    def group_of(self, net: str) -> str:
        return net.split("_")[0]


def load_instance(yaml_path: str, data_dir: str) -> Instance:
    with open(yaml_path, "r", encoding="utf-8", errors="replace") as f:
        raw = yaml.safe_load(f)
    lay, spec, rout, plc = raw["layout"], raw["spec"], raw["routing"], raw["placement"]
    delta = int(spec["delta"])

    def to_idx(x: int, y: int) -> XY:
        assert x % delta == 0 and y % delta == 0, f"({x},{y}) is not on the grid"
        return (x // delta, y // delta)

    nets: List[str] = []
    micro_of: Dict[str, XY] = {}
    all_micro: List[XY] = []
    with open(os.path.join(data_dir, "micro_coordinate.csv"), encoding="utf-8") as f:
        for row in csv.DictReader(f):
            ij = to_idx(int(row["micro_x"]), int(row["micro_y"]))
            all_micro.append(ij)
            if row["group"] != "dummy":
                name = f"{row['group']}_{row['micro_id']}"
                nets.append(name)
                micro_of[name] = ij

    cands: List[XY] = []
    with open(os.path.join(data_dir, "C4_candidate.csv"), encoding="utf-8") as f:
        for row in csv.DictReader(f):
            cands.append(to_idx(int(row["C4_x"]), int(row["C4_y"])))

    return Instance(
        L=int(lay["L"]), W=int(lay["W"]), H=int(lay["H"]), delta=delta,
        d_micro=float(spec["d_micro"]), d_C4=float(spec["d_C4"]), d_via=float(spec["d_via"]),
        sc={k: float(v) for k, v in spec["sc"].items()},
        sc_via={k: float(v) for k, v in spec["sc_via"].items()},
        sc_wire={k: float(v) for k, v in spec["sc_wire"].items()},
        rho={k.lower(): (float(v) if v is not None else None)
             for k, v in rout.get("rho", {}).items()},
        sigma={k: (float(v) if v is not None else None)
               for k, v in rout.get("sigma", {}).items()},
        q=float(plc["q"]), p=int(plc["p"]),
        nets=sorted(nets), micro_of=micro_of, all_micro=all_micro, candidates=cands,
        eta=float(raw.get("eta", 2 ** -4)),
        config_path=os.path.abspath(yaml_path),
    )


# --------------------------------------------------------------------------- #
# Geometry preprocessing
# --------------------------------------------------------------------------- #
class Geometry:
    def __init__(self, inst: Instance):
        self.inst = inst
        d = inst.delta
        self.top = inst.L - 1          # layer index of micro side (ML)
        self.bot = 0                   # layer index of C4 side (M1)

        self.cells: List[XY] = [(i, j) for i in range(inst.nx) for j in range(inst.ny)]
        self.cell_set = set(self.cells)

        # coordinate distance per grid step
        self.d = d

        self._build_arcs()
        self._build_bend_sets()

    # ------------------------------------------------------------------ #
    def _dist(self, a: XY, b: XY) -> float:
        return self.d * math.hypot(a[0] - b[0], a[1] - b[1])

    def cells_within(self, c: XY, limit: float) -> List[XY]:
        """Grid points (including c) whose coordinate distance from c is less than limit."""
        r = int(math.floor(limit / self.d)) + 1
        out = []
        for di in range(-r, r + 1):
            for dj in range(-r, r + 1):
                t = (c[0] + di, c[1] + dj)
                if t in self.cell_set and self._dist(c, t) < limit:
                    out.append(t)
        return out

    # ------------------------------------------------------------------ #
    def _build_arcs(self) -> None:
        """Planar arcs (8 directions within each layer) and via arcs (l -> l+1, upward only)."""
        d = self.d
        self.planar: List[Tuple[Node, Node]] = []
        self.arc_dir: Dict[Tuple[Node, Node], Tuple[int, int]] = {}
        self.alpha: Dict[Tuple[Node, Node], float] = {}
        for l in range(self.inst.L):
            for (i, j) in self.cells:
                for dd in DIRS:
                    t = (i + dd[0], j + dd[1])
                    if t in self.cell_set:
                        a = ((i, j, l), (t[0], t[1], l))
                        self.planar.append(a)
                        self.arc_dir[a] = dd
                        self.alpha[a] = d * (math.sqrt(2.0) if dd[0] and dd[1] else 1.0)
        # Vias have zero length (validator's routing_length is the sum of per-layer planar lengths)
        self.vias: List[Tuple[Node, Node]] = [
            ((i, j, l), (i, j, l + 1))
            for l in range(self.inst.L - 1) for (i, j) in self.cells
        ]
        for a in self.vias:
            self.alpha[a] = 0.0

        self.succ: Dict[Node, List[Node]] = defaultdict(list)
        self.pred: Dict[Node, List[Node]] = defaultdict(list)
        for (p, q) in self.planar + self.vias:
            self.succ[p].append(q)
            self.pred[q].append(p)

    # ------------------------------------------------------------------ #
    def _build_bend_sets(self) -> None:
        """E(pq): successor nodes with turn >= 90 degrees (forbidden)
           U(pq): successor nodes with turn == 45 degrees (bend count)"""
        self.E: Dict[Tuple[Node, Node], List[Node]] = {}
        self.U: Dict[Tuple[Node, Node], List[Node]] = {}
        for a in self.planar:
            p, q = a
            d1 = self.arc_dir[a]
            e, u = [], []
            for r in self.succ[q]:
                nxt = (q, r)
                if nxt not in self.arc_dir:      # transitions to a via are not planar turns
                    continue
                t = _turn(d1, self.arc_dir[nxt])
                if t >= 90:
                    e.append(r)
                elif t == 45:
                    u.append(r)
            self.E[a] = e
            self.U[a] = u


# --------------------------------------------------------------------------- #
# MILP
# --------------------------------------------------------------------------- #
class PDRCModel:
    def __init__(self, inst: Instance, eta: float = 1.0,
                 spacing_mode: str = "pairwise"):
        """Build the supplement's explicit linear constraints.

        The legacy 'pairwise' option is retained for command compatibility.
        Other spacing modes are not part of this formulation.
        """
        if spacing_mode != "pairwise":
            raise ValueError("Only the supplementary linear formulation is supported.")
        if not math.isfinite(eta) or eta <= 0:
            raise ValueError("eta must be finite and positive.")
        self.inst = inst
        self.eta = eta
        self.spacing_mode = spacing_mode
        self.geo = Geometry(inst)
        self.model: gp.Model | None = None

    # ------------------------------------------------------------------ #
    def _net_arcs(self, n: str) -> List[Tuple[Node, Node]]:
        """All grid edges plus the net's source/sink depot connections."""
        inst, geo = self.inst, self.geo
        arcs = list(geo.planar) + list(geo.vias)
        arcs += [(DEPOT, (k[0], k[1], geo.bot)) for k in inst.candidates]
        mic = inst.micro_of[n]
        arcs += [((mic[0], mic[1], geo.top), DEPOT)]
        return arcs

    # ------------------------------------------------------------------ #
    def build(self) -> gp.Model:
        inst, geo = self.inst, self.geo
        nets, L = inst.nets, inst.L
        m = gp.Model("pdrc")

        self.arcs: Dict[str, List[Tuple[Node, Node]]] = {n: self._net_arcs(n) for n in nets}
        arcset = {n: set(self.arcs[n]) for n in nets}

        # Node sets (per net)
        self.nodes: Dict[str, List[Node]] = {}
        for n in nets:
            s = {v for a in self.arcs[n] for v in a if v != DEPOT}
            self.nodes[n] = sorted(s)

        # ---- Variables ---- #
        def tag(v) -> str:                       # MPS-compatible: name without spaces
            return "_".join(str(t) for t in v)

        b = {(n, k): m.addVar(vtype=GRB.BINARY, name=f"b.{n}.{tag(k)}")
             for n in nets for k in inst.candidates}
        x = {(n, p, q): m.addVar(vtype=GRB.BINARY, name=f"x.{n}.{tag(p)}.{tag(q)}")
             for n in nets for (p, q) in self.arcs[n]}
        y = {(n, v): m.addVar(vtype=GRB.BINARY, name=f"y.{n}.{tag(v)}")
             for n in nets for v in self.nodes[n]}
        # eq:via -- binary via use at BOTH endpoints of each vertical edge.
        z = {(n, v): m.addVar(vtype=GRB.BINARY, name=f"z.{n}.{tag(v)}")
             for n in nets for v in self.nodes[n]}
        # eq:continuous -- nonnegative continuous auxiliaries on all same-layer edges.
        u = {(n, p, q): m.addVar(lb=0.0, name=f"u.{n}.{tag(p)}.{tag(q)}")
             for n in nets for (p, q) in geo.planar}
        c = m.addVars(nets, lb=0.0, name="c")                       # total wire length per net
        ll = m.addVars(nets, range(L), lb=0.0, name="len")          # per-layer wire length
        groups = sorted({inst.group_of(n) for n in nets})
        r1 = m.addVars(groups, lb=0.0, name="r1")
        r2 = m.addVars(groups, lb=0.0, name="r2")

        inc: Dict[Tuple[str, Node], List] = defaultdict(list)
        out: Dict[Tuple[str, Node], List] = defaultdict(list)
        for n in nets:
            for (p, q) in self.arcs[n]:
                out[n, p].append((p, q))
                inc[n, q].append((p, q))

        # ---- Objective: wire length + eta * bends ---- #
        m.setObjective(
            gp.quicksum(c[n] for n in nets)
            + self.eta * gp.quicksum(u[key] for key in u),
            GRB.MINIMIZE)

        # ---- C4 placement ---- #
        for n in nets:
            m.addConstr(gp.quicksum(b[n, k] for k in inst.candidates) == 1, name=f"one_c4.{n}")
            for k in inst.candidates:                                # depot -> C4 node
                m.addConstr(x[n, DEPOT, (k[0], k[1], geo.bot)] == b[n, k], name=f"src.{n}.{k}")
            mic = inst.micro_of[n]
            m.addConstr(x[n, (mic[0], mic[1], geo.top), DEPOT] == 1, name=f"snk.{n}")

        # S13 / eq:c4c4: candidate positions already satisfy C4 spacing.
        for n in nets:
            for candidate in inst.candidates:
                m.addConstr(gp.quicksum(b[other, candidate] for other in nets if other != n)
                            <= 1 - b[n, candidate], name=f"c4c4.{n}.{candidate}")

        # Window density: number of C4s inside a q x q window <= p
        win = inst.q * inst.sc["C4"]
        for (ox, oy) in [(k[0] * geo.d, k[1] * geo.d) for k in inst.candidates]:
            insides = [k for k in inst.candidates
                       if ox <= k[0] * geo.d < ox + win and oy <= k[1] * geo.d < oy + win]
            if len(insides) * len(nets) > inst.p:
                m.addConstr(gp.quicksum(b[n, k] for n in nets for k in insides) <= inst.p,
                            name=f"dens.{ox}.{oy}")

        # ---- Flow / node ---- #
        for n in nets:
            for v in self.nodes[n]:
                m.addConstr(gp.quicksum(x[n, p, q] for (p, q) in inc[n, v])
                            == gp.quicksum(x[n, p, q] for (p, q) in out[n, v]), name=f"flow.{n}.{v}")
                m.addConstr(gp.quicksum(x[n, p, q] for (p, q) in inc[n, v]) == y[n, v],
                            name=f"use.{n}.{v}")
                # eq:via: two consecutive vertical edges at one node are impossible.
                vertical = [(p, q) for p, q in inc[n, v]
                            if p != DEPOT and p[2] == v[2] - 1]
                vertical += [(p, q) for p, q in out[n, v]
                             if q != DEPOT and q[2] == v[2] + 1]
                m.addConstr(gp.quicksum(x[n, p, q] for p, q in vertical) == z[n, v],
                            name=f"via.{n}.{v}")

        # ---- Wire length ---- #
        for n in nets:
            for l in range(L):
                m.addConstr(ll[n, l] == gp.quicksum(
                    geo.alpha[(p, q)] * x[n, p, q]
                    for (p, q) in self.arcs[n] if p != DEPOT and q != DEPOT
                    and p[2] == l and q[2] == l), name=f"len.{n}.{l}")
            m.addConstr(c[n] == gp.quicksum(ll[n, l] for l in range(L)), name=f"tot.{n}")

        # ---- Per-layer length ratio rho ---- #
        for l in range(L):
            rv = inst.rho.get(f"m{l + 1}")
            if rv is not None:
                m.addConstr(gp.quicksum(ll[n, l] for n in nets)
                            <= rv * gp.quicksum(c[n] for n in nets),
                            name=f"rho.{l}")

        # ---- Group length deviation (max - min) / min <= sigma ---- #
        for g in groups:
            members = [n for n in nets if inst.group_of(n) == g]
            for n in members:
                m.addConstr(r1[g] <= c[n], name=f"gmin.{g}.{n}")
                m.addConstr(r2[g] >= c[n], name=f"gmax.{g}.{n}")
            sg = inst.sigma.get(g)
            if sg is not None:
                m.addConstr(r2[g] <= (1.0 + sg) * r1[g], name=f"gdev.{g}")

        # ---- Turn angle (only +-45 deg allowed) ---- #
        for n in nets:
            for a in self.arcs[n]:
                forb = geo.E.get(a)
                if not forb:
                    continue
                p, q = a
                t = [x[n, q, r] for r in forb if (q, r) in arcset[n]]
                if t:
                    m.addConstr(gp.quicksum(t) <= 1 - x[n, p, q], name=f"turn.{n}.{a}")

        # ---- Bend count ---- #
        for n in nets:
            for a in self.arcs[n]:
                if (n, a[0], a[1]) not in u:
                    continue
                p, q = a
                nxt = [x[n, q, r] for r in geo.U[a] if (q, r) in arcset[n]]
                m.addConstr(x[n, p, q] + gp.quicksum(nxt) - 1 <= u[n, p, q], name=f"bend.{n}.{a}")

        # ---- Spacing rules ---- #
        self._add_spacing(m, b, y, z)

        m.update()
        self.model = m
        self.vars = dict(b=b, x=x, y=y, z=z, u=u, c=c, ll=ll, r1=r1, r2=r2)
        return m

    # ------------------------------------------------------------------ #
    def _add_spacing(self, m, b, y, z) -> None:
        """eq:wiremicro through eq:viamicro, using the table's node sets.

        Big-M values use the number of binary terms in each neighborhood sum.
        These are valid local bounds and give the same integer feasible region
        as a single sufficiently large Theta, without an arbitrary large value.
        """
        inst, geo = self.inst, self.geo
        nets = inst.nets

        def neighborhood_union(centers, limit):
            return {v for center in centers for v in geo.cells_within(center, limit)}

        # B_n: all other signal micros and all dummy micros; own micro exempt.
        for n in nets:
            blocked = neighborhood_union(
                [v for v in inst.all_micro if v != inst.micro_of[n]], inst.sc_wire["micro"])
            m.addConstr(gp.quicksum(y[n, (i, j, geo.top)] for i, j in sorted(blocked)) == 0,
                        name=f"wiremicro.{n}")

        # S^wc_c: unused candidates remain dummy obstacles for every net.
        for candidate in inst.candidates:
            for i, j in geo.cells_within(candidate, inst.sc_wire["C4"]):
                for n in nets:
                    m.addConstr(y[n, (i, j, geo.bot)] <= b[n, candidate],
                                name=f"wirec4.{n}.{candidate}.{i}.{j}")

        # Same-layer neighborhoods. S^vv excludes the reference node exactly
        # as in the supplement; wire-wire neighborhoods include that node.
        ww = {c: geo.cells_within(c, inst.sc_wire["wire"]) for c in geo.cells}
        wv = {c: geo.cells_within(c, inst.sc_wire["via"]) for c in geo.cells}
        vv = {c: [t for t in geo.cells_within(c, inst.sc_via["via"]) if t != c]
              for c in geo.cells}
        for n in nets:
            others = [o for o in nets if o != n]
            for v in self.nodes[n]:
                cell, layer = v[:2], v[2]
                for label, neighborhood, trigger in (
                        ("wirewire", ww[cell], y[n, v]),
                        ("wirevia", wv[cell], z[n, v])):
                    bound = len(others) * len(neighborhood)
                    if bound:
                        lhs = gp.quicksum(y[o, (i, j, layer)] for o in others
                                         for i, j in neighborhood)
                        m.addConstr(lhs <= bound * (1 - trigger), name=f"{label}.{n}.{v}")
                bound = len(nets) * len(vv[cell])
                if bound:
                    lhs = gp.quicksum(z[o, (i, j, layer)] for o in nets for i, j in vv[cell])
                    m.addConstr(lhs <= bound * (1 - z[n, v]), name=f"viavia.{n}.{v}")

        # S19 / S^vc_c: every candidate contains a signal or dummy C4 bump.
        for candidate in inst.candidates:
            for i, j in geo.cells_within(candidate, inst.sc_via["C4"]):
                for n in nets:
                    m.addConstr(z[n, (i, j, geo.bot)] == 0,
                                name=f"viac4.{n}.{candidate}.{i}.{j}")

        # S20 / S^vm: signal and dummy micro bumps on the ML layer.
        blocked = neighborhood_union(inst.all_micro, inst.sc_via["micro"])
        for n in nets:
            m.addConstr(gp.quicksum(z[n, (i, j, geo.top)] for i, j in sorted(blocked)) == 0,
                        name=f"viamicro.{n}")

    # ------------------------------------------------------------------ #
    def solve(self, time_limit: float = 3600, mip_gap: float = 0.01, **params) -> gp.Model:
        if self.model is None:
            self.build()
        self.model.setParam("TimeLimit", time_limit)
        self.model.setParam("MIPGap", mip_gap)
        for k, v in params.items():
            self.model.setParam(k, v)
        self.model.optimize()
        return self.model

    def _selected_routes(self, values):
        """Return paths and disconnected selected arcs, without dropping cycles."""
        paths, leftovers = {}, {}
        for n in self.inst.nets:
            selected = {(p, q) for nn, p, q in values if nn == n and values[nn, p, q] > 0.5}
            starts = [q for p, q in selected if p == DEPOT]
            if len(starts) != 1:
                raise ValueError(f"{n}: expected one selected source")
            succ = {p: q for p, q in selected if p != DEPOT}
            path, seen = [starts[0]], {(DEPOT, starts[0])}
            while path[-1] != DEPOT:
                p = path[-1]
                if p not in succ or (p, succ[p]) in seen:
                    raise ValueError(f"{n}: selected source path does not reach sink")
                q = succ[p]
                seen.add((p, q))
                path.append(q)
            paths[n] = path[:-1]
            leftovers[n] = selected - seen
        return paths, leftovers

    def _paths_json(self, paths):
        return [dict(netname=n, **{
            f"m{l+1}": _collapse([[v[0]*self.geo.d, v[1]*self.geo.d]
                                  for v in paths[n] if v[2] == l])
            for l in range(self.inst.L)}) for n in self.inst.nets]

    def solution_diagnostics(self):
        """Report all selected arcs; do not change the feasible region."""
        values = {k: var.X for k, var in self.vars["x"].items()}
        paths, leftovers = self._selected_routes(values)
        full_length = sum(self.geo.alpha.get((p,q), 0.0)
                          for (n,p,q), used in values.items() if used > 0.5)
        path_length = sum(self.geo.alpha[(p,q)] for path in paths.values()
                          for p,q in zip(path,path[1:]))
        return dict(
            formulation="supplementary_node_spacing_v2",
            eta=self.eta, rho=self.inst.rho, sigma=self.inst.sigma,
            solver_status=self.model.Status, runtime_seconds=self.model.Runtime,
            best_bound=self.model.ObjBound, node_count=self.model.NodeCount,
            time_limit=self.model.Params.TimeLimit, mip_gap_target=self.model.Params.MIPGap,
            solver_version=list(gp.gurobi.version()),
            objective=self.model.ObjVal, gap=self.model.MIPGap,
            selected_wire_length=full_length, exported_path_wire_length=path_length,
            disconnected_wire_length=full_length-path_length,
            routing_json_is_complete=not any(leftovers.values()),
            disconnected_arcs={n: [[list(p),list(q)] for p,q in sorted(a)]
                               for n,a in leftovers.items()},
            selected_arcs={n: [[list(p),list(q)] for (nn,p,q),used in values.items()
                               if nn==n and used > 0.5] for n in self.inst.nets},
            arc_coordinate_convention="grid indices; zero-based layers; depot=(-1,-1,-1)",
        )

    # ------------------------------------------------------------------ #
    def to_routing_json(self) -> List[dict]:
        """Convert to routing JSON format read by the repo validator.
        Waypoints are kept only at turn points."""
        values = {k: var.X for k, var in self.vars["x"].items()}
        paths, leftovers = self._selected_routes(values)
        if any(leftovers.values()):
            print("[diagnostic] Disconnected cycles exist. Routing JSON contains only "
                  "terminal paths; inspect the diagnostics JSON for all selected arcs.")
        return self._paths_json(paths)


def _collapse(pts: List[List[int]]) -> List[List[int]]:
    """Remove intermediate collinear points, keeping only turn points."""
    if len(pts) <= 2:
        return pts
    keep = [pts[0]]
    for i in range(1, len(pts) - 1):
        ax, ay = pts[i][0] - keep[-1][0], pts[i][1] - keep[-1][1]
        bx, by = pts[i + 1][0] - pts[i][0], pts[i + 1][1] - pts[i][1]
        if ax * by - ay * bx != 0:          # direction changes
            keep.append(pts[i])
    keep.append(pts[-1])
    return keep


# --------------------------------------------------------------------------- #
def main() -> None:
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("-c", "--config", required=True)
    ap.add_argument("-d", "--data-dir", required=True)
    ap.add_argument("-o", "--out", default="",
                    help="Output JSON path (default: data-dir/routing_result.json)")
    ap.add_argument("--eta", type=float, default=None,
                    help="Bend-count penalty weight (default: read from input.yaml eta field)")
    ap.add_argument("--time-limit", type=float, default=3600)
    ap.add_argument("--gap", type=float, default=0.01)
    ap.add_argument("--max-combos", type=int, default=20,
                    help="max number of C4 candidate combinations the warm-start router tries")
    ap.add_argument("--layer-cost", default="",
                    help='per-layer length cost multiplier for the warm-start router, e.g. "3:3.0" '
                         '(discourages use of layers with an active rho bound)')
    ap.add_argument("--spacing-mode", default="pairwise",
                    choices=["pairwise"],
                    help="Legacy option; uses the supplement's explicit linear spacing constraints")
    ap.add_argument("--warm-start", action="store_true",
                    help="inject the constructive router (pdrc_router) solution as a MIP start")
    ap.add_argument("--log-file", default="",
                    help="Gurobi log path (default: data-dir/gurobi.log)")
    args = ap.parse_args()

    if not args.out:
        args.out = os.path.join(args.data_dir, "routing_result.json")
    if not args.log_file:
        args.log_file = os.path.join(args.data_dir, "gurobi.log")

    inst = load_instance(args.config, args.data_dir)
    eta = args.eta if args.eta is not None else inst.eta
    mdl = PDRCModel(inst, eta=eta, spacing_mode=args.spacing_mode)
    mdl.build()

    if args.warm_start:
        try:
            from pdrc_router import Router
        except ImportError:
            print("[warm start] pdrc_router not available, skipping warm start.")
            args.warm_start = False
    if args.warm_start:
        lc = {int(k): float(v) for k, v in
              (t.split(":") for t in args.layer_cost.split(",") if t)}
        rt = Router(inst, eta=eta, layer_cost=lc)
        paths = rt.solve(max_combos=args.max_combos)
        if paths is None:
            print("[warm start] router found no solution, proceeding without warm start.")
        else:
            sv = start_values(mdl, paths)
            for var in mdl.model.getVars():
                var.Start = sv.get(var.VarName, 0.0)
            print("[warm start] router solution injected (total WL "
                  f"{sum(rt.path_length(p) for p in paths.values()):.1f})")
    print(f"formulation=supplementary_node_spacing_v2 eta={eta:g}")
    print(f"nets={len(inst.nets)} layers={inst.L} grid={inst.nx}x{inst.ny} "
          f"cands={len(inst.candidates)}")
    print(f"vars={mdl.model.NumVars:,} constrs={mdl.model.NumConstrs:,} "
          f"gen={mdl.model.NumGenConstrs:,}")

    extra = {}
    if args.log_file:
        extra["LogFile"] = args.log_file
    try:
        mdl.solve(time_limit=args.time_limit, mip_gap=args.gap, **extra)
    except KeyboardInterrupt:
        pass
    if mdl.model.SolCount:
        diagnostic_path = os.path.splitext(args.out)[0] + "_diagnostics.json"
        with open(diagnostic_path, "w", encoding="utf-8") as f:
            json.dump(mdl.solution_diagnostics(), f, indent=2)
        js = mdl.to_routing_json()
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump(js, f, indent=1)
        print(f"obj={mdl.model.ObjVal:.2f}  gap={mdl.model.MIPGap * 100:.2f}%  -> {args.out}")




# --------------------------------------------------------------------------- #
# Warm start (pdrc_router.Router solution -> variable values)
# --------------------------------------------------------------------------- #
def start_values(mdl: "PDRCModel", paths) -> Dict[str, float]:
    """Convert constructive router paths to a dict of MILP variable values (name -> value)."""
    inst, geo = mdl.inst, mdl.geo
    v = {var.VarName: 0.0 for var in mdl.model.getVars()}

    def tag(t) -> str:
        return "_".join(str(z) for z in t)

    tot = {}
    for n in inst.nets:
        seq = [(c[0], c[1], l) for (c, l) in paths[n]]
        c4 = (seq[0][0], seq[0][1])
        v[f"b.{n}.{tag(c4)}"] = 1.0
        v[f"x.{n}.{tag(DEPOT)}.{tag(seq[0])}"] = 1.0
        v[f"x.{n}.{tag(seq[-1])}.{tag(DEPOT)}"] = 1.0
        for a in seq:
            v[f"y.{n}.{tag(a)}"] = 1.0
        lens = {l: 0.0 for l in range(inst.L)}
        arcs = list(zip(seq, seq[1:]))
        for p, q in arcs:
            v[f"x.{n}.{tag(p)}.{tag(q)}"] = 1.0
            if p[2] == q[2]:
                lens[p[2]] += geo.alpha[(p, q)]
            else:
                v[f"z.{n}.{tag(p)}"] = 1.0
                v[f"z.{n}.{tag(q)}"] = 1.0
        for (p, q), (q2, r) in zip(arcs, arcs[1:]):
            if p[2] == q[2] == r[2] and r in geo.U.get((p, q), []):
                v[f"u.{n}.{tag(p)}.{tag(q)}"] = 1.0
        tot[n] = sum(lens.values())
        v[f"c[{n}]"] = tot[n]
        for l in range(inst.L):
            v[f"len[{n},{l}]"] = lens[l]
    for g in sorted({inst.group_of(n) for n in inst.nets}):
        mem = [tot[n] for n in inst.nets if inst.group_of(n) == g]
        v[f"r1[{g}]"] = min(mem)
        v[f"r2[{g}]"] = max(mem)
    return v

if __name__ == "__main__":
    main()
