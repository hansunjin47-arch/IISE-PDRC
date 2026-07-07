# visualizer.py
#
# Visualizes routing output JSON in three ways:
#   1. Per-layer subplots
#   2. 3D multi-layer view
#   3. Top-down projection (all layers overlaid)
#
# Usage:
#   python visualizer.py -c input.yaml -r routing_output.json [-o output_dir]

from __future__ import annotations

import argparse
import json
import os
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import yaml
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401  (registers 3d projection)


# ─────────────────────────────────────────────────────────────────────────────
# 1. Data loading
# ─────────────────────────────────────────────────────────────────────────────

def load_routing_json(path: str) -> List[dict]:
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)


def load_config(path: str) -> dict:
    with open(path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


# ─────────────────────────────────────────────────────────────────────────────
# 2. Color helpers
# ─────────────────────────────────────────────────────────────────────────────

# Fixed colors per layer (up to 10 layers)
_LAYER_COLORS = [
    'royalblue', 'darkorange', 'green', 'red',
    'purple', 'brown', 'deeppink', 'gray', 'olive', 'cyan',
]


def layer_color(layer_idx: int) -> str:
    return _LAYER_COLORS[layer_idx % len(_LAYER_COLORS)]


def group_color(netname: str) -> tuple:
    """Return a color based on the group prefix (G1, G2, ...) using tab10."""
    group = netname.split('_')[0]   # 'G1_0001' → 'G1'
    try:
        idx = int(group[1:]) - 1    # G1 → 0, G2 → 1, ...
    except (ValueError, IndexError):
        idx = 0
    return plt.get_cmap('tab10')(idx % 10)


# ─────────────────────────────────────────────────────────────────────────────
# 3. Plot 1 — Per-layer plots (one file per layer)
#    Each layer is saved as a separate image, colored by group.
# ─────────────────────────────────────────────────────────────────────────────

def plot_per_layer(
    nets: List[dict],
    num_layers: int,
    out_dir: Optional[str] = None,
    base: str = 'routing',
) -> None:
    for layer_idx in range(num_layers):
        ln  = f'm{layer_idx + 1}'
        fig, ax = plt.subplots(figsize=(8, 7))
        ax.set_title(f'Layer {layer_idx + 1}', fontsize=13)
        ax.grid(True, linestyle='--', linewidth=0.3, alpha=0.5)
        ax.set_aspect('equal', adjustable='datalim')

        for net in nets:
            if ln not in net:
                continue
            pts   = net[ln]
            color = group_color(net.get('netname', ''))
            xs    = [p[0] for p in pts]
            ys    = [p[1] for p in pts]
            ax.plot(xs, ys, color=color, linewidth=0.8, alpha=0.7)
            ax.scatter(xs, ys, color=color, s=8, zorder=3)

        plt.tight_layout()
        out_path = (
            os.path.join(out_dir, f'{base}_layer{layer_idx + 1}.png')
            if out_dir else None
        )
        _save_or_show(fig, out_path, f'layer {layer_idx + 1}')


# ─────────────────────────────────────────────────────────────────────────────
# 4. Plot 2 — 3D multi-layer view
#    Each layer is a horizontal plane at z = layer_index.
#    Via connections are drawn as vertical gray lines between planes.
# ─────────────────────────────────────────────────────────────────────────────

def plot_3d(
    nets: List[dict],
    num_layers: int,
    out_path: Optional[str] = None,
) -> None:
    fig = plt.figure(figsize=(12, 9))
    ax  = fig.add_subplot(111, projection='3d')

    layer_names = [f'm{i}' for i in range(1, num_layers + 1)]

    # Collect x/y extent across all nets for background planes
    all_x: List[float] = []
    all_y: List[float] = []
    for net in nets:
        for ln in layer_names:
            if ln in net:
                all_x.extend(p[0] for p in net[ln])
                all_y.extend(p[1] for p in net[ln])
    x_range = [min(all_x), max(all_x)] if all_x else [0, 1]
    y_range = [min(all_y), max(all_y)] if all_y else [0, 1]

    # Draw each layer
    for layer_idx, ln in enumerate(layer_names):
        z     = float(layer_idx)
        color = layer_color(layer_idx)

        # Background plane
        xx, yy = np.meshgrid(x_range, y_range)
        zz = np.full_like(xx, z, dtype=float)
        ax.plot_surface(xx, yy, zz, alpha=0.07, color=color, zorder=1)

        # Routing segments
        for net in nets:
            if ln not in net:
                continue
            pts = net[ln]
            xs  = [p[0] for p in pts]
            ys  = [p[1] for p in pts]
            zs  = [z] * len(pts)
            ax.plot(xs, ys, zs, color=color, linewidth=0.7, alpha=0.7)

    # Draw via connections (vertical lines at junction points)
    for net in nets:
        for i in range(num_layers - 1):
            ln = layer_names[i]
            if ln not in net or len(net[ln]) < 1:
                continue
            vx, vy = net[ln][-1]   # last point of mN = via position
            ax.plot([vx, vx], [vy, vy], [float(i), float(i + 1)],
                    color='gray', linewidth=0.5, alpha=0.35)

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Layer')
    ax.set_zticks(list(range(num_layers)))
    ax.set_zticklabels([f'M{i + 1}' for i in range(num_layers)])

    # Legend
    legend_handles = [
        mpatches.Patch(color=layer_color(i), label=f'Layer {i + 1}')
        for i in range(num_layers)
    ]
    ax.legend(handles=legend_handles, loc='upper left', fontsize=8)

    plt.tight_layout()
    _save_or_show(fig, out_path, '3D')


# ─────────────────────────────────────────────────────────────────────────────
# 5. Plot 3 — Top-down projection
#    All layers projected onto 2D, color-coded by layer.
# ─────────────────────────────────────────────────────────────────────────────

def plot_projection(
    nets: List[dict],
    num_layers: int,
    out_path: Optional[str] = None,
) -> None:
    fig, ax = plt.subplots(figsize=(11, 10))
    ax.set_title('Top-down Projection (all layers)', fontsize=13)
    ax.set_aspect('equal', adjustable='datalim')
    ax.grid(True, linestyle='--', linewidth=0.3, alpha=0.5)

    for layer_idx in range(num_layers):
        ln    = f'm{layer_idx + 1}'
        color = layer_color(layer_idx)

        for net in nets:
            if ln not in net:
                continue
            pts = net[ln]
            xs  = [p[0] for p in pts]
            ys  = [p[1] for p in pts]
            ax.plot(xs, ys, color=color, linewidth=0.6, alpha=0.6)
            ax.scatter(xs, ys, color=color, s=4, zorder=3)

    legend_handles = [
        mpatches.Patch(color=layer_color(i), label=f'layer{i + 1}')
        for i in range(num_layers)
    ]
    ax.legend(handles=legend_handles, loc='upper right', fontsize=9)

    plt.tight_layout()
    _save_or_show(fig, out_path, 'projection')


# ─────────────────────────────────────────────────────────────────────────────
# 6. Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _save_or_show(fig: plt.Figure, out_path: Optional[str], label: str) -> None:
    if out_path:
        fig.savefig(out_path, dpi=150, bbox_inches='tight')
        print(f"  Saved {label} plot → '{out_path}'")
    else:
        plt.show()
    plt.close(fig)


# ─────────────────────────────────────────────────────────────────────────────
# 7. CLI
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    ap = argparse.ArgumentParser(
        description='Visualize routing output JSON (per-layer / 3D / projection).'
    )
    ap.add_argument('-c', '--config',  required=True, help='input.yaml path')
    ap.add_argument('-r', '--routing', required=True, help='routing output JSON path')
    ap.add_argument('-o', '--outdir',  default=None,
                    help='output directory for saved plots '
                         '(default: same directory as routing file)')
    args = ap.parse_args()

    cfg  = load_config(args.config)
    nets = load_routing_json(args.routing)

    num_layers = int(cfg['layout']['num_layer'])

    outdir = args.outdir or os.path.dirname(os.path.abspath(args.routing))
    os.makedirs(outdir, exist_ok=True)
    base = os.path.splitext(os.path.basename(args.routing))[0]

    print(f"[Visualizer] {len(nets)} nets, {num_layers} layers")

    plot_per_layer(nets, num_layers, out_dir=outdir, base=base)
    plot_3d(nets, num_layers,
            out_path=os.path.join(outdir, f'{base}_3d.png'))
    plot_projection(nets, num_layers,
                    out_path=os.path.join(outdir, f'{base}_projection.png'))

    print(f"[Visualizer] Done. All plots saved to '{outdir}'")


if __name__ == '__main__':
    main()
