"""Standalone version of export_routing_json.py — no setting.py dependency."""
import argparse
import csv
import json
import math
import pickle
import sys
import os


REVERSE_GROUP_MAP = {'group1': 'G1', 'group2': 'G2', 'group3': 'G3', 'group4': 'G4', 'group5': 'G5'}


def _load_micro_id_padding(instance_dir):
    mapping = {}
    path = f'{instance_dir}/micro_coordinate.csv'
    with open(path, 'r', encoding='utf-8') as f:
        for row in csv.DictReader(f):
            try:
                mapping[(row['group'], int(row['micro_id']))] = row['micro_id']
            except ValueError:
                continue
    return mapping


def convert_netname(netname, micro_id_padding):
    group, micro_id = netname.split('_', 1)
    group_letter = REVERSE_GROUP_MAP.get(group, group)
    padded_id = micro_id_padding.get((group_letter, int(micro_id)), micro_id)
    return f"{group_letter}_{padded_id}"


def _sign(v):
    return (v > 0) - (v < 0)


def simplify_collinear(points):
    pts = [p for i, p in enumerate(points) if i == 0 or p != points[i - 1]]
    if len(pts) <= 2:
        return pts
    simplified = [pts[0]]
    for i in range(1, len(pts) - 1):
        d1 = (_sign(pts[i][0] - pts[i - 1][0]), _sign(pts[i][1] - pts[i - 1][1]))
        d2 = (_sign(pts[i + 1][0] - pts[i][0]), _sign(pts[i + 1][1] - pts[i][1]))
        if d1 != d2:
            simplified.append(pts[i])
    simplified.append(pts[-1])
    return simplified


def export_routing_json(pkl_path, instance_dir, out_path):
    micro_id_padding = _load_micro_id_padding(instance_dir)

    with open(pkl_path, 'rb') as f:
        signals = pickle.load(f)

    nets = []
    for sig in signals.values():
        num_layers = len(sig.layer_routes)
        entry = {"netname": convert_netname(sig.netname, micro_id_padding)}
        for our_layer_idx in range(num_layers):
            route = sig.layer_routes[our_layer_idx]
            if len(route) < 2:
                continue
            m_idx = num_layers - our_layer_idx
            pts = [(float(x), float(y)) for x, y in reversed(route)]
            entry[f"m{m_idx}"] = [list(p) for p in simplify_collinear(pts)]
        nets.append(entry)

    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(nets, f, indent=2)

    print(f"wrote {len(nets)} nets to {out_path}")


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--pkl', required=True, help="path to signals_*.pkl file")
    ap.add_argument('--instance-dir', required=True, help="path to instance data dir (has micro_coordinate.csv)")
    ap.add_argument('-o', '--output', required=True, help="output JSON path")
    args = ap.parse_args()

    export_routing_json(args.pkl, args.instance_dir, args.output)
