#!/usr/bin/env python3
"""rho = c*L sweep on a tuning set (use instances you do NOT report in the tables).
Usage: python sweep.py bqpgub --dims 200 500 --counts 3 3 --c 0.5 1 2 4.2 8
"""
import argparse, os, sys
import numpy as np
from hladmm import hladmm, read_instance
ap = argparse.ArgumentParser()
ap.add_argument("family", choices=["bqpgub", "cmisp"])
ap.add_argument("--dims", type=int, nargs="+", required=True)
ap.add_argument("--counts", type=int, nargs="+", required=True)
ap.add_argument("--dir", default="."); ap.add_argument("--prefix", default=None)
ap.add_argument("--c", type=float, nargs="+", default=[0.5, 1, 2, 4.2, 8])
ap.add_argument("--seed", type=int, default=0)
a = ap.parse_args()
prefix = a.prefix or {"bqpgub": "bqpgubs", "cmisp": "cmisp"}[a.family]
print("name c  value  mean_iters  hit_limit  max_Pres  time")
for N, cnt in zip(a.dims, a.counts):
    for k in range(1, cnt + 1):
        b = f"{prefix}-{N}-{k}"
        pth = next((os.path.join(a.dir, b + e) for e in (".txt", ".mat") if os.path.isfile(os.path.join(a.dir, b + e))), None)
        if pth is None: sys.exit(f"cannot find {b}")
        R, p, g = read_instance(pth)
        for c in a.c:
            try:
                _, f, i = hladmm(R, p, g, rho_scale=c, seed=a.seed)
            except FloatingPointError:
                print(b, c, "DIVERGED"); continue
            print(b, c, round(f, 2), np.mean(i["iters"]), sum(t >= 2000 for t in i["iters"]),
                  f"{np.nanmax(i['pres']):.2e}", round(i["time_total"], 2), flush=True)
