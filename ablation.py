#!/usr/bin/env python3
"""Ablation for HLADMM, several seeds per instance.  Variants (same instance, same seed):
  full        : ADMM + rounding + one-flip + restarts
  no_local    : ADMM + rounding only (best over restarts)
  no_admm     : random start + rounding + one-flip, same number of restarts, --restarts (NOT time-matched)
  ls_matched  : random-start rounding + one-flip, restarted until the wall-clock of 'full'
  pg          : projected gradient (step 1/L) + rounding + one-flip, same restarts
  one_restart : first restart of 'full' (identical to a separate R_max=1 run)
  guided      : ONE ADMM run, then random roundings of its fractional z + one-flip search,
                until the wall-clock of 'full' (relaxation-guided multistart)
Usage: python ablation.py bqpgub --dims 200 500 --counts 5 5 --seeds 10 --restarts 20 --tol 0.001
Writes one CSV row per (instance, seed) and prints a paired summary vs 'full'.
"""
import argparse, csv, os, sys
import numpy as np
from hladmm import hladmm, read_instance

ap = argparse.ArgumentParser()
ap.add_argument("family", choices=["bqpgub", "cmisp"])
ap.add_argument("--dims", type=int, nargs="+", required=True)
ap.add_argument("--counts", type=int, nargs="+", required=True)
ap.add_argument("--dir", default=".")
ap.add_argument("--prefix", default=None, help="file prefix (default bqpgubs / cmisp)")
ap.add_argument("--out", default=None)
ap.add_argument("--rho_scale", type=float, default=5.0, help="rho = c*L; theory needs c > 4.18 for kappa=1.1 (final runs use c = 5)")
ap.add_argument("--kappa", type=float, default=1.1)
ap.add_argument("--max_iter", type=int, default=2000)
ap.add_argument("--tol", type=float, default=0.01)
ap.add_argument("--seeds", type=int, default=10)
ap.add_argument("--restarts", type=int, default=20)
ap.add_argument("--order", choices=["zx", "xz"], default="zx")
a = ap.parse_args()
prefix = a.prefix or {"bqpgub": "bqpgubs", "cmisp": "cmisp"}[a.family]
base = dict(rho_scale=a.rho_scale, kappa=a.kappa, max_iter=a.max_iter, tol=a.tol,
            elite_size=5, order=a.order, L_safety=1.0)   # L_safety=1.0: same L as hladmm_final.py
VARS = ["full", "no_local", "no_admm", "ls_matched", "pg", "one_restart", "guided"]


def find(b):
    for e in (".txt", ".mat", ".npz"):
        pth = os.path.join(a.dir, b + e)
        if os.path.isfile(pth):
            return pth
    sys.exit(f"cannot find {b}.txt/.mat in {a.dir} (use --prefix?)")


rows = []
with open(a.out or f"{a.family}_ablation.csv", "w", newline="") as fh:
    wr = csv.writer(fh)
    wr.writerow(["name", "n", "m", "seed"] + VARS + ["t_full", "t_admm", "t_ls", "t_no_admm", "t_pg"])
    for N, cnt in zip(a.dims, a.counts):
        for k in range(1, cnt + 1):
            R, p, g = read_instance(find(f"{prefix}-{N}-{k}"))
            for s in range(a.seeds):
                _, f_full, i1 = hladmm(R, p, g, n_restarts=a.restarts, seed=s, **base)
                _, f_na, i2 = hladmm(R, p, g, n_restarts=a.restarts, seed=s, no_admm=True, **base)
                _, f_ls, _ = hladmm(R, p, g, n_restarts=10**9, seed=s, no_admm=True,
                                    time_budget=i1["time_total"], **base)
                _, f_pg, i3 = hladmm(R, p, g, n_restarts=a.restarts, seed=s, method="pg", **base)
                _, f_gd, _ = hladmm(R, p, g, n_restarts=1, seed=s, guided=True,
                                    time_budget=i1["time_total"], **base)
                vals = dict(full=f_full, no_local=max(i1["f_round"]), no_admm=f_na,
                            ls_matched=f_ls, pg=f_pg, one_restart=i1["f_ref"][0], guided=f_gd)
                row = [f"{a.family}-{N}-{k}", p.size, len(g), s] + [vals[v] for v in VARS] + \
                      [round(i1["time_total"], 2), round(i1["t_admm"], 2), round(i1["t_ls"], 2),
                       round(i2["time_total"], 2), round(i3["time_total"], 2)]
                wr.writerow(row); fh.flush(); print(row, flush=True)
                rows.append(vals)

print("\nPaired summary vs full (over all instance-seed pairs):")
for v in VARS[1:]:
    d = np.array([r[v] - r["full"] for r in rows])
    tol = 1e-9
    print(f"{v:12s} mean diff {d.mean():12.3f}   worse {int((d < -tol).sum()):4d}  "
          f"equal {int((abs(d) <= tol).sum()):4d}  better {int((d > tol).sum()):4d}")
