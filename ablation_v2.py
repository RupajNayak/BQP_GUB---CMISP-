#!/usr/bin/env python3
"""Ablation for HLADMM (extends ablation.py).  Needs hladmm.py (the version with
method="pg", guided=True, time_budget=...) in the same folder.

Variants (same instance, same seed; see ablation.py for the definitions):
  full, no_local, no_admm, ls_matched, pg, one_restart, guided

Extra columns written to the CSV:
  starts_at_limit_full   number of the R_max starts of 'full' that ran into the iteration limit
  n_starts_ls_matched    number of random-start local searches completed by ls_matched
  n_roundings_guided     number of random roundings (with local search) drawn by guided
  t_* columns            wall-clock seconds of each part

Run (one command per size, in the background):
  caffeinate -dimsu nohup python3 -u ablation_v2.py bqpgub --dims 500 --counts 5 --seeds 5 \
      --restarts 20 --tol 0.001 --out abl_500.csv > abl_500.log 2>&1 &

Merge several CSV files and print the paired summary by n:
  python3 ablation_v2.py summarize abl_500.csv abl_1000.csv abl_2500.csv
"""
import argparse
import csv
import datetime
import os
import platform
import sys

import numpy as np

VARS = ["full", "no_local", "no_admm", "ls_matched", "pg", "one_restart", "guided"]
EXTRA = ["starts_at_limit_full", "n_starts_ls_matched", "n_roundings_guided",
         "t_full", "t_admm", "t_ls", "t_no_admm", "t_pg", "t_ls_matched", "t_guided"]
HEADER = ["name", "n", "m", "seed"] + VARS + EXTRA


# ------------------------------------------------------------------------- summary
def summarize(rows):
    """rows: list of dicts with float values for VARS and an 'n' field."""
    def block(sub, title):
        print(f"\n{title}  ({len(sub)} instance-seed pairs)")
        print(f"{'variant':12s} better equal worse | mean rel. diff to full (%)   min      max")
        full = np.array([r["full"] for r in sub])
        for v in VARS[1:]:
            x = np.array([r[v] for r in sub])
            tol = 1e-9 * np.maximum(1.0, np.abs(full))
            diff = x - full
            rel = 100.0 * diff / np.abs(full)
            print(f"{v:12s} {int((diff > tol).sum()):6d} {int((abs(diff) <= tol).sum()):5d} "
                  f"{int((diff < -tol).sum()):5d} | {rel.mean():+24.2f} {rel.min():+8.2f} {rel.max():+8.2f}")
    for n in sorted({int(r["n"]) for r in rows}):
        block([r for r in rows if int(r["n"]) == n], f"n = {n}")
    block(rows, "ALL SIZES")
    if "n_starts_ls_matched" in rows[0]:
        for n in sorted({int(r["n"]) for r in rows}):
            sub = [r for r in rows if int(r["n"]) == n]
            print(f"n={n}: ls_matched starts median {np.median([r['n_starts_ls_matched'] for r in sub]):.0f}, "
                  f"guided roundings median {np.median([r['n_roundings_guided'] for r in sub]):.0f}, "
                  f"starts of full at the iteration limit median {np.median([r['starts_at_limit_full'] for r in sub]):.0f}"
                  f", full time median {np.median([r['t_full'] for r in sub]):.1f}s")


def read_csvs(files):
    rows = []
    for f in files:
        with open(f, newline="") as fh:
            for r in csv.DictReader(fh):
                rows.append({k: (float(v) if k != "name" else v) for k, v in r.items()})
    return rows


# ------------------------------------------------------------------------- run
def find(folder, prefixes, N, k):
    for q in prefixes:
        for e in (".txt", ".mat", ".npz"):
            pth = os.path.join(folder, f"{q}-{N}-{k}{e}")
            if os.path.isfile(pth):
                return pth
    sys.exit(f"cannot find {prefixes[0]}-{N}-{k}.(txt|mat|npz) in {folder} (use --prefix?)")


def run(a):
    from hladmm import hladmm, read_instance
    prefixes = [a.prefix] if a.prefix else (["bqpgubs", "bqpgub"] if a.family == "bqpgub" else ["cmisp", "CMISP"])
    base = dict(rho_scale=a.rho_scale, kappa=a.kappa, max_iter=a.max_iter, tol=a.tol,
                elite_size=5, order=a.order, L_safety=1.0)
    out = a.out or f"{a.family}_ablation.csv"
    with open(out + ".settings.txt", "w") as fh:
        fh.write(f"date: {datetime.datetime.now().isoformat(timespec='seconds')}\n")
        fh.write(f"python {platform.python_version()}, numpy {np.__version__}, {platform.platform()}\n")
        fh.write("command: " + " ".join(sys.argv) + "\n")
        for k_, v_ in sorted(vars(a).items()):
            fh.write(f"{k_} = {v_}\n")
        fh.write(f"solver options: {base}\n")
    rows = []
    with open(out, "w", newline="") as fh:
        wr = csv.writer(fh)
        wr.writerow(HEADER)
        for N, cnt in zip(a.dims, a.counts):
            for k in range(1, cnt + 1):
                R, p, g = read_instance(find(a.dir, prefixes, N, k))
                for s in range(a.seed0, a.seed0 + a.seeds):
                    _, f_full, i1 = hladmm(R, p, g, n_restarts=a.restarts, seed=s, **base)
                    T = i1["time_total"]
                    _, f_na, i2 = hladmm(R, p, g, n_restarts=a.restarts, seed=s, no_admm=True, **base)
                    _, f_ls, i4 = hladmm(R, p, g, n_restarts=10 ** 9, seed=s, no_admm=True,
                                         time_budget=T, **base)
                    _, f_pg, i3 = hladmm(R, p, g, n_restarts=a.restarts, seed=s, method="pg", **base)
                    _, f_gd, i5 = hladmm(R, p, g, n_restarts=1, seed=s, guided=True,
                                         time_budget=T, **base)
                    vals = dict(full=f_full, no_local=max(i1["f_round"]), no_admm=f_na,
                                ls_matched=f_ls, pg=f_pg, one_restart=i1["f_ref"][0], guided=f_gd)
                    extra = dict(starts_at_limit_full=int(sum(it >= a.max_iter for it in i1["iters"])),
                                 n_starts_ls_matched=len(i4["f_ref"]),
                                 n_roundings_guided=int(i5.get("n_samples", 0)) + 1,   # +1: the deterministic rounding
                                 t_full=round(T, 2), t_admm=round(i1["t_admm"], 2), t_ls=round(i1["t_ls"], 2),
                                 t_no_admm=round(i2["time_total"], 2), t_pg=round(i3["time_total"], 2),
                                 t_ls_matched=round(i4["time_total"], 2), t_guided=round(i5["time_total"], 2))
                    name = f"{a.family}-{N}-{k}"
                    wr.writerow([name, p.size, len(g), s] + [vals[v] for v in VARS] + [extra[c] for c in EXTRA])
                    fh.flush()
                    print(name, "seed", s, {v: round(vals[v], 1) for v in VARS}, extra, flush=True)
                    rows.append(dict(name=name, n=p.size, **vals, **extra))
    summarize(rows)


def main():
    if len(sys.argv) > 1 and sys.argv[1] == "summarize":
        summarize(read_csvs(sys.argv[2:]))
        return
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("family", choices=["bqpgub", "cmisp"])
    ap.add_argument("--dims", type=int, nargs="+", required=True)
    ap.add_argument("--counts", type=int, nargs="+", required=True)
    ap.add_argument("--dir", default=".")
    ap.add_argument("--prefix", default=None)
    ap.add_argument("--out", default=None)
    ap.add_argument("--rho_scale", type=float, default=5.0)
    ap.add_argument("--kappa", type=float, default=1.1)
    ap.add_argument("--max_iter", type=int, default=2000)
    ap.add_argument("--tol", type=float, default=0.001)
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--seed0", type=int, default=0, help="first seed (to continue an earlier run)")
    ap.add_argument("--restarts", type=int, default=20)
    ap.add_argument("--order", choices=["zx", "xz"], default="zx")
    run(ap.parse_args())


if __name__ == "__main__":
    main()
