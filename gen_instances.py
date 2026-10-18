#!/usr/bin/env python3
"""Seeded BQP-GUB instance generator.  Writes <prefix>-<n>-<k>.npz (read directly by hladmm.py).

Reproduce the MATLAB script (randn, symmetrized, random blocks of size 5, n=7500, m=1500):
  python gen_instances.py --dims 7500 --counts 5 --bs 5 --dist normal --symmetric --perm
Uniform[-50,50] entries, consecutive blocks of size 2:
  python gen_instances.py --dims 2500 3000 --counts 5 5 --bs 2 --dist uniform
Options: --bs block size (m = n/bs), --dist normal|uniform, --symmetric (R=(R+R')/2),
         --perm (blocks are a random partition, not consecutive), --integer (uniform only,
         stored as int8), --seed (instance (n,k) uses stream [seed,n,k]), --force.
"""
import argparse, os
import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("--dims", type=int, nargs="+", required=True)
ap.add_argument("--counts", type=int, nargs="+", required=True)
ap.add_argument("--bs", type=int, default=2)
ap.add_argument("--dist", choices=["uniform", "normal"], default="uniform")
ap.add_argument("--symmetric", action="store_true")
ap.add_argument("--perm", action="store_true")
ap.add_argument("--integer", action="store_true")
ap.add_argument("--dir", default=".")
ap.add_argument("--prefix", default="bqpgubs")
ap.add_argument("--seed", type=int, default=12345)
ap.add_argument("--force", action="store_true")
a = ap.parse_args()
if a.integer and a.dist != "uniform":
    ap.error("--integer only with --dist uniform")

for N, cnt in zip(a.dims, a.counts):
    for k in range(1, cnt + 1):
        path = os.path.join(a.dir, f"{a.prefix}-{N}-{k}.npz")
        if os.path.exists(path) and not a.force:
            print("exists, skipped:", path); continue
        rng = np.random.default_rng([a.seed, N, k])
        if a.dist == "normal":
            R = rng.standard_normal((N, N)); p = rng.standard_normal(N)
        else:
            R = rng.uniform(-50, 50, (N, N)); p = rng.uniform(-50, 50, N)
        if a.symmetric:
            R += R.T; R *= 0.5
        if a.integer:
            R = np.rint(R).astype(np.int8); p = np.rint(p).astype(np.int8)
        perm = rng.permutation(N) if a.perm else np.arange(N)
        sizes = np.full(-(-N // a.bs), a.bs); sizes[-1] = N - a.bs * (len(sizes) - 1)
        np.savez(path, R=R, p=p, sizes=sizes, perm=perm.astype(np.int32), seed=a.seed, bs=a.bs)
        print("wrote", path, f"(m={len(sizes)}, block size {a.bs}, dist={a.dist})", flush=True)
