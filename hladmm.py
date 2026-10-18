#!/usr/bin/env python3
"""HLADMM for BQP-GUB:   max p'x + x'Rx   s.t.  sum_{j in C_k} x_j = 1,  x binary.

Solver (paper's Algorithm 1) + instance reader + experiment runner in one file.

Usage
-----
  python hladmm.py bqpgub --dims 20 50 100 --counts 5 5 5
  python hladmm.py cmisp  --dims 20 50      --counts 5 5 --out cmisp.csv
  python hladmm.py bqpgub --dims 5000 --counts 1 --rho 10 --kappa 1.1 --seed 1

Requires: numpy, scipy.   Instance files are looked up in --dir (default '.').
"""
import argparse, csv, os, re, sys, time
import numpy as np
from scipy.sparse.linalg import eigsh


# ----------------------------------------------------------------------------
# Solver
# ----------------------------------------------------------------------------
def spectral_norm(S):
    """||S||_2 for symmetric S (Lanczos; falls back to power iteration)."""
    try:
        return float(abs(eigsh(S, k=1, which="LM", return_eigenvectors=False, tol=1e-8)[0]))
    except Exception:
        v = np.random.default_rng(0).standard_normal(S.shape[0])
        for _ in range(200):
            v = S @ v
            v /= np.linalg.norm(v)
        return float(abs(v @ (S @ v)))


def proj_simplex(v):
    u = np.sort(v)[::-1]
    css = np.cumsum(u) - 1.0
    k = np.nonzero(u - css / np.arange(1, v.size + 1) > 0)[0][-1]
    return np.maximum(v - css[k] / (k + 1), 0.0)


def hladmm(R, p, groups, rho=10.0, kappa=1.1, max_iter=2000, tol=0.01,
           n_restarts=20, elite_size=5, xupdate="standard", seed=0, verbose=False,
           rho_scale=None, no_admm=False, order="zx", method="admm",
           stop_consec=2, time_budget=None, guided=False, sample_eps=0.15, L_safety=1.01):
    """Returns (x_best, f_best, info).  groups: list of 0-based index arrays.

    xupdate='standard': x+ = Pi_[0,1]((rho z - lam + eta x - grad g(x)) / (rho+eta))
    xupdate='paper'   : x+ = Pi_[0,1](z - lam/rho - grad g(x)/eta)   (formula as printed)
    order     : 'zx' (default): z-step, then x-step (no box), then dual.  Smooth g is in the
                last block, so lam+ = -grad g(x) - eta (x+ - x); this is the ordering the
                nonconvex-ADMM convergence theory needs.
                'xz': x-step (with box), z-step, dual (the original draft's ordering).
    rho_scale : if given, rho = rho_scale * L (L = ||R+R'||_2) instead of the fixed rho
    no_admm   : ablation: skip the ADMM loop (random start -> rounding -> local search)
    method    : 'admm' (default) or 'pg' (projected gradient on the relaxation, step 1/L,
                same rounding + local search; ablation)
    stop_consec : stop when ||x+ - x|| <= tol for this many consecutive iterations
                (2 is what the primal-residual bound in Section 4 needs)
    L_safety  : L = L_safety * ||R+R'||_2.  Use 1.0 to match hladmm_final.py (the paper's tables).
    guided    : relaxation-guided multistart. After each ADMM run, keep drawing random roundings
                from the fractional z (probabilities (1-sample_eps)*z + sample_eps/|C_k| per block),
                run local search from each, until `time_budget` seconds have passed in total.
                Use with n_restarts=1.
    time_budget : if set, keep starting restarts until this many seconds have passed
                (use with n_restarts=10**9 for the time-matched multistart baseline)
    """
    t0 = time.perf_counter()
    rng = np.random.default_rng(seed)
    p = np.asarray(p, float).ravel()
    n, m = p.size, len(groups)
    S = R + R.T                                   # grad g(x) = -(p + S x)
    d = np.diag(R).copy()
    if no_admm:
        L = float("nan")                          # not needed: skip the eigensolver
    else:
        L = L_safety * max(spectral_norm(S), np.finfo(float).eps)   # 1% safety margin by default; 1.0 = exact ||R+R'||_2
    eta = kappa * L                               # eta > L  (assumption A3)
    if rho_scale is not None:
        rho = rho_scale * L
    groups = [np.asarray(g, dtype=np.intp) for g in groups]
    gid = np.empty(n, dtype=np.intp)
    for k, g in enumerate(groups):
        gid[g] = k
    sizes = np.array([g.size for g in groups])
    equal = bool(np.all(sizes == sizes[0]))
    G = np.vstack(groups) if equal else None      # m-by-s index matrix

    def proj_blocks(V):
        Z = np.zeros_like(V)
        if equal:
            s = G.shape[1]
            W = V[G]
            U = -np.sort(-W, axis=1)
            css = np.cumsum(U, axis=1) - 1.0
            k = (U - css / np.arange(1, s + 1) > 0).sum(axis=1)
            theta = css[np.arange(m), k - 1] / k
            Z[G] = np.maximum(W - theta[:, None], 0.0)
        else:
            for g in groups:
                Z[g] = proj_simplex(V[g])
        return Z

    def argmax_blocks(z):
        if equal:
            return G[np.arange(m), z[G].argmax(axis=1)]
        return np.array([g[z[g].argmax()] for g in groups], dtype=np.intp)

    def local_search(sel):
        """One-flip swaps inside blocks; O(1)-per-candidate evaluation with w = S x."""
        x = np.zeros(n); x[sel] = 1.0
        w = S @ x
        improved, passes = True, 0
        while improved:                           # terminates: each move strictly increases f
            improved, passes = False, passes + 1
            for k, g in enumerate(groups):
                i = sel[k]
                delta = p[g] - p[i] + w[g] - w[i] + d[g] + d[i] - S[i, g]
                a = int(delta.argmax())
                if delta[a] > 1e-12:
                    j = g[a]
                    x[i], x[j] = 0.0, 1.0
                    w += S[j] - S[i]              # S symmetric: rows = columns
                    sel[k] = j
                    improved = True
        return x

    elite = []                                    # list of (f, x)

    def add_elite(fb, xb):
        if any(abs(f - fb) < 1e-9 and np.array_equal(xe, xb) for f, xe in elite):
            return
        if len(elite) < elite_size:
            elite.append((fb, xb))
        else:
            iw = min(range(len(elite)), key=lambda q: elite[q][0])
            if fb > elite[iw][0]:
                elite[iw] = (fb, xb)

    def sample_blocks(z):
        """Random rounding: one index per block, drawn from the (smoothed) fractional z."""
        if equal:
            W = np.maximum(z[G], 0.0)
            W = W / np.maximum(W.sum(axis=1, keepdims=True), 1e-300)
            W = (1 - sample_eps) * W + sample_eps / W.shape[1]
            u = rng.random((m, 1))
            idx = np.minimum((np.cumsum(W, axis=1) < u).sum(axis=1), W.shape[1] - 1)
            return G[np.arange(m), idx]
        sel = np.empty(m, dtype=np.intp)
        for k, g in enumerate(groups):
            w = np.maximum(z[g], 0.0); w = w / max(w.sum(), 1e-300)
            w = (1 - sample_eps) * w + sample_eps / g.size
            sel[k] = g[rng.choice(g.size, p=w / w.sum())]
        return sel
    info = dict(L=L, eta=eta, rho=rho, iters=[], pres=[], dres=[], fviol=[], f_round=[], f_ref=[])
    info.update(t_admm=0.0, t_ls=0.0)
    for r in range(n_restarts):
        if time_budget is not None and r > 0 and time.perf_counter() - t0 >= time_budget:
            break
        tr = time.perf_counter()
        x = rng.random(n); z = x.copy(); lam = np.zeros(n)
        t, pres, dres, small = 0, float("nan"), float("nan"), 0
        for t in range(1, (0 if no_admm else max_iter) + 1):
            if not (np.abs(x).max() < 1e8 and np.abs(lam).max() < 1e8):   # also catches NaN/inf
                raise FloatingPointError("HLADMM diverged: increase rho (try rho_scale >= 4.2)")
            gx = -(p + S @ x)
            if method == "pg":
                xn = proj_blocks(x - gx / L)                        # projected gradient, step 1/L
                zn = xn
                dx = np.linalg.norm(xn - x); pres = 0.0; dres = float("nan")
                x, z = xn, zn
                small = small + 1 if dx <= tol else 0
                if small >= stop_consec:
                    break
                continue
            if order == "zx":
                zn = proj_blocks(x + lam / rho)                     # Pi_Delta (block-wise)
                xn = (eta * x + rho * zn - gx - lam) / (eta + rho)  # linearized x-step, no box
            else:
                if xupdate == "paper":
                    xn = z - lam / rho - gx / eta
                else:
                    xn = (rho * z - lam + eta * x - gx) / (rho + eta)
                xn = np.clip(xn, 0.0, 1.0)        # Pi_[0,1]^n
                zn = proj_blocks(xn + lam / rho)  # Pi_Delta (block-wise)
            lam = lam + rho * (xn - zn)           # dual update
            dx = np.linalg.norm(xn - x)
            pres = np.linalg.norm(xn - zn)
            dres = rho * np.linalg.norm(zn - z)
            x, z = xn, zn
            if not np.isfinite(dx):
                raise FloatingPointError("HLADMM diverged: increase rho (try rho_scale >= 2)")
            small = small + 1 if dx <= tol else 0
            if small >= stop_consec:
                break
        info["t_admm"] += time.perf_counter() - tr
        fviol = np.abs(np.bincount(gid, weights=x, minlength=m) - 1).max()

        sel = argmax_blocks(z)                    # group-wise rounding
        xb = np.zeros(n); xb[sel] = 1.0
        f_round = float(p @ xb + xb @ (R @ xb))
        tl = time.perf_counter()
        xb = local_search(sel)                    # best-improvement one-flip refinement
        info["t_ls"] += time.perf_counter() - tl
        fb = float(p @ xb + xb @ (R @ xb))

        add_elite(fb, xb)
        if guided and time_budget is not None:
            n_s = 0
            while time.perf_counter() - t0 < time_budget:
                xs = local_search(sample_blocks(z))
                add_elite(float(p @ xs + xs @ (R @ xs)), xs)
                n_s += 1
            info["n_samples"] = n_s
        for key, val in (("iters", t), ("pres", pres), ("dres", dres),
                         ("fviol", fviol), ("f_round", f_round), ("f_ref", fb)):
            info[key].append(val)
        if verbose:
            print(f"  restart {r+1}: iters={t} f_round={f_round:.4f} f_refined={fb:.4f}")

    f_best, x_best = max(elite, key=lambda e: e[0])
    info["elite_f"] = sorted((f for f, _ in elite), reverse=True)
    info["time_total"] = time.perf_counter() - t0
    return x_best, f_best, info


# ----------------------------------------------------------------------------
# Instance reader  (formats used by the old MATLAB drivers)
#   BQP-GUB: "R = [...]", "p = [...]", "m = k", "C1 = [...]", "C2 = ..."
#   CMISP  : R row by row before the first "C1 = ..." line, then "p = ...", "Ck = ..."
# ----------------------------------------------------------------------------
_NUM = re.compile(r"[-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?")


def _nums(s):
    """All numbers in s; ignores brackets, ',', ';', quotes (transpose) and other junk."""
    return [float(t) for t in _NUM.findall(s)]


def _matrix(s):
    rows = [r for r in s.replace("[", "").replace("]", "").split(";") if r.strip()]
    return np.array([_nums(r) for r in rows])


def read_instance(path):
    """Returns R (n x n), p (n,), groups (list of 0-based index arrays)."""
    if path.endswith(".npz"):                     # written by gen_instances.py
        D = np.load(path)
        p = np.asarray(D["p"], float).ravel(); R = np.asarray(D["R"], float)
        perm = D["perm"] if "perm" in D.files else np.arange(p.size)
        groups = np.split(np.asarray(perm, dtype=np.intp), np.cumsum(D["sizes"])[:-1])
        return R, p, groups
    if path.endswith(".mat"):
        from scipy.io import loadmat
        D = loadmat(path)
        R = np.asarray(D["R"], float); p = np.asarray(D["p"], float).ravel()
        if "groups" in D or "C" in D:
            cells = D["groups"] if "groups" in D else D["C"]
            groups = [np.asarray(c).ravel().astype(int) - 1 for c in cells.ravel()]
        else:
            ks = sorted((k for k in D if re.fullmatch(r"C\d+", k)), key=lambda k: int(k[1:]))
            groups = [np.asarray(D[k]).ravel().astype(int) - 1 for k in ks]
        return R, p, groups

    with open(path) as fh:
        lines = [ln.strip() for ln in fh if ln.strip()]
    rhs = lambda ln: re.sub(r"^\w+\s*=\s*", "", ln)
    c_idx = [i for i, ln in enumerate(lines) if re.match(r"C\d+\s*=", ln)]
    r_line = next((ln for ln in lines if re.match(r"R\s*=", ln)), None)
    p_line = next(ln for ln in lines if re.match(r"p\s*=", ln))
    p = np.array(_nums(rhs(p_line)))
    if r_line is not None:                        # "R = ..." possibly spanning several lines
        r_idx = next(i for i, ln in enumerate(lines) if re.match(r"R\s*=", ln))
        txt, j = rhs(lines[r_idx]), r_idx + 1
        while j < len(lines) and not re.match(r"[A-Za-z_]\w*\s*=", lines[j]):
            txt += " " + lines[j]; j += 1
        vals = _nums(txt)
        if len(vals) != p.size ** 2:
            raise ValueError(f"{path}: R gives {len(vals)} numbers, expected {p.size ** 2}; "
                             f"R line starts with {lines[r_idx][:80]!r}")
        R = np.array(vals).reshape(p.size, p.size)
    else:                                         # CMISP: all numbers before the first C-line
        vals = []
        for i in range(c_idx[0]):
            if not re.match(r"[A-Za-z_]\w*\s*=", lines[i]):
                vals += _nums(lines[i])
        if len(vals) == p.size ** 2 + 1:          # optional leading size line
            vals = vals[1:]
        if len(vals) != p.size ** 2:
            raise ValueError(f"{path}: found {len(vals)} matrix numbers, expected {p.size ** 2}")
        R = np.array(vals).reshape(p.size, p.size)
    groups = [np.array(_nums(rhs(lines[i])), dtype=int) - 1 for i in c_idx]
    n = p.size
    assert R.shape == (n, n), f"R is {R.shape}, p has {n} entries"
    assert sorted(np.concatenate(groups).tolist()) == list(range(n)), "groups do not partition 1..n"
    return R, p, groups


# ----------------------------------------------------------------------------
# Experiment runner
# ----------------------------------------------------------------------------
def run(family, dims, counts, folder, out, **kw):
    prefix = {"bqpgub": "bqpgubs", "cmisp": "cmisp"}[family]
    cols = ["name", "n", "m", "minCk", "maxCk", "value", "iters", "Fviol", "Pres", "Dres", "time"]
    with open(out, "w", newline="") as fh:
        wr = csv.writer(fh); wr.writerow(cols)
        for N, cnt in zip(dims, counts):
            for k in range(1, cnt + 1):
                prefs = [prefix] + [q for q in ("bqpgub", "bqpgubs") if family == "bqpgub" and q != prefix]
                path = next((os.path.join(folder, f"{q}-{N}-{k}{e}") for q in prefs for e in (".txt", ".mat", ".npz")
                             if os.path.isfile(os.path.join(folder, f"{q}-{N}-{k}{e}"))), None)
                if path is None:
                    print(f"WARNING: cannot find {prefix}-{N}-{k}.txt/.mat in {folder}; skipped", flush=True)
                    continue
                name = f"{family}-{N}-{k}"
                print("Solving", name, flush=True)
                R, p, groups = read_instance(path)

                t = time.perf_counter()           # time the solver call only
                x, f, info = hladmm(R, p, groups, **kw)
                elapsed = time.perf_counter() - t

                # independent checks
                assert np.all((x == 0) | (x == 1)), "x not binary"
                assert all(x[g].sum() == 1 for g in groups), "GUB violated"
                fcheck = float(p @ x + x @ R @ x)
                assert abs(fcheck - f) <= 1e-6 * max(1, abs(f)), "value mismatch"

                r = int(np.argmax(info["f_ref"]))
                sizes = [g.size for g in groups]
                wr.writerow([name, p.size, len(groups), min(sizes), max(sizes), fcheck,
                             info["iters"][r], info["fviol"][r], info["pres"][r],
                             info["dres"][r], round(elapsed, 3)])
                fh.flush()
                print(f"  value={fcheck:.4f} iters={info['iters'][r]} time={elapsed:.2f}s "
                      f"(L={info['L']:.1f}, eta={info['eta']:.1f}, rho={info['rho']:.1f})", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("family", choices=["bqpgub", "cmisp"])
    ap.add_argument("--dims", type=int, nargs="+", required=True)
    ap.add_argument("--counts", type=int, nargs="+", required=True)
    ap.add_argument("--dir", default=".")
    ap.add_argument("--out", default=None)
    ap.add_argument("--rho", type=float, default=10.0)
    ap.add_argument("--rho_scale", type=float, default=None, help="rho = rho_scale * L (overrides --rho)")
    ap.add_argument("--kappa", type=float, default=1.1)
    ap.add_argument("--max_iter", type=int, default=2000)
    ap.add_argument("--tol", type=float, default=0.01)
    ap.add_argument("--restarts", type=int, default=20)
    ap.add_argument("--elite", type=int, default=5)
    ap.add_argument("--xupdate", choices=["standard", "paper"], default="standard")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--order", choices=["zx", "xz"], default="zx")
    a = ap.parse_args()
    run(a.family, a.dims, a.counts, a.dir, a.out or f"{a.family}_results.csv",
        rho=a.rho, kappa=a.kappa, max_iter=a.max_iter, tol=a.tol, n_restarts=a.restarts,
        elite_size=a.elite, xupdate=a.xupdate, seed=a.seed, rho_scale=a.rho_scale, order=a.order)
