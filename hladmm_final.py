#!/usr/bin/env python3
"""
Final HLADMM for BQP-GUB.

    max  p'x + x'Rx
    s.t. sum_{j in C_k} x_j = 1,  k = 1,...,m
         x_j in {0,1}

Continuous iteration (z -> x -> lambda):

    z^{t+1}      = Pi_Delta(x^t + lambda^t / rho)
    x^{t+1}      = (eta x^t + rho z^{t+1} - grad g(x^t) - lambda^t) / (eta + rho)
    lambda^{t+1} = lambda^t + rho (x^{t+1} - z^{t+1})

with g(x) = -p'x - x'Rx, grad g(x) = -(p + (R + R')x),
L = ||R + R'||_2, eta = kappa L, and rho = c L.

Usage
-----
python hladmm_final.py bqpgub --dims 20 50 100 --counts 2 2 2 --dir . --out pilot.csv
python hladmm_final.py cmisp --dims 20 50 --counts 2 2 --dir . --out cmisp_pilot.csv
"""

import argparse
import csv
import os
import re
import time

import numpy as np
from scipy.sparse.linalg import eigsh


# ---------------------------------------------------------------------------
# Numerical utilities
# ---------------------------------------------------------------------------

def spectral_norm(S):
    """Compute ||S||_2 for a symmetric matrix S."""
    try:
        return float(abs(eigsh(
            S, k=1, which="LM",
            return_eigenvectors=False,
            tol=1e-8,
        )[0]))
    except Exception:
        v = np.random.default_rng(0).standard_normal(S.shape[0])
        v /= np.linalg.norm(v)
        for _ in range(200):
            v = S @ v
            norm_v = np.linalg.norm(v)
            if norm_v == 0:
                return 0.0
            v /= norm_v
        return float(abs(v @ (S @ v)))


def condition_A(L, eta, rho):
    """delta(eta, rho) of (14); assumption (A) of the paper is delta > 0."""
    delta = eta + (rho - L) / 2.0 - (2.0 * eta ** 2 + 2.0 * (L + eta) ** 2) / rho
    return float(delta), bool(delta > 0)


def project_simplex(v):
    """Project a vector onto {u >= 0, sum(u) = 1}."""
    v = np.asarray(v, dtype=float)

    if v.size == 0:
        raise ValueError("Cannot project an empty block.")

    if not np.all(np.isfinite(v)):
        raise ValueError(f"Non-finite input to simplex projection: {v}")

    u = np.sort(v)[::-1]
    css = np.cumsum(u) - 1.0
    ind = np.arange(1, v.size + 1)

    valid = np.nonzero(u - css / ind > 0)[0]

    if valid.size == 0:
        raise ValueError(f"Simplex projection failed for input: {v}")

    rho_index = valid[-1]
    theta = css[rho_index] / (rho_index + 1)

    return np.maximum(v - theta, 0.0)


def project_blocks(v, groups):
    """Project each GUB block onto its probability simplex."""
    z = np.zeros_like(v)

    for group in groups:
        z[group] = project_simplex(v[group])

    return z


def select_blocks(z, groups):
    """Select the largest component in each block."""
    return np.array(
        [group[z[group].argmax()] for group in groups],
        dtype=int,
    )


# ---------------------------------------------------------------------------
# Instance reader
# ---------------------------------------------------------------------------

_NUMBER = re.compile(
    r"[-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?"
)


def _numbers(text):
    return [float(value) for value in _NUMBER.findall(text)]


def read_instance(path):
    """
    Read a BQP-GUB or CMISP instance.

    Supported BQP-GUB text format:

        p = [...]
        R = [...]
        C1 = [...]
        C2 = [...]
        ...

    Supported CMISP text format:

        <rows of R>
        p = [...]
        C1 = [...]
        C2 = [...]
        ...

    Returns p, R, groups, where group indices are zero-based.
    """
    if path.endswith(".npz"):                      # written by gen_instances.py
        data = np.load(path)
        p = np.asarray(data["p"], dtype=float).ravel()
        R = np.asarray(data["R"], dtype=float)
        perm = data["perm"] if "perm" in data.files else np.arange(p.size)
        groups = np.split(np.asarray(perm, dtype=int), np.cumsum(data["sizes"])[:-1])
        return p, R, groups

    if path.endswith(".mat"):                      # MATLAB: R, p, groups (cell, 1-based)
        from scipy.io import loadmat
        data = loadmat(path)
        R = np.asarray(data["R"], dtype=float)
        p = np.asarray(data["p"], dtype=float).ravel()
        if "groups" in data or "C" in data:
            cells = data["groups"] if "groups" in data else data["C"]
            groups = [np.asarray(c).ravel().astype(int) - 1 for c in cells.ravel()]
        else:
            keys = sorted((k for k in data if re.fullmatch(r"C\d+", k)),
                          key=lambda k: int(k[1:]))
            groups = [np.asarray(data[k]).ravel().astype(int) - 1 for k in keys]
        return p, R, groups

    with open(path, encoding="utf-8") as file:
        lines = [line.strip() for line in file if line.strip()]

    c_lines = [
        index for index, line in enumerate(lines)
        if re.match(r"C\d+\s*=", line)
    ]

    if not c_lines:
        raise ValueError(f"No GUB blocks Ck found in {path}.")

    p_line = next(
        line for line in lines if re.match(r"p\s*=", line)
    )
    p = np.asarray(
        _numbers(re.sub(r"^\w+\s*=\s*", "", p_line)),
        dtype=float,
    )

    r_line = next(
        (line for line in lines if re.match(r"R\s*=", line)),
        None,
    )

    if r_line is not None:
        r_index = next(
            index for index, line in enumerate(lines)
            if re.match(r"R\s*=", line)
        )

        text = re.sub(r"^\w+\s*=\s*", "", lines[r_index])
        j = r_index + 1

        while j < len(lines) and not re.match(
            r"[A-Za-z_]\w*\s*=", lines[j]
        ):
            text += " " + lines[j]
            j += 1

        values = _numbers(text)

        if len(values) != p.size ** 2:
            raise ValueError(
                f"Expected {p.size ** 2} entries in R, "
                f"found {len(values)} in {path}."
            )

        R = np.asarray(values, dtype=float).reshape(p.size, p.size)

    else:
        values = []

        for index in range(c_lines[0]):
            if not re.match(r"[A-Za-z_]\w*\s*=", lines[index]):
                values.extend(_numbers(lines[index]))

        # Some CMISP files contain a leading instance-size value.
        if len(values) == p.size ** 2 + 1:
            values = values[1:]

        if len(values) != p.size ** 2:
            raise ValueError(
                f"Expected {p.size ** 2} matrix entries in {path}, "
                f"found {len(values)}."
            )

        R = np.asarray(values, dtype=float).reshape(p.size, p.size)

    groups = [
        np.asarray(
            _numbers(re.sub(r"^\w+\s*=\s*", "", lines[index])),
            dtype=int,
        ) - 1
        for index in c_lines
    ]

    n = p.size

    if R.shape != (n, n):
        raise ValueError(
            f"R has shape {R.shape}, expected ({n}, {n}) in {path}."
        )

    all_indices = np.concatenate(groups)

    if sorted(all_indices.tolist()) != list(range(n)):
        raise ValueError(
            f"The blocks in {path} do not form a partition of 1,...,n."
        )

    for k, group in enumerate(groups):
        if group.size == 0:
            raise ValueError(f"Block C{k + 1} is empty in {path}.")

    if not np.all(np.isfinite(R)):
        raise ValueError(f"R contains non-finite values in {path}.")

    if not np.all(np.isfinite(p)):
        raise ValueError(f"p contains non-finite values in {path}.")

    return p, R, groups


# ---------------------------------------------------------------------------
# HLADMM solver
# ---------------------------------------------------------------------------

def hladmm(
    p,
    R,
    groups,
    kappa=1.1,
    c=5.0,
    rho=None,
    max_iter=2000,
    tol=0.01,
    n_restarts=5,
    elite_size=5,
    seed=0,
    use_local_search=True,
    verbose=False,
):
    """
    Solve a BQP-GUB instance using the revised HLADMM method.

    Returns x_best, f_best, info.
    """
    start_time = time.perf_counter()
    rng = np.random.default_rng(seed)

    p = np.asarray(p, dtype=float).ravel()
    R = np.asarray(R, dtype=float)

    # The objective depends only on S = R + R' (x'Rx = x'Sx / 2); no other n-by-n copy is kept.
    S = R + R.T
    diag_R = 0.5 * np.diag(S)

    n = p.size
    m = len(groups)

    groups = [
        np.asarray(group, dtype=int).ravel()
        for group in groups
    ]

    all_indices = np.concatenate(groups)

    if sorted(all_indices.tolist()) != list(range(n)):
        raise ValueError("The blocks do not form a partition of 0,...,n-1.")

    for k, group in enumerate(groups):
        if group.size == 0:
            raise ValueError(f"Block C{k + 1} is empty.")

    gid = np.empty(n, dtype=int)
    for k, group in enumerate(groups):
        gid[group] = k

    sizes = np.array([group.size for group in groups])
    equal = bool(np.all(sizes == sizes[0]))
    G = np.vstack(groups) if equal else None      # m-by-s index matrix

    def proj_blocks(v):
        if not equal:
            return project_blocks(v, groups)
        s_ = G.shape[1]
        W = v[G]
        U = -np.sort(-W, axis=1)
        css = np.cumsum(U, axis=1) - 1.0
        k_ = (U - css / np.arange(1, s_ + 1) > 0).sum(axis=1)
        theta = css[np.arange(m), k_ - 1] / k_
        z_ = np.zeros_like(v)
        z_[G] = np.maximum(W - theta[:, None], 0.0)
        return z_

    def pick_blocks(z_):
        if not equal:
            return select_blocks(z_, groups)
        return G[np.arange(m), z_[G].argmax(axis=1)]

    L = spectral_norm(S)
    eta = kappa * L

    if kappa <= 1.0:
        raise ValueError("kappa must be strictly greater than 1.")

    rho = c * L if rho is None else rho
    delta_A, cond_A = condition_A(L, eta, rho)

    def objective(x):
        return float(p @ x + 0.5 * (x @ (S @ x)))

    def local_one_flip(selected):
        """
        Best-improvement one-flip search.

        With S = R + R', w = S x, replacing selected i by j has gain

            delta = p_j - p_i + w_j - w_i + R_jj + R_ii - (R_ij + R_ji).
        """
        x = np.zeros(n)
        x[selected] = 1.0
        w = S @ x

        improved = True

        while improved:
            improved = False

            for k, block in enumerate(groups):
                if block.size == 1:
                    continue

                i = selected[k]

                delta = (
                    p[block]
                    - p[i]
                    + w[block]
                    - w[i]
                    + diag_R[block]
                    + diag_R[i]
                    - S[i, block]
                )

                delta[block == i] = -np.inf

                best_local = int(delta.argmax())
                best_gain = delta[best_local]

                if best_gain > 1e-9:
                    j = block[best_local]

                    x[i] = 0.0
                    x[j] = 1.0

                    w += S[j] - S[i]               # S is symmetric: rows are contiguous
                    selected[k] = j
                    improved = True

        return x

    elite = []

    info = {
        "L": L,
        "eta": eta,
        "rho": rho,
        "delta": delta_A,
        "condA": cond_A,
        "iters": [],
        "pres": [],
        "dres": [],
        "fviol": [],
        "f_round": [],
        "f_ref": [],
    }

    for restart in range(n_restarts):
        x = rng.random(n)
        z = x.copy()
        lam = np.zeros(n)

        iterations = 0
        pres = np.nan
        dres = np.nan
        small_count = 0

        for iteration in range(1, max_iter + 1):
            if not (
                np.all(np.isfinite(x))
                and np.all(np.isfinite(lam))
            ):
                raise FloatingPointError(
                    "HLADMM diverged. "
                    "Increase c (for example, c=10) or reduce kappa."
                )

            grad_g = -(p + S @ x)

            z_new = proj_blocks(x + lam / rho)

            x_new = (
                eta * x + rho * z_new - grad_g - lam
            ) / (eta + rho)

            lam_new = lam + rho * (x_new - z_new)

            dx = np.linalg.norm(x_new - x)
            pres = np.linalg.norm(x_new - z_new)
            dres = rho * np.linalg.norm(z_new - z)

            x, z, lam = x_new, z_new, lam_new
            iterations = iteration

            if not np.isfinite(dx):
                raise FloatingPointError(
                    "HLADMM diverged. "
                    "Increase c (for example, c=10) or reduce kappa."
                )

            if dx <= tol:
                small_count += 1
                if small_count >= 2:
                    break
            else:
                small_count = 0

        selected = pick_blocks(z)

        x_binary = np.zeros(n)
        x_binary[selected] = 1.0
        f_round = objective(x_binary)

        if use_local_search:
            x_refined = local_one_flip(selected.copy())
        else:
            x_refined = x_binary

        f_refined = objective(x_refined)

        block_sums = np.bincount(gid, weights=x, minlength=m)
        f_violation = float(np.max(np.abs(block_sums - 1.0)))

        duplicate = any(
            abs(value - f_refined) < 1e-9
            and np.array_equal(solution, x_refined)
            for value, solution in elite
        )

        if not duplicate:
            if len(elite) < elite_size:
                elite.append((f_refined, x_refined))
            else:
                worst = min(
                    range(len(elite)),
                    key=lambda index: elite[index][0],
                )
                if f_refined > elite[worst][0]:
                    elite[worst] = (f_refined, x_refined)

        info["iters"].append(iterations)
        info["pres"].append(float(pres))
        info["dres"].append(float(dres))
        info["fviol"].append(f_violation)
        info["f_round"].append(f_round)
        info["f_ref"].append(f_refined)

        if verbose:
            print(
                f"restart {restart + 1}: "
                f"iters={iterations}, "
                f"f_round={f_round:.6f}, "
                f"f_refined={f_refined:.6f}",
                flush=True,
            )

    if not elite:
        raise RuntimeError("No feasible refined solution was produced.")

    f_best, x_best = max(elite, key=lambda item: item[0])

    info["elite_values"] = sorted(
        (value for value, _ in elite),
        reverse=True,
    )
    info["time_total"] = time.perf_counter() - start_time

    return x_best, f_best, info


# ---------------------------------------------------------------------------
# Experiment runner
# ---------------------------------------------------------------------------

def find_instance(data_dir, family, dimension, instance_id):
    prefixes = ["bqpgubs", "bqpgub"] if family == "bqpgub" else ["cmisp"]
    for prefix in prefixes:
        for ext in (".txt", ".mat", ".npz"):
            path = os.path.join(data_dir, f"{prefix}-{dimension}-{instance_id}{ext}")
            if os.path.isfile(path):
                return path
    return None


def run_experiments(
    family,
    dimensions,
    counts,
    data_dir=".",
    output_file="results.csv",
    **solver_options,
):
    max_iter = solver_options.get("max_iter", 2000)

    columns = [
        "Filename", "n", "m", "minCk", "maxCk", "HLADMM", "Iter",
        "F-Violation", "P-Resi", "D-Resi", "Time", "Starts-at-limit",
        "L", "eta", "rho", "condA",
    ]

    with open(output_file, "w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(columns)

        for dimension, count in zip(dimensions, counts):
            for instance_id in range(1, count + 1):
                name = f"{family}-{dimension}-{instance_id}"
                path = find_instance(data_dir, family, dimension, instance_id)

                if path is None:
                    print(f"WARNING: missing {name} in {data_dir}; skipped.", flush=True)
                    continue

                print(f"Solving {name}  ({os.path.basename(path)})", flush=True)

                p, R, groups = read_instance(path)

                solver_start = time.perf_counter()
                x_best, f_best, info = hladmm(p, R, groups, **solver_options)
                elapsed = time.perf_counter() - solver_start

                # Independent checks.
                if not np.all((x_best == 0) | (x_best == 1)):
                    raise AssertionError("Returned solution is not binary.")

                for group in groups:
                    if x_best[group].sum() != 1:
                        raise AssertionError("GUB constraint violated.")

                f_check = float(p @ x_best + x_best @ (R @ x_best))

                if abs(f_check - f_best) > 1e-6 * max(1.0, abs(f_best)):
                    raise AssertionError("Objective-value mismatch.")

                best_restart = int(np.argmax(info["f_ref"]))
                sizes = [g.size for g in groups]
                at_limit = int(sum(t >= max_iter for t in info["iters"]))

                writer.writerow([
                    name, p.size, len(groups), min(sizes), max(sizes), f_check,
                    info["iters"][best_restart], info["fviol"][best_restart],
                    info["pres"][best_restart], info["dres"][best_restart],
                    round(elapsed, 3), at_limit,
                    info["L"], info["eta"], info["rho"], info["condA"],
                ])
                file.flush()

                print(
                    f"  value={f_check:.6f}, iterations={info['iters'][best_restart]}, "
                    f"starts at limit={at_limit}/{len(info['iters'])}, "
                    f"time={elapsed:.3f}s, rho/L={info['rho'] / info['L']:.2f}, condA={info['condA']}",
                    flush=True,
                )


# ---------------------------------------------------------------------------
# Command-line interface
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Revised HLADMM for BQP-GUB and CMISP instances."
    )

    parser.add_argument(
        "family",
        choices=["bqpgub", "cmisp"],
    )
    parser.add_argument(
        "--dims",
        type=int,
        nargs="+",
        required=True,
    )
    parser.add_argument(
        "--counts",
        type=int,
        nargs="+",
        required=True,
    )
    parser.add_argument("--dir", default=".")
    parser.add_argument("--out", default=None)
    parser.add_argument("--kappa", type=float, default=1.1)
    parser.add_argument("--c", type=float, default=5.0)
    parser.add_argument("--max-iter", type=int, default=2000)
    parser.add_argument("--tol", type=float, default=0.01)
    parser.add_argument("--restarts", type=int, default=5)
    parser.add_argument("--elite", type=int, default=5)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--no-local-search", action="store_true")
    parser.add_argument("--verbose", action="store_true")

    args = parser.parse_args()

    if len(args.dims) != len(args.counts):
        parser.error("--dims and --counts must have the same length.")

    run_experiments(
        family=args.family,
        dimensions=args.dims,
        counts=args.counts,
        data_dir=args.dir,
        output_file=args.out or f"{args.family}_results.csv",
        kappa=args.kappa,
        c=args.c,
        max_iter=args.max_iter,
        tol=args.tol,
        n_restarts=args.restarts,
        elite_size=args.elite,
        seed=args.seed,
        use_local_search=not args.no_local_search,
        verbose=args.verbose,
    )


if __name__ == "__main__":
    main()
