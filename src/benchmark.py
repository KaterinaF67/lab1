from __future__ import annotations

import time
import numpy as np
import pandas as pd
from scipy.optimize import linprog
from scipy.sparse import lil_matrix


def _instance(seed: int, n_stock: int, n_orders: int):
    rng = np.random.default_rng(seed)
    stock_lengths = np.round(np.sort(rng.uniform(8.0, 15.0, n_stock)), 2)
    stock_available = rng.integers(20, 60, n_stock).astype(float)
    order_lengths = np.round(np.sort(rng.uniform(1.0, 4.8, n_orders)), 2)
    kerf = 0.02
    end_trim = 0.05

    L = stock_lengths[:, None]
    ell = order_lengths[None, :]
    y = np.floor((L - end_trim + kerf) / (ell + kerf)).astype(int)
    y = np.maximum(y, 1)
    waste = L - y * ell - (y - 1) * kerf
    waste = np.maximum(waste, end_trim)

    feasible = np.zeros((n_stock, n_orders))
    uniform = np.ones(n_orders) / n_orders
    for i in range(n_stock):
        rnd = rng.dirichlet(np.ones(n_orders) * 3.0)
        weights = 0.65 * uniform + 0.35 * rnd
        weights /= weights.sum()
        feasible[i] = stock_available[i] * 0.88 * weights
    demand = np.maximum(1, np.floor((y * feasible).sum(axis=0) * 0.95)).astype(float)

    n = n_stock * n_orders
    A = lil_matrix((n_orders + n_stock, n), dtype=float)
    b = np.zeros(n_orders + n_stock)
    for j in range(n_orders):
        for i in range(n_stock):
            A[j, i * n_orders + j] = -float(y[i, j])
        b[j] = -demand[j]
    for i in range(n_stock):
        for j in range(n_orders):
            A[n_orders + i, i * n_orders + j] = 1.0
        b[n_orders + i] = stock_available[i]
    return waste.reshape(-1), A.tocsr(), b


def run_benchmark(repeats: int = 5) -> pd.DataFrame:
    sizes = [
        (5, 6), (8, 10), (12, 15), (16, 20), (22, 28),
        (30, 40), (40, 55), (55, 75), (70, 95), (90, 120),
    ]
    rows = []
    for idx, (n_stock, n_orders) in enumerate(sizes):
        c, A, b = _instance(20000 + idx, n_stock, n_orders)
        for method in ("highs-ds", "highs-ipm"):
                                                                                  
            linprog(c, A_ub=A, b_ub=b, bounds=(0, None), method=method, options={"presolve": True})
            times, nits, crossovers, objectives = [], [], [], []
            success = True
            for _ in range(repeats):
                t0 = time.perf_counter()
                res = linprog(c, A_ub=A, b_ub=b, bounds=(0, None), method=method, options={"presolve": True})
                times.append(time.perf_counter() - t0)
                nits.append(int(res.nit))
                crossovers.append(int(getattr(res, "crossover_nit", 0)))
                objectives.append(float(res.fun) if res.success else np.nan)
                success = success and bool(res.success)
            rows.append({
                "n_stock": n_stock,
                "n_orders": n_orders,
                "n_variables": n_stock * n_orders,
                "n_constraints": n_stock + n_orders,
                "method": method,
                "median_time_s": float(np.median(times)),
                "median_iterations": int(np.median(nits)),
                "median_crossover_iterations": int(np.median(crossovers)),
                "success": success,
                "objective": float(np.nanmedian(objectives)),
                "repeats": repeats,
            })
    return pd.DataFrame(rows)
