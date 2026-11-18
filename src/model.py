from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import time
import numpy as np
import pandas as pd
from scipy.optimize import linprog
from scipy.sparse import lil_matrix

from .data import Instance


@dataclass
class SolveOutput:
    result: object
    solution: pd.DataFrame
    demand_summary: pd.DataFrame
    stock_summary: pd.DataFrame
    shadow_prices: pd.DataFrame
    elapsed_s: float
    method: str


def build_lp(instance: Instance):
    I = len(instance.stock)
    J = len(instance.orders)
    n = I * J

    c = instance.waste_matrix.reshape(-1).astype(float)
    A = lil_matrix((J + I, n), dtype=float)
    b = np.zeros(J + I, dtype=float)

                                                                
    for j in range(J):
        for i in range(I):
            A[j, i * J + j] = -float(instance.yield_matrix[i, j])
        b[j] = -float(instance.orders.loc[j, "demand_pieces"])

                                           
    for i in range(I):
        for j in range(J):
            A[J + i, i * J + j] = 1.0
        b[J + i] = float(instance.stock.loc[i, "available_blanks"])

    return c, A.tocsr(), b


def solve_instance(
    instance: Instance,
    method: str = "highs-ds",
    presolve: bool = True,
    disp: bool = False,
) -> SolveOutput:
    c, A_ub, b_ub = build_lp(instance)
    t0 = time.perf_counter()
    res = linprog(
        c,
        A_ub=A_ub,
        b_ub=b_ub,
        bounds=(0, None),
        method=method,
        options={"presolve": presolve, "disp": disp},
    )
    elapsed = time.perf_counter() - t0

    if not res.success:
        return SolveOutput(res, pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), elapsed, method)

    I = len(instance.stock)
    J = len(instance.orders)
    x = np.asarray(res.x).reshape(I, J)

    rows = []
    for i in range(I):
        for j in range(J):
            rows.append({
                "stock_id": instance.stock.loc[i, "stock_id"],
                "order_id": instance.orders.loc[j, "order_id"],
                "x_avg_blanks": x[i, j],
                "pieces_per_blank": int(instance.yield_matrix[i, j]),
                "produced_pieces": x[i, j] * instance.yield_matrix[i, j],
                "waste_m_per_blank": instance.waste_matrix[i, j],
                "waste_m": x[i, j] * instance.waste_matrix[i, j],
            })
    solution = pd.DataFrame(rows)

    produced = (instance.yield_matrix * x).sum(axis=0)
    demand = instance.orders["demand_pieces"].to_numpy(dtype=float)
    demand_summary = pd.DataFrame({
        "order_id": instance.orders["order_id"],
        "demand_pieces": demand,
        "produced_pieces": produced,
        "surplus_pieces": produced - demand,
    })

    used = x.sum(axis=1)
    available = instance.stock["available_blanks"].to_numpy(dtype=float)
    slack = available - used
                                                                    
    slack[np.abs(slack) < 1e-9] = 0.0
    stock_summary = pd.DataFrame({
        "stock_id": instance.stock["stock_id"],
        "available_blanks": available,
        "used_avg_blanks": used,
        "slack_blanks": slack,
    })

    marg = np.asarray(res.ineqlin.marginals)
    residual = np.asarray(res.ineqlin.residual)
    demand_marg = marg[:J]
    stock_marg = marg[J:]
    demand_res = residual[:J]
    stock_res = residual[J:]

    shadow_rows = []
    for j in range(J):
        shadow_rows.append({
            "constraint": f"demand_{instance.orders.loc[j, 'order_id']}",
            "kind": "demand",
            "active": bool(demand_res[j] <= 1e-7),
            "slack": float(demand_res[j]),
            "raw_marginal_dz_db": float(demand_marg[j]),
            "economic_shadow_price": float(-demand_marg[j]),
            "interpretation": "increase in minimum waste per +1 demanded piece",
        })
    for i in range(I):
        shadow_rows.append({
            "constraint": f"stock_{instance.stock.loc[i, 'stock_id']}",
            "kind": "stock",
            "active": bool(stock_res[i] <= 1e-7),
            "slack": float(stock_res[i]),
            "raw_marginal_dz_db": float(stock_marg[i]),
            "economic_shadow_price": float(-stock_marg[i]),
            "interpretation": "decrease in minimum waste per +1 available blank",
        })
    shadow_prices = pd.DataFrame(shadow_rows)

    return SolveOutput(res, solution, demand_summary, stock_summary, shadow_prices, elapsed, method)


def save_solver_log(output: SolveOutput, path: Path, scipy_version: str, highs_version: str, presolve: bool = True) -> None:
    res = output.result
    lines = [
        "HiGHS solver log summary",
        f"method={output.method}",
        f"scipy_version={scipy_version}",
        f"highs_version={highs_version}",
        f"presolve={presolve}",
        f"status_code={getattr(res, 'status', None)}",
        f"success={getattr(res, 'success', None)}",
        f"message={getattr(res, 'message', '')}",
        f"objective={getattr(res, 'fun', None)}",
        f"iterations={getattr(res, 'nit', None)}",
        f"crossover_iterations={getattr(res, 'crossover_nit', None)}",
        f"elapsed_s={output.elapsed_s:.9f}",
    ]
    if getattr(res, "success", False):
        lines += [
            f"max_inequality_residual={float(np.max(np.maximum(-np.asarray(res.ineqlin.residual), 0.0))):.3e}",
            f"min_variable={float(np.min(res.x)):.9f}",
        ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
