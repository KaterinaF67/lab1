from __future__ import annotations

from pathlib import Path
import json
import numpy as np
import pandas as pd

from .data import Instance


def verify_from_csv(instance: Instance, solution_csv: Path, expected_objective: float | None = None, tol: float = 1e-7):
    sol = pd.read_csv(solution_csv)
    I = len(instance.stock)
    J = len(instance.orders)
    x = np.zeros((I, J), dtype=float)

    stock_map = {s: i for i, s in enumerate(instance.stock["stock_id"])}
    order_map = {o: j for j, o in enumerate(instance.orders["order_id"])}
    for row in sol.itertuples(index=False):
        x[stock_map[row.stock_id], order_map[row.order_id]] = float(row.x_avg_blanks)

    nonnegative_violation = max(0.0, -float(np.min(x)))

    stock_used = x.sum(axis=1)
    stock_cap = instance.stock["available_blanks"].to_numpy(dtype=float)
    stock_violation = np.maximum(stock_used - stock_cap, 0.0)

    produced = (instance.yield_matrix * x).sum(axis=0)
    demand = instance.orders["demand_pieces"].to_numpy(dtype=float)
    demand_violation = np.maximum(demand - produced, 0.0)

    objective = float(np.sum(instance.waste_matrix * x))
    objective_difference = None if expected_objective is None else abs(objective - expected_objective)

    report = {
        "tolerance": tol,
        "nonnegative_violation": nonnegative_violation,
        "max_stock_violation": float(stock_violation.max(initial=0.0)),
        "max_demand_violation": float(demand_violation.max(initial=0.0)),
        "objective_recomputed": objective,
        "objective_difference": objective_difference,
        "feasible": bool(
            nonnegative_violation <= tol
            and stock_violation.max(initial=0.0) <= tol
            and demand_violation.max(initial=0.0) <= tol
            and (objective_difference is None or objective_difference <= 1e-6)
        ),
    }
    return report


def save_verification(report: dict, path: Path) -> None:
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
