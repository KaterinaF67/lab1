from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import json
import numpy as np
import pandas as pd

N = 25
Q = 1
Y = 26
VARIANT = 1 + ((17 * N + 7 * Q + Y) % 20)
SEED = 10000 + 100 * Q + N
KERF_M = 0.02
END_TRIM_M = 0.05


@dataclass
class Instance:
    stock: pd.DataFrame
    orders: pd.DataFrame
    yield_matrix: np.ndarray
    waste_matrix: np.ndarray
    kerf_m: float = KERF_M
    end_trim_m: float = END_TRIM_M


def generate_instance(seed: int = SEED, n_stock: int = 5, n_orders: int = 6) -> Instance:
    rng = np.random.default_rng(seed)

    stock_lengths = np.round(np.sort(rng.uniform(8.2, 14.0, n_stock)), 2)
    stock_available = rng.integers(18, 34, n_stock).astype(float)
    order_lengths = np.round(np.sort(rng.uniform(1.1, 4.6, n_orders)), 2)

    L = stock_lengths[:, None]
    ell = order_lengths[None, :]
    yield_matrix = np.floor((L - END_TRIM_M + KERF_M) / (ell + KERF_M)).astype(int)
    yield_matrix = np.maximum(yield_matrix, 1)
    waste_matrix = L - yield_matrix * ell - (yield_matrix - 1) * KERF_M

                                                                        
    feasible_plan = np.zeros((n_stock, n_orders), dtype=float)
    uniform = np.ones(n_orders) / n_orders
    for i in range(n_stock):
        random_weights = rng.dirichlet(np.ones(n_orders) * 2.0)
        weights = 0.5 * random_weights + 0.5 * uniform
        weights /= weights.sum()
        feasible_plan[i] = stock_available[i] * 0.90 * weights

    production = (yield_matrix * feasible_plan).sum(axis=0)
    demand = np.maximum(1, np.floor(production * 0.95)).astype(int)

    stock = pd.DataFrame({
        "stock_id": [f"S{i+1}" for i in range(n_stock)],
        "length_m": stock_lengths,
        "available_blanks": stock_available,
    })
    orders = pd.DataFrame({
        "order_id": [f"O{j+1}" for j in range(n_orders)],
        "piece_length_m": order_lengths,
        "demand_pieces": demand,
    })
    return Instance(stock, orders, yield_matrix, waste_matrix)


def save_instance(instance: Instance, data_dir: Path) -> None:
    data_dir.mkdir(parents=True, exist_ok=True)
    instance.stock.to_csv(data_dir / "stock_types.csv", index=False)
    instance.orders.to_csv(data_dir / "orders.csv", index=False)

    yield_df = pd.DataFrame(
        instance.yield_matrix,
        index=instance.stock["stock_id"],
        columns=instance.orders["order_id"],
    )
    yield_df.index.name = "stock_id"
    yield_df.to_csv(data_dir / "yield_matrix.csv")

    waste_df = pd.DataFrame(
        instance.waste_matrix,
        index=instance.stock["stock_id"],
        columns=instance.orders["order_id"],
    )
    waste_df.index.name = "stock_id"
    waste_df.to_csv(data_dir / "waste_matrix_m.csv", float_format="%.6f")

    metadata = {
        "student_number_N": N,
        "group_index_Q": Q,
        "year_Y": Y,
        "variant_formula": "1 + ((17*N + 7*Q + Y) mod 20)",
        "variant": VARIANT,
        "seed_formula": "10000 + 100*Q + N",
        "seed": SEED,
        "kerf_m": KERF_M,
        "end_trim_m": END_TRIM_M,
        "model_note": "continuous averaged LP: x_ij may be fractional",
    }
    (data_dir / "metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def load_instance(data_dir: Path) -> Instance:
    stock = pd.read_csv(data_dir / "stock_types.csv")
    orders = pd.read_csv(data_dir / "orders.csv")
    yield_matrix = pd.read_csv(data_dir / "yield_matrix.csv", index_col=0).to_numpy(dtype=int)
    waste_matrix = pd.read_csv(data_dir / "waste_matrix_m.csv", index_col=0).to_numpy(dtype=float)
    return Instance(stock, orders, yield_matrix, waste_matrix)
