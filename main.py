from __future__ import annotations

from pathlib import Path
import json
import platform
import subprocess
import sys

import numpy as np
import pandas as pd
import scipy
import scipy.optimize._highspy._core as highs_core

from src.data import generate_instance, save_instance, load_instance, SEED, VARIANT
from src.model import solve_instance, save_solver_log
from src.verify import verify_from_csv, save_verification
from src.benchmark import run_benchmark
from src.plots import plot_scenarios, plot_stock, plot_benchmark_time, plot_benchmark_iterations

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
RESULTS = ROOT / "results"
PLOTS = RESULTS / "plots"


def _save_output(prefix: str, output) -> None:
    output.solution.to_csv(RESULTS / f"{prefix}_solution.csv", index=False, float_format="%.9f")
    output.demand_summary.to_csv(RESULTS / f"{prefix}_demand_check.csv", index=False, float_format="%.9f")
    output.stock_summary.to_csv(RESULTS / f"{prefix}_stock_check.csv", index=False, float_format="%.9f")


def main() -> None:
    RESULTS.mkdir(parents=True, exist_ok=True)
    PLOTS.mkdir(parents=True, exist_ok=True)

                                                         
    instance = generate_instance(seed=SEED, n_stock=5, n_orders=6)
    save_instance(instance, DATA)
    instance = load_instance(DATA)

    highs_version = f"{highs_core.HIGHS_VERSION_MAJOR}.{highs_core.HIGHS_VERSION_MINOR}.{highs_core.HIGHS_VERSION_PATCH}"

    base = solve_instance(instance, method="highs-ds", presolve=True)
    if not base.result.success:
        raise RuntimeError(f"Base solve failed: {base.result.message}")
    _save_output("base", base)
    base.shadow_prices.to_csv(RESULTS / "shadow_prices.csv", index=False, float_format="%.9f")
    save_solver_log(base, RESULTS / "solver_base.log", scipy.__version__, highs_version, presolve=True)

                                                                                      
                                                                                     
    with (RESULTS / "solver_base_full.log").open("w", encoding="utf-8") as log_file:
        subprocess.run(
            [sys.executable, "-m", "src.highs_diagnostic", "--data", str(DATA), "--method", "highs-ds"],
            cwd=ROOT,
            stdout=log_file,
            stderr=subprocess.STDOUT,
            check=True,
        )

    verification = verify_from_csv(instance, RESULTS / "base_solution.csv", expected_objective=float(base.result.fun))
    save_verification(verification, RESULTS / "verification_base.json")
    if not verification["feasible"]:
        raise RuntimeError("Independent verification failed")

                                                                                              
    stock_shadow = base.shadow_prices[(base.shadow_prices["kind"] == "stock") & base.shadow_prices["active"]].copy()
    stock_shadow = stock_shadow.sort_values("economic_shadow_price", ascending=False)
    top_resources = [s.replace("stock_", "") for s in stock_shadow["constraint"].tolist()]
    if not top_resources:
        raise RuntimeError("No active stock constraint found for scenario analysis")
    deficit_resource = top_resources[0]

    idx_map = {sid: i for i, sid in enumerate(instance.stock["stock_id"])}
    deficit_idx = idx_map[deficit_resource]

    deficit_instance = load_instance(DATA)
    deficit_instance.stock.loc[deficit_idx, "available_blanks"] *= 0.80
    deficit = solve_instance(deficit_instance, method="highs-ds", presolve=True)
    if not deficit.result.success:
        raise RuntimeError(f"Deficit scenario failed: {deficit.result.message}")
    _save_output("deficit", deficit)
    save_solver_log(deficit, RESULTS / "solver_deficit.log", scipy.__version__, highs_version, presolve=True)
    verification_deficit = verify_from_csv(
        deficit_instance, RESULTS / "deficit_solution.csv", expected_objective=float(deficit.result.fun)
    )
    save_verification(verification_deficit, RESULTS / "verification_deficit.json")
    if not verification_deficit["feasible"]:
        raise RuntimeError("Independent verification failed for deficit scenario")

    stress_instance = load_instance(DATA)
    stress_resources = top_resources[:2] if len(top_resources) >= 2 else top_resources
    for sid in stress_resources:
        stress_instance.stock.loc[idx_map[sid], "available_blanks"] *= 0.70
    stress = solve_instance(stress_instance, method="highs-ds", presolve=True)
    if not stress.result.success:
        raise RuntimeError(f"Stress scenario failed: {stress.result.message}")
    _save_output("stress", stress)
    save_solver_log(stress, RESULTS / "solver_stress.log", scipy.__version__, highs_version, presolve=True)
    verification_stress = verify_from_csv(
        stress_instance, RESULTS / "stress_solution.csv", expected_objective=float(stress.result.fun)
    )
    save_verification(verification_stress, RESULTS / "verification_stress.json")
    if not verification_stress["feasible"]:
        raise RuntimeError("Independent verification failed for stress scenario")

    base_obj = float(base.result.fun)
    def_obj = float(deficit.result.fun)
    stress_obj = float(stress.result.fun)
    base_cap = float(instance.stock.loc[deficit_idx, "available_blanks"])
    deficit_cap = float(deficit_instance.stock.loc[deficit_idx, "available_blanks"])
    raw_marginal = float(
        base.shadow_prices.loc[base.shadow_prices["constraint"] == f"stock_{deficit_resource}", "raw_marginal_dz_db"].iloc[0]
    )
    predicted_delta = raw_marginal * (deficit_cap - base_cap)
    actual_delta = def_obj - base_obj

    scenarios = pd.DataFrame([
        {
            "scenario": "base",
            "status": "optimal",
            "objective_waste_m": base_obj,
            "delta_vs_base_m": 0.0,
            "change": "исходные данные",
            "iterations": int(base.result.nit),
        },
        {
            "scenario": "deficit",
            "status": "optimal",
            "objective_waste_m": def_obj,
            "delta_vs_base_m": actual_delta,
            "change": f"запас {deficit_resource} -20%",
            "iterations": int(deficit.result.nit),
        },
        {
            "scenario": "stress",
            "status": "optimal",
            "objective_waste_m": stress_obj,
            "delta_vs_base_m": stress_obj - base_obj,
            "change": "запасы " + ", ".join(stress_resources) + " -30%",
            "iterations": int(stress.result.nit),
        },
    ])
    scenarios.to_csv(RESULTS / "scenarios.csv", index=False, float_format="%.9f")

    benchmark = run_benchmark(repeats=5)
    benchmark.to_csv(RESULTS / "benchmark_methods.csv", index=False, float_format="%.9f")

    plot_stock(base.stock_summary, PLOTS / "stock_usage.png")
    plot_scenarios(scenarios, PLOTS / "scenario_objective.png")
    plot_benchmark_time(benchmark, PLOTS / "benchmark_time.png")
    plot_benchmark_iterations(benchmark, PLOTS / "benchmark_iterations.png")

    run_info = {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "scipy": scipy.__version__,
        "highs": highs_version,
        "variant": VARIANT,
        "seed": SEED,
        "base_objective_waste_m": base_obj,
        "independent_verification": verification,
        "deficit_resource": deficit_resource,
        "shadow_prediction_delta_m": predicted_delta,
        "actual_deficit_delta_m": actual_delta,
    }
    (RESULTS / "run_info.json").write_text(json.dumps(run_info, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"Variant: {VARIANT}; seed: {SEED}")
    print(f"Base status: {base.result.message}")
    print(f"Minimum waste: {base_obj:.6f} m")
    print(f"Independent verification: {verification['feasible']}")
    print(f"Deficit: {deficit_resource} -20%, objective {def_obj:.6f} m")
    print(f"Stress: {', '.join(stress_resources)} -30%, objective {stress_obj:.6f} m")
    print("Extra task benchmark saved to results/benchmark_methods.csv")


if __name__ == "__main__":
    main()
