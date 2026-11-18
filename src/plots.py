from __future__ import annotations

from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd


def plot_scenarios(scenarios: pd.DataFrame, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    ax.bar(scenarios["scenario"], scenarios["objective_waste_m"])
    ax.set_ylabel("Минимальные отходы, м")
    ax.set_xlabel("Сценарий")
    ax.set_title("Сценарный анализ")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=170)
    plt.close(fig)


def plot_stock(stock_summary: pd.DataFrame, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    x = range(len(stock_summary))
    ax.bar(x, stock_summary["available_blanks"], label="Доступно")
    ax.bar(x, stock_summary["used_avg_blanks"], label="Использовано")
    ax.set_xticks(list(x), stock_summary["stock_id"])
    ax.set_ylabel("Среднее число заготовок")
    ax.set_title("Использование запаса труб")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=170)
    plt.close(fig)


def plot_benchmark_time(bench: pd.DataFrame, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    for method, grp in bench.groupby("method"):
        ax.plot(grp["n_variables"], grp["median_time_s"] * 1000.0, marker="o", label=method)
    ax.set_xlabel("Число переменных")
    ax.set_ylabel("Медианное время, мс")
    ax.set_title("Доп. задание: время решения")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=170)
    plt.close(fig)


def plot_benchmark_iterations(bench: pd.DataFrame, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    for method, grp in bench.groupby("method"):
        ax.plot(grp["n_variables"], grp["median_iterations"], marker="o", label=method)
    ax.set_xlabel("Число переменных")
    ax.set_ylabel("Итерации")
    ax.set_title("Доп. задание: число итераций")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=170)
    plt.close(fig)
