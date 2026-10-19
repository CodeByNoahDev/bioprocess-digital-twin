"""Figures for the README / report."""
from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .batches import FAULTS, UNITS, VARIABLES
from .softsensor import BANDS, WAVENUMBERS, savgol_derivative

FAULT_COLOURS = {
    "normal": "#7f8c8d",
    "temp_excursion": "#e67e22",
    "ph_probe_drift": "#8e44ad",
    "low_glutamine_feed": "#27ae60",
    "do_control_failure": "#2980b9",
    "poor_inoculum": "#c0392b",
}


def fleet_profiles(path: str, daily: pd.DataFrame, meta: pd.DataFrame) -> None:
    """Normal batches as grey lines, one example of each fault in colour."""
    panels = ["vcd", "viability", "glucose", "lactate", "ammonia", "osmolality", "titer", "glutamine"]
    fig, axes = plt.subplots(2, 4, figsize=(16, 7))
    normal_ids = set(meta.loc[meta["fault"] == "normal", "batch_id"])
    examples = meta[meta["fault"] != "normal"].groupby("fault").head(1)
    for ax, var in zip(axes.ravel(), panels):
        for b, g in daily[daily["batch_id"].isin(normal_ids)].groupby("batch_id"):
            ax.plot(g["day"], g[var], color=FAULT_COLOURS["normal"], alpha=0.25, lw=0.8)
        for _, row in examples.iterrows():
            g = daily[daily["batch_id"] == row["batch_id"]]
            ax.plot(g["day"], g[var], color=FAULT_COLOURS[row["fault"]], lw=1.8, label=row["fault"])
        ax.set_title(f"{var} ({UNITS[var]})", fontsize=10)
        ax.set_xlabel("day")
    axes[0, 0].plot([], [], color=FAULT_COLOURS["normal"], label="normal batches")
    fig.legend(*axes[0, 0].get_legend_handles_labels(), loc="lower center", ncol=6, fontsize=9)
    fig.suptitle("Simulated CHO fed-batch fleet: normal variation (grey) vs one batch per fault")
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    fig.savefig(path, dpi=120)
    plt.close(fig)


def monitoring_examples(path: str, monitoring: pd.DataFrame, meta: pd.DataFrame) -> None:
    picks = [meta[meta["fault"] == f].iloc[0] for f in ("normal", "ph_probe_drift", "temp_excursion", "low_glutamine_feed")]
    fig, axes = plt.subplots(2, len(picks), figsize=(16, 6.5), sharex=True)
    for col, row in enumerate(picks):
        g = monitoring[monitoring["batch_id"] == row["batch_id"]].sort_values("day")
        for r, (stat, label) in enumerate((("t2", "Hotelling T$^2$"), ("spe", "SPE (Q)"))):
            ax = axes[r, col]
            ax.semilogy(g["day"], g[stat], "o-", color=FAULT_COLOURS[row["fault"]])
            ax.semilogy(g["day"], g[f"{stat}_limit"], "r--", lw=1, label="99 % limit")
            if pd.notna(row["fault_onset_d"]):
                ax.axvline(row["fault_onset_d"], color="k", ls=":", lw=1, label="fault starts")
            if r == 0:
                ax.set_title(f"{row['batch_id']}: {row['fault']}", fontsize=10)
            ax.set_ylabel(label if col == 0 else "")
            if r == 1:
                ax.set_xlabel("day")
    axes[0, 0].legend(fontsize=8)
    fig.suptitle("Day-by-day MSPC monitoring (statistic above the red line = out of normal pattern)")
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def contribution_heatmap(path: str, contrib: pd.DataFrame) -> None:
    """Rows: fault types; columns: variables; value: mean share of the alarm."""
    fig, ax = plt.subplots(figsize=(11, 4))
    im = ax.imshow(contrib.to_numpy(), cmap="Reds", vmin=0, aspect="auto")
    ax.set_xticks(range(len(contrib.columns)), contrib.columns, rotation=40, ha="right")
    ax.set_yticks(range(len(contrib.index)), contrib.index)
    for i in range(contrib.shape[0]):
        for j in range(contrib.shape[1]):
            v = contrib.iloc[i, j]
            ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=8, color="white" if v > 0.3 else "black")
    ax.set_title("Which variables drive the alarm? (mean contribution share at the alarm day)")
    fig.colorbar(im, ax=ax, fraction=0.03)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def raman_overview(path: str, X: np.ndarray, glucose: np.ndarray, coef: np.ndarray) -> None:
    idx = np.argsort(glucose)[np.linspace(0, len(glucose) - 1, 6).astype(int)]
    colours = plt.cm.viridis(np.linspace(0, 1, len(idx)))
    fig, axes = plt.subplots(3, 1, figsize=(11, 9), sharex=True)
    for i, c in zip(idx, colours):
        axes[0].plot(WAVENUMBERS, X[i], color=c, lw=0.9, label=f"{glucose[i]:.0f} mM")
        axes[1].plot(WAVENUMBERS, savgol_derivative(X[i : i + 1])[0], color=c, lw=0.9)
    axes[0].set_title("Raw simulated Raman spectra (dominated by fluorescence background)")
    axes[0].legend(title="glucose", fontsize=8, ncol=3)
    axes[1].set_title("After Savitzky-Golay 1st derivative (baseline removed, bands visible)")
    axes[2].plot(WAVENUMBERS, coef, color="black", lw=1)
    for centre, _, _ in BANDS["glucose"]:
        axes[2].axvline(centre, color="#27ae60", ls="--", lw=1)
    axes[2].set_title("PLS regression vector - green lines: glucose bands (the model should 'look' here)")
    axes[2].set_xlabel("Raman shift (cm$^{-1}$)")
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def softsensor_tracking(path: str, y_true, y_pred, hourly: dict, daily_ref: dict) -> None:
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5), gridspec_kw={"width_ratios": [1, 2]})
    ax1.scatter(y_true, y_pred, s=12, alpha=0.6)
    lims = [min(y_true.min(), y_pred.min()), max(y_true.max(), y_pred.max())]
    ax1.plot(lims, lims, "k--", lw=1)
    ax1.set_xlabel("True glucose (mM)")
    ax1.set_ylabel("Raman PLS prediction (mM)")
    ax1.set_title("Held-out batches (daily samples)")

    ax2.plot(hourly["time_d"], hourly["y_true"], color="black", lw=1.2, label="true glucose")
    ax2.plot(hourly["time_d"], hourly["y_pred"], color="#2980b9", lw=0.9, alpha=0.8, label="Raman soft sensor (hourly)")
    ax2.plot(daily_ref["time_d"], daily_ref["y_ref"], "o", color="#c0392b", label="offline analyser (daily)")
    ax2.set_xlabel("day")
    ax2.set_ylabel("glucose (mM)")
    ax2.set_title("One held-out batch: the soft sensor sees every feed; the daily sample does not")
    ax2.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def optimisation_map(path: str, results: pd.DataFrame, min_viability: float, best: dict) -> None:
    product = results.pivot(index="temp_shift_day", columns="feed_frac", values="product_g")
    viab = results.pivot(index="temp_shift_day", columns="feed_frac", values="harvest_viability")
    order = [str(v) for v in product.index]
    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(product.to_numpy(), cmap="YlGn", aspect="auto")
    for i in range(product.shape[0]):
        for j in range(product.shape[1]):
            ok = viab.iloc[i, j] >= min_viability
            ax.text(j, i, f"{product.iloc[i, j]:.2f} g\n{viab.iloc[i, j]:.1f} %", ha="center", va="center",
                    fontsize=8, color="black" if ok else "#c0392b", fontweight="normal" if ok else "bold")
    if best:
        bi = list(product.index).index(best["temp_shift_day"])
        bj = list(product.columns).index(best["feed_frac"])
        ax.add_patch(plt.Rectangle((bj - 0.5, bi - 0.5), 1, 1, fill=False, ec="blue", lw=3))
    ax.set_xticks(range(product.shape[1]), [f"{f:.0%}" for f in product.columns])
    ax.set_yticks(range(product.shape[0]), order)
    ax.set_xlabel("daily feed volume (% of starting volume)")
    ax.set_ylabel("temperature shift day")
    ax.set_title(f"Antibody made (g) and harvest viability\nred = viability < {min_viability:.0f} %, blue box = best feasible")
    fig.colorbar(im, ax=ax, label="product (g per batch)")
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
