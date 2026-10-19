"""End-to-end demo: simulate a batch fleet, monitor it with MSPC, build a
Raman soft sensor, and use the twin to choose process settings.

Run with:  python -m biotwin            (full run, ~1 minute)
           python -m biotwin --quick    (small run for a smoke test)
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from . import plots
from . import softsensor as ss
from .batches import FAULTS, VARIABLES, make_fleet
from .mspc import EvolvingMonitor
from .optimize import best_setting, scan

MIN_HARVEST_VIABILITY = 70.0


def run_mspc(fleet, figs: Path) -> dict:
    meta = fleet.meta
    train_ids = meta.loc[meta["split"] == "train", "batch_id"]
    test_meta = meta[meta["split"] == "test"].set_index("batch_id")
    train = fleet.daily[fleet.daily["batch_id"].isin(train_ids)]
    test = fleet.daily[fleet.daily["batch_id"].isin(test_meta.index)]

    monitor = EvolvingMonitor(limits="loo").fit(train)
    monitoring = monitor.monitor(test)
    alarms = monitor.first_alarm(monitoring)
    test_meta["alarm_day"] = [alarms[b] for b in test_meta.index]

    # Same model, but limits from the training batches' own statistics.
    naive = EvolvingMonitor(limits="train").fit(train)
    naive_alarms = naive.first_alarm(naive.monitor(test))

    normal = test_meta[test_meta["fault"] == "normal"]
    summary = {
        "n_train_batches": int(len(train_ids)),
        "pca_components_by_day": {d: m.n_components for d, m in monitor.models.items()},
        "false_alarm_rate_normal_batches": {
            "loo_limits": float(normal["alarm_day"].notna().mean()),
            "naive_training_limits": float(np.mean([naive_alarms[b] is not None for b in normal.index])),
        },
        "per_day_out_of_limit_rate_normal": float(
            monitoring[monitoring["batch_id"].isin(normal.index)]["out"].mean()
        ),
        "faults": {},
    }

    contrib_rows = {}
    for fault in FAULTS:
        g = test_meta[test_meta["fault"] == fault]
        detected = g["alarm_day"].notna()
        delays = (g["alarm_day"] - g["fault_onset_d"])[detected]
        shares = []
        for b, row in g.iterrows():
            day = int(row["alarm_day"]) if pd.notna(row["alarm_day"]) else max(monitor.models)
            shares.append(monitor.variable_contributions(fleet.daily[fleet.daily["batch_id"] == b], day).iloc[0])
        mean_share = pd.concat(shares, axis=1).mean(axis=1)
        contrib_rows[fault] = mean_share
        summary["faults"][fault] = {
            "description": FAULTS[fault],
            "detection_rate": float(detected.mean()),
            "mean_days_from_onset_to_alarm": None if delays.empty else round(float(delays.mean()), 2),
            "top_contributing_variables": list(mean_share.sort_values(ascending=False).index[:3]),
        }

    plots.monitoring_examples(str(figs / "mspc_monitoring.png"), monitoring, test_meta.reset_index())
    plots.contribution_heatmap(str(figs / "fault_contributions.png"), pd.DataFrame(contrib_rows).T[VARIABLES])
    return summary


def run_softsensor(fleet, seed: int, figs: Path) -> dict:
    meta = fleet.meta
    train_ids = list(meta.loc[meta["split"] == "train", "batch_id"])
    normal_test = list(meta.loc[(meta["split"] == "test") & (meta["fault"] == "normal"), "batch_id"])
    faulty = list(meta.loc[meta["fault"] != "normal", "batch_id"])
    rng = np.random.default_rng(seed)

    cal = ss.build_dataset(fleet.hourly, train_ids, rng, spike_fraction=0.5)
    test = ss.build_dataset(fleet.hourly, normal_test, rng)

    # 1) choose preprocessing + number of components by grouped CV
    comparison = {}
    for name in ss.PREPROCESSING:
        a, scores = ss.select_components(cal["X"], cal["y_ref"], cal["groups"], preprocessing=name)
        comparison[name] = {"components": a, "rmsecv_mM": round(scores[a - 1], 3)}
    # Same parsimony rule as for components: among options within 5 % of the
    # best RMSECV, take the one needing the fewest PLS components.
    best_rmse = min(c["rmsecv_mM"] for c in comparison.values())
    close = [k for k, c in comparison.items() if c["rmsecv_mM"] <= 1.05 * best_rmse]
    best = min(close, key=lambda k: comparison[k]["components"])
    model = ss.make_model(comparison[best]["components"], best).fit(cal["X"], cal["y_ref"])

    # 2) held-out batches, against the analyser and against the (simulated) truth
    vs_ref = ss.evaluate(model, test["X"], test["y_ref"])
    vs_true = ss.evaluate(model, test["X"], test["y_true"])

    # 3) robustness: abnormal batches, with vs without spiked calibration samples
    robustness = {}
    if faulty:
        stream = ss.build_dataset(fleet.hourly, faulty, rng, every_hours=1)
        unspiked = ss.build_dataset(fleet.hourly, train_ids, rng, spike_fraction=0.0)
        a0, _ = ss.select_components(unspiked["X"], unspiked["y_ref"], unspiked["groups"], preprocessing=best)
        model0 = ss.make_model(a0, best).fit(unspiked["X"], unspiked["y_ref"])
        high = stream["y_true"] > 40
        for label, m in (("without_spiking", model0), ("with_spiking", model)):
            pred = m.predict(stream["X"]).ravel()
            robustness[label] = {
                "rmse_all_faulty_hours_mM": round(ss.rmse(stream["y_true"], pred), 3),
                "rmse_where_glucose_above_40mM": round(ss.rmse(stream["y_true"][high], pred[high]), 3) if high.any() else None,
            }

    # figures
    example = normal_test[0]
    hourly = ss.build_dataset(fleet.hourly, [example], rng, every_hours=1)
    hourly["y_pred"] = model.predict(hourly["X"]).ravel()
    daily = ss.build_dataset(fleet.hourly, [example], rng)
    plots.softsensor_tracking(str(figs / "softsensor_tracking.png"), test["y_true"],
                              model.predict(test["X"]).ravel(), hourly, daily)
    plots.raman_overview(str(figs / "raman_spectra.png"), cal["X"], cal["y_true"], ss.regression_vector(model))

    return {
        "calibration_spectra": int(len(cal["y_ref"])),
        "preprocessing_comparison": comparison,
        "selected_preprocessing": best,
        "test_rmse_vs_analyser_mM": round(vs_ref["rmse"], 3),
        "test_rmse_vs_true_mM": round(vs_true["rmse"], 3),
        "test_r2_vs_true": round(vs_true["r2"], 3),
        "robustness_on_faulty_batches": robustness,
    }


def run_optimisation(figs: Path) -> dict:
    results = scan()
    best = best_setting(results, MIN_HARVEST_VIABILITY)
    nominal = results[(results["temp_shift_day"] == 5) & np.isclose(results["feed_frac"], 0.04)].iloc[0]
    plots.optimisation_map(str(figs / "process_optimisation.png"), results, MIN_HARVEST_VIABILITY, best)
    return {
        "min_harvest_viability": MIN_HARVEST_VIABILITY,
        "best_feasible": {k: (round(v, 3) if isinstance(v, float) else v) for k, v in best.items()},
        "nominal_recipe": {k: (round(v, 3) if isinstance(v, float) else v) for k, v in nominal.to_dict().items()},
    }


def run(out_dir: str = "reports", seed: int = 3, quick: bool = False) -> dict:
    out = Path(out_dir)
    figs = out / "figures"
    figs.mkdir(parents=True, exist_ok=True)
    start = time.perf_counter()

    fleet = make_fleet(
        n_train=20 if quick else 40, n_test_normal=8 if quick else 20, n_per_fault=2 if quick else 6, seed=seed
    )
    plots.fleet_profiles(str(figs / "fleet_profiles.png"), fleet.daily, fleet.meta)
    metrics = {
        "mspc": run_mspc(fleet, figs),
        "soft_sensor": run_softsensor(fleet, seed, figs),
        "optimisation": run_optimisation(figs),
    }
    metrics["runtime_s"] = round(time.perf_counter() - start, 1)
    (out / "metrics.json").write_text(json.dumps(metrics, indent=2, default=str))
    return metrics
