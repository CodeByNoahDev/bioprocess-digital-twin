"""Generate a "historical" fleet of batches: normal variation plus known faults.

Real plants have run-to-run variation (cell bank vials differ a little, media
lots differ, operators feed at slightly different times) and occasional
faults. Each batch here gets randomly perturbed kinetics and recipe, may get
one fault, and is then "sampled" daily with realistic analyser noise.
"""
from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np
import pandas as pd

from .model import BatchResult, Conditions, KineticParams, Recipe, nominal_conditions, simulate_batch

VARIABLES = [
    "vcd", "viability", "glucose", "glutamine", "lactate", "ammonia", "osmolality", "titer", "temp", "ph", "do",
]
UNITS = {
    "vcd": "1e6 cells/mL", "viability": "%", "glucose": "mM", "glutamine": "mM", "lactate": "mM",
    "ammonia": "mM", "osmolality": "mOsm/kg", "titer": "mg/L", "temp": "deg C", "ph": "-", "do": "% sat",
}

FAULTS = {
    "temp_excursion": "Jacket/heater fault: +2 deg C for 2 days",
    "ph_probe_drift": "pH probe drifts; controller holds the wrong pH while the probe reads 7.0",
    "low_glutamine_feed": "Wrong feed lot: glutamine at 25 % of spec",
    "do_control_failure": "Sparger fouling: dissolved oxygen drops to 3 % for 3 days",
    "poor_inoculum": "Seed train problem: half the cells at 85 % viability",
}


@dataclass
class Fleet:
    daily: pd.DataFrame                 # one row per batch per day (measured values)
    hourly: dict[str, pd.DataFrame]     # true hourly state per batch
    meta: pd.DataFrame                  # batch_id, fault, fault_onset_d, split


def perturbed_inputs(rng: np.random.Generator) -> tuple[Recipe, KineticParams]:
    """Normal batch-to-batch variability (all lognormal, a few % each)."""
    p, r = KineticParams(), Recipe()
    ln = lambda sd: rng.lognormal(0.0, sd)
    params = replace(
        p,
        mu_max=p.mu_max * ln(0.03),
        qp_beta=p.qp_beta * ln(0.06),
        y_x_glc=p.y_x_glc * ln(0.03),
        kd0=p.kd0 * ln(0.15),
        ki_amm=p.ki_amm * ln(0.05),
        y_lac_glc=p.y_lac_glc * ln(0.05),
    )
    recipe = replace(
        r,
        xv0=r.xv0 * ln(0.06),
        glc0=r.glc0 * ln(0.02),
        gln0=r.gln0 * ln(0.03),
        feed_frac=r.feed_frac * ln(0.02),
        temp_shift_d=r.temp_shift_d + rng.normal(0, 0.08),
    )
    return recipe, params


def run_batch(fault: str | None, rng: np.random.Generator) -> tuple[BatchResult, float | None]:
    """Simulate one batch, optionally with a fault. Returns (result, onset day)."""
    recipe, params = perturbed_inputs(rng)
    conditions, feed_override, onset = nominal_conditions, None, None

    if fault == "temp_excursion":
        onset = rng.uniform(5.5, 8.0)

        def conditions(t: float, r: Recipe) -> Conditions:
            c = nominal_conditions(t, r)
            return replace(c, temp=c.temp + 2.0) if onset <= t < onset + 2 else c

    elif fault == "ph_probe_drift":
        onset = rng.uniform(3.0, 6.0)

        def conditions(t: float, r: Recipe) -> Conditions:
            # The probe reads too high, so the controller doses acid/CO2 until
            # the *probe* says 7.0 - the real pH sinks while the reading looks perfect.
            c = nominal_conditions(t, r)
            return replace(c, ph=c.ph - 0.06 * (t - onset)) if t >= onset else c

    elif fault == "do_control_failure":
        onset = rng.uniform(5.0, 8.0)

        def conditions(t: float, r: Recipe) -> Conditions:
            c = nominal_conditions(t, r)
            return replace(c, do=3.0) if onset <= t < onset + 3 else c

    elif fault == "low_glutamine_feed":
        onset = float(recipe.feed_start_d)
        feed_override = lambda day, r: replace(r, feed_gln=r.feed_gln * 0.25)

    elif fault == "poor_inoculum":
        onset = 0.0
        recipe = replace(recipe, xv0=recipe.xv0 * 0.5, viability0=0.85)

    elif fault is not None:
        raise ValueError(f"Unknown fault {fault!r}")

    return simulate_batch(recipe, params, conditions, feed_override), onset


def measure(result: BatchResult, rng: np.random.Generator) -> pd.DataFrame:
    """Daily offline samples + daily-average online probe readings, with noise."""
    s = result.samples
    h = result.hourly
    n = len(s)
    rel = lambda cv: 1 + rng.normal(0, cv, n)

    # Online probes: average of the 24 h before each sample (day 0: first hour).
    def probe_mean(col: str) -> np.ndarray:
        out = []
        for day in s["time_d"]:
            window = h[(h["time_d"] > day - 1) & (h["time_d"] <= day)] if day > 0 else h.iloc[:1]
            out.append(window[col].mean())
        return np.array(out)

    return pd.DataFrame({
        "day": s["time_d"].round().astype(int),
        "vcd": s["Xv"] * rel(0.05),
        "viability": np.clip(s["viability"] + rng.normal(0, 1.0, n), 0, 100),
        "glucose": s["GLC"] * rel(0.03),
        "glutamine": np.clip(s["GLN"] * rel(0.06) + rng.normal(0, 0.05, n), 0, None),
        "lactate": s["LAC"] * rel(0.04),
        "ammonia": s["AMM"] * rel(0.05),
        "osmolality": s["osmolality"] * rel(0.01),
        "titer": np.clip(s["P"] * rel(0.04) + rng.normal(0, 5, n), 0, None),
        "temp": probe_mean("temp") + rng.normal(0, 0.05, n),
        "ph": probe_mean("ph_measured") + rng.normal(0, 0.01, n),
        "do": probe_mean("do") + rng.normal(0, 1.0, n),
    })


def make_fleet(
    n_train: int = 40, n_test_normal: int = 20, n_per_fault: int = 6, seed: int = 0
) -> Fleet:
    rng = np.random.default_rng(seed)
    plan = (
        [(None, "train")] * n_train
        + [(None, "test")] * n_test_normal
        + [(f, "test") for f in FAULTS for _ in range(n_per_fault)]
    )
    daily, hourly, meta = [], {}, []
    for i, (fault, split) in enumerate(plan):
        batch_id = f"B{i:03d}"
        result, onset = run_batch(fault, rng)
        d = measure(result, rng)
        d.insert(0, "batch_id", batch_id)
        daily.append(d)
        hourly[batch_id] = result.hourly
        meta.append({"batch_id": batch_id, "fault": fault or "normal", "fault_onset_d": onset, "split": split})
    return Fleet(pd.concat(daily, ignore_index=True), hourly, pd.DataFrame(meta))
