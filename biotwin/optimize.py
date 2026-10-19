"""Use the digital twin to explore process settings before touching a real reactor.

Each real 14-day run costs weeks and a lot of money; the twin runs in a
fraction of a second. Here we scan two classic levers:

* when to drop the temperature from 37 to 33 deg C (grow first, then produce)
* how much feed to add per day

and look at harvest titer, harvest viability and total antibody made.
Low harvest viability is a problem downstream (more host-cell proteins and
debris to remove), so it is treated as a constraint.
"""
from __future__ import annotations

from dataclasses import replace

import numpy as np
import pandas as pd

from .model import KineticParams, Recipe, simulate_batch


def scan(
    shift_days: list[float | None] = (None, 3, 4, 5, 6, 7, 8, 9),
    feed_fracs: list[float] = (0.02, 0.03, 0.04, 0.05, 0.06),
    params: KineticParams = KineticParams(),
) -> pd.DataFrame:
    rows = []
    for shift in shift_days:
        for feed in feed_fracs:
            h = simulate_batch(replace(Recipe(), temp_shift_d=shift, feed_frac=feed), params).hourly
            last = h.iloc[-1]
            rows.append({
                "temp_shift_day": "none" if shift is None else int(shift),
                "feed_frac": feed,
                "titer_g_l": last["P"] / 1000,
                "harvest_viability": last["viability"],
                "product_g": last["P"] / 1000 * last["volume_l"],
                "peak_vcd": h["Xv"].max(),
            })
    return pd.DataFrame(rows)


def best_setting(results: pd.DataFrame, min_viability: float = 70.0, objective: str = "product_g") -> dict:
    feasible = results[results["harvest_viability"] >= min_viability]
    if feasible.empty:
        return {}
    return feasible.loc[feasible[objective].idxmax()].to_dict()
