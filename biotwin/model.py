"""Mechanistic model of a CHO-cell fed-batch bioreactor (the "digital twin").

Monoclonal antibodies (mAbs) - most of the biologics made in Irish pharma
plants - are produced by Chinese Hamster Ovary (CHO) cells grown in stirred
bioreactors for ~14 days. This module simulates one batch with ordinary
differential equations (ODEs):

    states           units
    Xv   viable cells     1e6 cells/mL
    Xd   dead cells       1e6 cells/mL
    GLC  glucose          mM
    GLN  glutamine        mM
    LAC  lactate          mM
    AMM  ammonia          mM
    P    antibody titer   mg/L
    SALT other solutes    mOsm/kg (salts etc. added with medium and feed)

Kinetics are Monod-type (growth slows as a nutrient runs low) with inhibition
by the waste products lactate and ammonia, a Luedeking-Piret production term,
and temperature / pH / dissolved-oxygen (DO) / osmolality effects. Osmolality
matters because feed is concentrated: every feed raises it, and above
~350 mOsm/kg cells grow more slowly and die faster. That is what stops "just
feed more" from being the answer. Parameter values are
illustrative - chosen to give textbook-looking CHO profiles (peak ~15-20e6
cells/mL, ~2-4 g/L titer) - not fitted to a specific cell line.

Feeding is "bolus" fed-batch: once a day, after sampling, a volume of
concentrated feed is added (and glucose is topped up if it is low).
"""
from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Callable

import numpy as np
import pandas as pd
from scipy.integrate import solve_ivp

STATES = ("Xv", "Xd", "GLC", "GLN", "LAC", "AMM", "P", "SALT")


def osmolality(glc, gln, lac, amm, salt):
    """mOsm/kg: each dissolved particle counts; lactate counts twice because
    the base added to neutralise it (pH control) brings in sodium."""
    return salt + glc + gln + 2 * lac + amm


@dataclass(frozen=True)
class KineticParams:
    mu_max: float = 1.1        # 1/day, max specific growth rate at 37 C
    k_glc: float = 1.0         # mM, Monod constant for glucose
    k_gln: float = 0.3         # mM, Monod constant for glutamine
    ki_lac: float = 60.0       # mM, lactate inhibition constant
    ki_amm: float = 20.0       # mM, ammonia inhibition constant
    kd0: float = 0.01          # 1/day, basal death rate
    kd_amm: float = 0.12       # 1/day, extra death caused by ammonia (max)
    k_d_amm: float = 12.0      # mM, ammonia level giving half the max extra death
    kd_starve: float = 0.8     # 1/day, extra death when glucose runs out
    k_starve: float = 0.1      # mM
    k_lysis: float = 0.05      # 1/day, dead cells breaking apart
    y_x_glc: float = 0.55      # (1e6 cells/mL) made per mM glucose
    m_glc: float = 0.10        # mM/day per 1e6 cells/mL, maintenance glucose
    y_x_gln: float = 1.6       # (1e6 cells/mL) made per mM glutamine
    m_gln: float = 0.015       # mM/day per 1e6 cells/mL
    k_gln_deg: float = 0.05    # 1/day, glutamine breaks down to ammonia in media
    y_lac_glc: float = 1.4     # mol lactate per mol glucose used for growth
    q_lac_max: float = 0.8     # mM/day per 1e6 cells/mL, lactate uptake (max)
    k_lac: float = 4.0         # mM
    mu_shift: float = 0.25     # 1/day, growth rate below which cells start eating lactate
    y_amm_gln: float = 0.6     # mol ammonia per mol glutamine consumed
    qp_alpha: float = 10.0     # pg/cell per unit growth (growth-associated production)
    qp_beta: float = 20.0      # pg/cell/day (non-growth-associated production)
    temp_growth: float = 0.12  # growth  ~ exp(temp_growth * (T - 37))
    temp_death: float = 0.10   # death   ~ exp(temp_death * (T - 37))
    temp_qp: float = -0.08     # qp      ~ exp(temp_qp * (T - 37)): cooler = more productive
    ph_opt: float = 7.0
    ph_width: float = 0.35
    k_do: float = 2.0          # % air saturation; CHO only suffers at very low DO
    osm_ref: float = 360.0     # mOsm/kg above which osmotic stress starts
    osm_scale: float = 150.0   # growth ~ exp(-(osm - osm_ref) / osm_scale)
    kd_osm: float = 0.04       # 1/day extra death per 100 mOsm/kg above osm_ref


@dataclass(frozen=True)
class Recipe:
    duration_d: int = 14
    v0_l: float = 1.0
    xv0: float = 0.4           # 1e6 cells/mL at inoculation
    viability0: float = 0.97
    glc0: float = 33.0         # mM (~6 g/L)
    gln0: float = 6.0
    lac0: float = 1.0
    amm0: float = 0.6
    salt0: float = 260.0       # mOsm/kg from salts etc. in the base medium
    feed_start_d: int = 3
    feed_frac: float = 0.04    # daily feed volume as a fraction of starting volume
    feed_glc: float = 220.0    # mM in the feed medium
    feed_gln: float = 60.0
    feed_salt: float = 450.0   # mOsm/kg of other solutes in the (concentrated) feed
    glc_trigger: float = 15.0  # top glucose up if a sample reads below this...
    glc_target: float = 25.0   # ...back to this
    glc_stock: float = 2500.0  # mM, concentrated glucose solution
    temp_c: float = 37.0
    temp_shift_d: float | None = 5.0
    temp_after_c: float = 33.0
    ph_setpoint: float = 7.0
    do_setpoint: float = 40.0  # % air saturation


@dataclass(frozen=True)
class Conditions:
    """What the cells actually experience vs what the probes report."""
    temp: float
    ph: float
    do: float
    ph_measured: float


ConditionFn = Callable[[float, Recipe], Conditions]


def nominal_conditions(t: float, recipe: Recipe) -> Conditions:
    shifted = recipe.temp_shift_d is not None and t >= recipe.temp_shift_d
    temp = recipe.temp_after_c if shifted else recipe.temp_c
    return Conditions(temp, recipe.ph_setpoint, recipe.do_setpoint, recipe.ph_setpoint)


def specific_rates(y: np.ndarray, c: Conditions, p: KineticParams) -> dict[str, float]:
    """Per-cell rates at one instant (the biology lives here)."""
    xv, xd, glc, gln, lac, amm, _, salt = np.maximum(y, 0.0)
    f_temp = np.exp(p.temp_growth * (c.temp - 37))
    f_ph = np.exp(-(((c.ph - p.ph_opt) / p.ph_width) ** 2))
    f_do = c.do / (p.k_do + c.do)
    osm_excess = max(osmolality(glc, gln, lac, amm, salt) - p.osm_ref, 0.0)
    f_osm = np.exp(-osm_excess / p.osm_scale)

    mu = (
        p.mu_max * f_temp * f_ph * f_do * f_osm
        * glc / (p.k_glc + glc)
        * gln / (p.k_gln + gln)
        * p.ki_lac / (p.ki_lac + lac)
        * p.ki_amm / (p.ki_amm + amm)
    )
    # Ammonia toxicity is sigmoidal (Hill n=2): harmless at low levels, then
    # rising sharply. Off-optimum pH adds stress on top.
    amm_tox = p.kd_amm * amm**2 / (p.k_d_amm**2 + amm**2)
    mu_d = (p.kd0 + amm_tox) * np.exp(p.temp_death * (c.temp - 37)) / max(f_ph, 0.2)
    mu_d += p.kd_starve * p.k_starve / (p.k_starve + glc) + p.kd_osm * osm_excess / 100

    q_glc = mu / p.y_x_glc + p.m_glc * glc / (p.k_glc + glc)
    q_gln = mu / p.y_x_gln + p.m_gln * gln / (p.k_gln + gln)
    # Lactate is made by fast glycolysis during growth and consumed once growth
    # slows (the "lactate shift" seen in most CHO fed-batches).
    q_lac = p.y_lac_glc * mu / p.y_x_glc - p.q_lac_max * lac / (p.k_lac + lac) * np.exp(-mu / p.mu_shift)
    q_amm = p.y_amm_gln * q_gln
    q_p = (p.qp_alpha * mu + p.qp_beta) * np.exp(p.temp_qp * (c.temp - 37))
    return {"mu": mu, "mu_d": mu_d, "q_glc": q_glc, "q_gln": q_gln, "q_lac": q_lac, "q_amm": q_amm, "q_p": q_p}


def derivatives(t: float, y: np.ndarray, recipe: Recipe, p: KineticParams, conditions: ConditionFn) -> np.ndarray:
    # Uptake terms all contain C / (K + C), so they fade to zero as a nutrient
    # runs out and concentrations cannot go negative.
    xv, xd, _, gln = np.maximum(y[:4], 0.0)
    r = specific_rates(y, conditions(t, recipe), p)
    return np.array([
        (r["mu"] - r["mu_d"]) * xv,
        r["mu_d"] * xv - p.k_lysis * xd,
        -r["q_glc"] * xv,
        -r["q_gln"] * xv - p.k_gln_deg * gln,
        r["q_lac"] * xv,
        r["q_amm"] * xv + p.k_gln_deg * gln,
        r["q_p"] * xv,  # pg/cell/day x 1e6 cells/mL = mg/L/day
        0.0,            # other solutes only change when feed is added
    ])


@dataclass
class BatchResult:
    hourly: pd.DataFrame            # true process state every hour
    samples: pd.DataFrame           # true state at each daily sample (taken *before* feeding)
    feeds: list[dict] = field(default_factory=list)


def _feed(y: np.ndarray, volume: float, add_l: float, glc: float, gln: float, salt: float) -> tuple[np.ndarray, float]:
    """Mix a bolus of feed into the reactor (everything else is diluted)."""
    new_v = volume + add_l
    y = y * volume / new_v
    y[2] += glc * add_l / new_v
    y[3] += gln * add_l / new_v
    y[7] += salt * add_l / new_v
    return y, new_v


def simulate_batch(
    recipe: Recipe = Recipe(),
    params: KineticParams = KineticParams(),
    conditions: ConditionFn = nominal_conditions,
    feed_override: Callable[[int, Recipe], Recipe] | None = None,
) -> BatchResult:
    """Integrate one fed-batch run day by day, feeding after each daily sample.

    feed_override(day, recipe) can return a modified recipe for that day's
    feed (used to inject feeding faults).
    """
    xv0 = recipe.xv0
    y = np.array([
        xv0, xv0 * (1 - recipe.viability0) / recipe.viability0,
        recipe.glc0, recipe.gln0, recipe.lac0, recipe.amm0, 0.0, recipe.salt0,
    ])
    volume = recipe.v0_l
    rows, samples, feeds = [], [], []

    for day in range(recipe.duration_d):
        samples.append(_row(float(day), y, volume, recipe, conditions))
        # --- feeding happens right after the morning sample ---
        if day >= recipe.feed_start_d:
            r = feed_override(day, recipe) if feed_override else recipe
            y, volume = _feed(y, volume, r.feed_frac * r.v0_l, r.feed_glc, r.feed_gln, r.feed_salt)
            top_up = 0.0
            if y[2] < r.glc_trigger:
                # volume of stock needed to bring glucose to target
                top_up = volume * (r.glc_target - y[2]) / (r.glc_stock - r.glc_target)
                y, volume = _feed(y, volume, top_up, r.glc_stock, 0.0, 0.0)
            feeds.append({"day": day, "feed_l": r.feed_frac * r.v0_l, "glucose_top_up_l": top_up})

        t_eval = day + np.arange(0, 24) / 24
        sol = solve_ivp(
            derivatives, (day, day + 1), y, t_eval=t_eval,
            args=(recipe, params, conditions), method="LSODA", rtol=1e-6, atol=1e-8,
        )
        for k, t in enumerate(sol.t):
            rows.append(_row(t, sol.y[:, k], volume, recipe, conditions))
        y = np.maximum(sol.y[:, -1], 0.0)

    final = _row(float(recipe.duration_d), y, volume, recipe, conditions)
    rows.append(final)
    samples.append(final)  # harvest sample
    return BatchResult(pd.DataFrame(rows), pd.DataFrame(samples), feeds)


def _row(t: float, y: np.ndarray, volume: float, recipe: Recipe, conditions: ConditionFn) -> dict:
    c = conditions(t, recipe)
    state = dict(zip(STATES, np.maximum(y, 0.0)))
    total = state["Xv"] + state["Xd"]
    return {
        "time_d": t,
        **state,
        "viability": 100 * state["Xv"] / total if total > 0 else 0.0,
        "osmolality": osmolality(state["GLC"], state["GLN"], state["LAC"], state["AMM"], state["SALT"]),
        "volume_l": volume,
        "temp": c.temp,
        "ph": c.ph,
        "ph_measured": c.ph_measured,
        "do": c.do,
    }


def with_changes(obj, **changes):
    """Small helper so callers can write with_changes(params, mu_max=0.9)."""
    return replace(obj, **changes)
