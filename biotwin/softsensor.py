"""Raman spectroscopy soft sensor for glucose (Process Analytical Technology).

Offline glucose is measured once a day, but glucose is consumed and fed all
day. An in-line Raman probe takes a spectrum every few minutes; a chemometric
model (PLS regression) turns each spectrum into a glucose estimate. That is a
"soft sensor": software that estimates something you cannot measure directly
often enough.

The spectra here are simulated from the batch state: each molecule adds its
own Raman bands, on top of a large, drifting fluorescence background, laser
power fluctuations and detector noise. Band positions are approximate and
for illustration only - this is not a reference spectral library.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.signal import savgol_filter
from sklearn.cross_decomposition import PLSRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import GroupKFold, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer

WAVENUMBERS = np.arange(400.0, 1801.0, 2.0)  # cm^-1, the "fingerprint" region

# (centre cm^-1, width cm^-1, relative height)
BANDS = {
    "glucose": [(1125, 12, 1.0), (1060, 14, 0.6), (911, 10, 0.4), (1365, 15, 0.3), (520, 12, 0.35)],
    "lactate": [(856, 10, 1.0), (1040, 12, 0.5), (1420, 14, 0.45), (1310, 12, 0.2)],
    "glutamine": [(1410, 12, 0.5), (1090, 10, 0.4), (1330, 12, 0.3)],
    "ammonia": [(1450, 20, 0.3)],
    "protein": [(1003, 5, 1.0), (1660, 25, 1.2), (1250, 30, 0.7), (1450, 20, 0.8)],
}
# Signal per unit concentration (arbitrary units per mM, per 1e6 cells/mL, per g/L)
SENSITIVITY = {"glucose": 0.012, "lactate": 0.010, "glutamine": 0.008, "ammonia": 0.004}
CELL_SIGNAL, TITER_SIGNAL = 0.015, 0.06


def pure_spectrum(component: str) -> np.ndarray:
    s = np.zeros_like(WAVENUMBERS)
    for centre, width, height in BANDS[component]:
        s += height * np.exp(-0.5 * ((WAVENUMBERS - centre) / width) ** 2)
    return s


PURE = {name: pure_spectrum(name) for name in BANDS}


def simulate_spectra(states, rng: np.random.Generator, batch_background: float = 1.0) -> np.ndarray:
    """One spectrum per row of `states` (a DataFrame with GLC, LAC, ... columns)."""
    n = len(states)
    w = (WAVENUMBERS - WAVENUMBERS.min()) / np.ptp(WAVENUMBERS)
    spectra = (
        np.outer(states["GLC"], PURE["glucose"]) * SENSITIVITY["glucose"]
        + np.outer(states["LAC"], PURE["lactate"]) * SENSITIVITY["lactate"]
        + np.outer(states["GLN"], PURE["glutamine"]) * SENSITIVITY["glutamine"]
        + np.outer(states["AMM"], PURE["ammonia"]) * SENSITIVITY["ammonia"]
        + np.outer(states["Xv"] + 0.5 * states["Xd"], PURE["protein"]) * CELL_SIGNAL
        + np.outer(states["P"] / 1000, PURE["protein"]) * TITER_SIGNAL
    )
    # Fluorescence background: big, smooth, grows with biomass and debris, and
    # differs between batches (media lot, probe window fouling).
    biomass = (states["Xv"] + states["Xd"]).to_numpy()[:, None]
    amplitude = batch_background * (1.0 + 0.08 * biomass) * rng.lognormal(0, 0.05, (n, 1))
    slope = rng.normal(0.6, 0.15, (n, 1))
    background = amplitude * (1.5 * np.exp(-2.0 * w) + slope * w)
    gain = rng.normal(1.0, 0.03, (n, 1))  # laser power / focus drift
    return gain * (spectra + background) + rng.normal(0, 0.003, spectra.shape)


def savgol_derivative(X: np.ndarray, window: int = 15, poly: int = 2) -> np.ndarray:
    """First derivative with Savitzky-Golay smoothing: removes the flat/sloped baseline."""
    return savgol_filter(X, window, poly, deriv=1, axis=1)


def snv(X: np.ndarray) -> np.ndarray:
    """Standard Normal Variate: centre and scale each spectrum on its own.

    Removes multiplicative effects such as laser power changes.
    """
    return (X - X.mean(axis=1, keepdims=True)) / X.std(axis=1, keepdims=True)


PREPROCESSING = {
    "raw": [],
    "savgol_d1": [("savgol", savgol_derivative)],
    "savgol_d1+snv": [("savgol", savgol_derivative), ("snv", snv)],
}


def make_model(n_components: int, preprocessing: str = "savgol_d1") -> Pipeline:
    steps = [(name, FunctionTransformer(fn)) for name, fn in PREPROCESSING[preprocessing]]
    # scale=False: spectra share units, so only mean-centre (standard in chemometrics).
    steps.append(("pls", PLSRegression(n_components=n_components, scale=False)))
    return Pipeline(steps)


def rmse(y_true, y_pred) -> float:
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def build_dataset(
    hourly: dict, batch_ids, rng: np.random.Generator, spike_fraction: float = 0.0, every_hours: int = 24
) -> dict[str, np.ndarray]:
    """Spectra + reference glucose for a set of batches.

    every_hours=24 mimics one offline sample a day; every_hours=1 gives the
    continuous in-line stream. spike_fraction > 0 adds "spiked" calibration
    samples: a sample is split and known amounts of glucose are added. That
    widens the calibration range and - crucially - breaks the correlation
    between glucose and everything else that trends with culture age, so PLS
    has to learn glucose itself rather than a proxy for time.
    """
    X, y_ref, y_true, groups, times = [], [], [], [], []
    for b in batch_ids:
        h = hourly[b].iloc[::every_hours].copy()
        if spike_fraction > 0:
            spiked = h.sample(frac=spike_fraction, random_state=int(rng.integers(1 << 31))).copy()
            spiked["GLC"] += rng.uniform(5, 30, len(spiked))
            h = pd.concat([h, spiked])
        background = rng.lognormal(0, 0.15)  # each batch/probe has its own fluorescence level
        X.append(simulate_spectra(h, rng, background))
        y_true.append(h["GLC"].to_numpy())
        y_ref.append(h["GLC"].to_numpy() * (1 + rng.normal(0, 0.03, len(h))))  # analyser error
        groups += [b] * len(h)
        times.append(h["time_d"].to_numpy())
    return {
        "X": np.vstack(X), "y_ref": np.concatenate(y_ref), "y_true": np.concatenate(y_true),
        "groups": np.array(groups), "time_d": np.concatenate(times),
    }


def select_components(X, y, groups, max_components: int = 15, preprocessing: str = "savgol_d1", folds: int = 5) -> tuple[int, list[float]]:
    """Choose the number of PLS components by grouped cross-validation.

    Folds are split *by batch* (GroupKFold): spectra from the same batch are
    strongly correlated, so letting one batch sit in both train and validation
    would leak information and make the model look better than it is.
    Uses the "one standard error" idea in simplified form: the smallest model
    whose RMSECV is within 5 % of the best.
    """
    cv = GroupKFold(n_splits=folds)
    scores = []
    for a in range(1, max_components + 1):
        pred = cross_val_predict(make_model(a, preprocessing), X, y, cv=cv, groups=groups)
        scores.append(rmse(y, pred.ravel()))
    best = min(scores)
    chosen = next(a for a, s in enumerate(scores, start=1) if s <= 1.05 * best)
    return chosen, scores


def evaluate(model: Pipeline, X, y) -> dict[str, float]:
    pred = model.predict(X).ravel()
    return {"rmse": rmse(y, pred), "r2": float(r2_score(y, pred))}


def regression_vector(model: Pipeline) -> np.ndarray:
    """PLS coefficients in preprocessed-spectrum space (one weight per wavenumber)."""
    return np.asarray(model.named_steps["pls"].coef_).ravel()
