"""Multivariate Statistical Process Control (MSPC) for batch processes.

Watching 10 variables on 10 separate charts misses problems that only show up
as a *combination* (e.g. VCD slightly low AND lactate slightly high). MSPC
builds one PCA model of what normal batches look like and asks two questions
about every new batch:

* Hotelling's T^2 - is the batch unusual *within* the normal pattern?
  (e.g. everything a bit higher than usual, but in the usual proportions)
* SPE / Q       - does the batch break the normal pattern altogether?
  (the correlations between variables have changed - a new kind of problem)

Batch data are 3-D (batches x variables x time). Following Nomikos &
MacGregor (1995), each batch is "unfolded" into one long row:
[vcd_day0, viability_day0, ..., vcd_day1, ...].

To monitor a batch *while it is running*, one model is built per day using
only the data available up to that day (an "evolving" model set).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import stats

from .batches import VARIABLES


def unfold(daily: pd.DataFrame, max_day: int, variables: list[str] = VARIABLES) -> tuple[np.ndarray, list[str], list[str]]:
    """Batch-wise unfolding of days 0..max_day. Returns (X, batch_ids, column_names)."""
    d = daily[daily["day"] <= max_day].sort_values(["batch_id", "day"])
    wide = d.pivot(index="batch_id", columns="day", values=variables)  # columns: (variable, day)
    wide = wide.swaplevel(axis=1).reindex(columns=pd.MultiIndex.from_product([range(max_day + 1), variables]))
    names = [f"{var}_d{day}" for day, var in wide.columns]
    return wide.to_numpy(dtype=float), list(wide.index), names


def _scaled_chi2_limit(values: np.ndarray, alpha: float) -> float:
    """Control limit g * chi2(h) matched to the mean and variance of `values`.

    This is Box's approximation, the standard way to set SPE limits
    (Nomikos & MacGregor 1995); it works for any positive quadratic statistic.
    """
    m, v = values.mean(), values.var(ddof=1)
    g, h = v / (2 * m), 2 * m**2 / v
    return float(g * stats.chi2.ppf(alpha, h))


@dataclass
class _PCAFit:
    mean: np.ndarray
    std: np.ndarray
    loadings: np.ndarray   # (n_columns, A)
    eigenvalues: np.ndarray  # variance of each score

    def project(self, X: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        Z = (X - self.mean) / self.std
        T = Z @ self.loadings
        E = Z - T @ self.loadings.T
        return Z, T, E


def _fit_pca(X: np.ndarray, n_components: int) -> _PCAFit:
    mean = X.mean(axis=0)
    std = X.std(axis=0, ddof=1)
    std[std < 1e-9] = 1.0
    Z = (X - mean) / std
    _, S, Vt = np.linalg.svd(Z, full_matrices=False)
    eig = S**2 / (len(X) - 1)
    return _PCAFit(mean, std, Vt[:n_components].T, eig[:n_components])


class BatchPCA:
    """PCA monitoring model with T^2 and SPE control limits.

    limits="loo": limits come from leave-one-out statistics - each training
        batch is scored by a model that never saw it, the way a future batch
        will be. "train": limits from the training batches' own statistics,
        which is optimistic because PCA has partly fitted their noise.
    """

    def __init__(self, var_explained: float = 0.8, alpha: float = 0.99, limits: str = "loo"):
        self.var_explained = var_explained
        self.alpha = alpha
        self.limits = limits

    def fit(self, X: np.ndarray) -> "BatchPCA":
        n = len(X)
        full = _fit_pca(X, min(n - 1, X.shape[1]))
        ratio = np.cumsum(full.eigenvalues) / full.eigenvalues.sum()
        self.n_components = int(min(np.searchsorted(ratio, self.var_explained) + 1, n - 2))
        self.model = _fit_pca(X, self.n_components)

        if self.limits == "loo":
            t2, spe = [], []
            for i in range(n):
                sub = _fit_pca(np.delete(X, i, axis=0), self.n_components)
                t, q = self._stats(sub, X[i : i + 1])
                t2.append(t[0])
                spe.append(q[0])
            t2, spe = np.array(t2), np.array(spe)
        else:
            t2, spe = self._stats(self.model, X)
        self.t2_limit = _scaled_chi2_limit(t2, self.alpha)
        self.spe_limit = _scaled_chi2_limit(spe, self.alpha)
        return self

    @staticmethod
    def _stats(fit: _PCAFit, X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        _, T, E = fit.project(X)
        return (T**2 / fit.eigenvalues).sum(axis=1), (E**2).sum(axis=1)

    def score(self, X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """(T^2, SPE) for each row of X."""
        return self._stats(self.model, X)

    def contributions(self, X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Per-column contributions to SPE and to T^2 for each row of X.

        SPE contribution = squared residual of that column.
        T^2 contribution = z_k * sum_a(t_a * p_ka / lambda_a); the columns
        that push the batch furthest out along the model directions score highest.
        """
        Z, T, E = self.model.project(X)
        spe_c = E**2
        t2_c = Z * ((T / self.model.eigenvalues) @ self.model.loadings.T)
        return spe_c, t2_c


class EvolvingMonitor:
    """One BatchPCA per day, so a running batch can be checked every morning."""

    def __init__(self, days: range = range(1, 15), **pca_kwargs):
        self.days = days
        self.pca_kwargs = pca_kwargs
        self.models: dict[int, BatchPCA] = {}

    def fit(self, daily_normal: pd.DataFrame) -> "EvolvingMonitor":
        for day in self.days:
            X, _, _ = unfold(daily_normal, day)
            self.models[day] = BatchPCA(**self.pca_kwargs).fit(X)
        return self

    def monitor(self, daily: pd.DataFrame) -> pd.DataFrame:
        """T^2/SPE and limits for every batch on every day (long format)."""
        rows = []
        for day, model in self.models.items():
            X, ids, _ = unfold(daily, day)
            t2, spe = model.score(X)
            for b, t, q in zip(ids, t2, spe):
                rows.append({
                    "batch_id": b, "day": day,
                    "t2": t, "t2_limit": model.t2_limit,
                    "spe": q, "spe_limit": model.spe_limit,
                    "out": (t > model.t2_limit) or (q > model.spe_limit),
                })
        return pd.DataFrame(rows)

    @staticmethod
    def first_alarm(monitoring: pd.DataFrame, consecutive: int = 2) -> dict[str, int | None]:
        """Day of the first alarm per batch: `consecutive` out-of-limit days in a row.

        Requiring two days in a row is a simple run rule that cuts false
        alarms from single noisy samples.
        """
        alarms = {}
        for batch_id, g in monitoring.sort_values("day").groupby("batch_id"):
            run, alarm = 0, None
            for day, out in zip(g["day"], g["out"]):
                run = run + 1 if out else 0
                if run >= consecutive:
                    alarm = int(day)
                    break
            alarms[batch_id] = alarm
        return alarms

    def variable_contributions(self, daily: pd.DataFrame, day: int) -> pd.DataFrame:
        """SPE + positive T^2 contributions summed per variable, each row scaled to sum 1."""
        model = self.models[day]
        X, ids, names = unfold(daily, day)
        spe_c, t2_c = model.contributions(X)
        total = spe_c / spe_c.sum(axis=1, keepdims=True) + np.clip(t2_c, 0, None) / (
            np.clip(t2_c, 0, None).sum(axis=1, keepdims=True) + 1e-12
        )
        per_col = pd.DataFrame(total, index=ids, columns=names)
        per_var = per_col.T.groupby(lambda c: c.rsplit("_d", 1)[0]).sum().T[VARIABLES]
        return per_var.div(per_var.sum(axis=1), axis=0)
