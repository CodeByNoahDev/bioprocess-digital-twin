import numpy as np
import pandas as pd
import pytest

from biotwin import softsensor as ss
from biotwin.batches import VARIABLES, make_fleet
from biotwin.mspc import BatchPCA, EvolvingMonitor, unfold


LOADINGS = np.random.default_rng(42).normal(size=(3, 30))  # one fixed "process"


def _correlated(rng, n):
    k, p = LOADINGS.shape
    return rng.normal(size=(n, k)) @ LOADINGS + 0.3 * rng.normal(size=(n, p))


def test_unfold_layout():
    fleet = make_fleet(n_train=3, n_test_normal=0, n_per_fault=0, seed=1)
    X, ids, names = unfold(fleet.daily, max_day=4)
    assert X.shape == (3, 5 * len(VARIABLES))
    assert names[0] == f"{VARIABLES[0]}_d0" and names[len(VARIABLES)] == f"{VARIABLES[0]}_d1"
    first = fleet.daily[(fleet.daily["batch_id"] == ids[0]) & (fleet.daily["day"] == 1)]
    assert X[0, len(VARIABLES)] == pytest.approx(first[VARIABLES[0]].iloc[0])


def test_loo_limits_control_false_alarms_better_than_naive():
    # With only 40 training batches the limits themselves are uncertain, so
    # average over several training sets rather than trusting one.
    rates = {"loo": [], "train": []}
    for seed in range(4):
        rng = np.random.default_rng(seed)
        train, new = _correlated(rng, 40), _correlated(rng, 2000)
        for limits in rates:
            m = BatchPCA(limits=limits).fit(train)
            t2, spe = m.score(new)
            rates[limits].append(np.mean((t2 > m.t2_limit) | (spe > m.spe_limit)))
    assert np.mean(rates["loo"]) < 0.08  # nominal: about 2 % (two 99 % limits)
    assert all(l < t for l, t in zip(rates["loo"], rates["train"]))


def test_broken_correlation_shows_in_spe_and_contributions():
    rng = np.random.default_rng(1)
    train = _correlated(rng, 60)
    m = BatchPCA().fit(train)
    x = train[:1].copy()
    x[0, 7] += 6 * train[:, 7].std()  # one variable alone moves: correlation broken
    _, spe = m.score(x)
    assert spe[0] > m.spe_limit
    spe_c, _ = m.contributions(x)
    assert spe_c[0].argmax() == 7


def test_evolving_monitor_flags_fault_not_normal():
    fleet = make_fleet(n_train=25, n_test_normal=3, n_per_fault=1, seed=2)
    meta = fleet.meta
    train = fleet.daily[fleet.daily["batch_id"].isin(meta.loc[meta["split"] == "train", "batch_id"])]
    test = fleet.daily[fleet.daily["batch_id"].isin(meta.loc[meta["split"] == "test", "batch_id"])]
    mon = EvolvingMonitor().fit(train)
    alarms = mon.first_alarm(mon.monitor(test))
    faults = meta[meta["fault"].isin(["temp_excursion", "do_control_failure", "poor_inoculum"])]
    assert all(alarms[b] is not None for b in faults["batch_id"])
    normals = meta[(meta["split"] == "test") & (meta["fault"] == "normal")]["batch_id"]
    assert sum(alarms[b] is not None for b in normals) <= 1


def test_first_alarm_needs_consecutive_days():
    m = pd.DataFrame({"batch_id": "B", "day": [1, 2, 3, 4, 5], "out": [True, False, True, True, False]})
    assert EvolvingMonitor.first_alarm(m, consecutive=2) == {"B": 4}
    assert EvolvingMonitor.first_alarm(m, consecutive=3) == {"B": None}


def test_preprocessing_functions():
    X = np.random.default_rng(3).normal(size=(5, ss.WAVENUMBERS.size)) + 10
    Z = ss.snv(X)
    assert np.allclose(Z.mean(axis=1), 0) and np.allclose(Z.std(axis=1), 1)
    # A constant offset (flat baseline) disappears after the first derivative.
    assert np.allclose(ss.savgol_derivative(X), ss.savgol_derivative(X + 3.0))


def test_pure_spectrum_peaks_where_expected():
    assert ss.WAVENUMBERS[ss.PURE["glucose"].argmax()] == pytest.approx(1125, abs=2)
    assert ss.WAVENUMBERS[ss.PURE["lactate"].argmax()] == pytest.approx(856, abs=2)


def test_soft_sensor_predicts_glucose_on_new_batches():
    fleet = make_fleet(n_train=15, n_test_normal=4, n_per_fault=0, seed=4)
    rng = np.random.default_rng(0)
    train_ids = fleet.meta.loc[fleet.meta["split"] == "train", "batch_id"]
    test_ids = fleet.meta.loc[fleet.meta["split"] == "test", "batch_id"]
    cal = ss.build_dataset(fleet.hourly, train_ids, rng, spike_fraction=0.5)
    test = ss.build_dataset(fleet.hourly, test_ids, rng)
    a, _ = ss.select_components(cal["X"], cal["y_ref"], cal["groups"], max_components=8)
    model = ss.make_model(a).fit(cal["X"], cal["y_ref"])
    result = ss.evaluate(model, test["X"], test["y_true"])
    assert result["rmse"] < 2.0
    assert result["r2"] > 0.8
