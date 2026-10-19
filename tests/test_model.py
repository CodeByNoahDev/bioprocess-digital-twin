from dataclasses import replace

import numpy as np
import pytest

from biotwin.batches import FAULTS, VARIABLES, make_fleet, run_batch
from biotwin.model import Recipe, simulate_batch


@pytest.fixture(scope="module")
def nominal():
    return simulate_batch()


def test_states_stay_physical(nominal):
    h = nominal.hourly
    for col in ("Xv", "Xd", "GLC", "GLN", "LAC", "AMM", "P", "SALT"):
        assert (h[col] >= 0).all(), col
    assert h["viability"].between(0, 100).all()


def test_volume_balance(nominal):
    added = sum(f["feed_l"] + f["glucose_top_up_l"] for f in nominal.feeds)
    assert nominal.hourly["volume_l"].iloc[-1] == pytest.approx(Recipe().v0_l + added)


def test_product_mass_never_decreases(nominal):
    # Feeding dilutes the titer (mg/L) but cannot destroy antibody (mg).
    mass = nominal.hourly["P"] * nominal.hourly["volume_l"]
    assert (np.diff(mass) >= -1e-6).all()


def test_textbook_profile(nominal):
    s = nominal.samples
    assert 15 < s["Xv"].max() < 30            # peak VCD, 1e6 cells/mL
    assert 60 < s["viability"].iloc[-1] < 90  # harvest viability
    assert 2000 < s["P"].iloc[-1] < 7000      # titer, mg/L
    lac = s["LAC"].to_numpy()
    assert lac.argmax() < len(lac) - 1 and lac[-1] < lac.max()  # lactate shift


def test_temperature_shift_raises_titer():
    shifted = simulate_batch(Recipe(temp_shift_d=5)).hourly["P"].iloc[-1]
    unshifted = simulate_batch(Recipe(temp_shift_d=None)).hourly["P"].iloc[-1]
    assert shifted > 1.2 * unshifted


def test_more_feed_raises_osmolality():
    low = simulate_batch(replace(Recipe(), feed_frac=0.02)).hourly["osmolality"].iloc[-1]
    high = simulate_batch(replace(Recipe(), feed_frac=0.06)).hourly["osmolality"].iloc[-1]
    assert 280 < simulate_batch().hourly["osmolality"].iloc[0] < 320
    assert high > low + 20


@pytest.mark.parametrize("fault", list(FAULTS))
def test_each_fault_changes_the_batch(fault):
    normal, _ = run_batch(None, np.random.default_rng(11))
    faulty, onset = run_batch(fault, np.random.default_rng(11))  # same normal variation
    assert onset is not None
    diff = (faulty.samples[["Xv", "GLC", "GLN", "P"]] - normal.samples[["Xv", "GLC", "GLN", "P"]]).abs()
    probe_diff = (faulty.hourly[["temp", "do"]] - normal.hourly[["temp", "do"]]).abs().max().max()
    assert diff.to_numpy().max() > 0.5 or probe_diff > 1


def test_fleet_is_reproducible_and_complete():
    a = make_fleet(n_train=2, n_test_normal=1, n_per_fault=1, seed=5)
    b = make_fleet(n_train=2, n_test_normal=1, n_per_fault=1, seed=5)
    assert a.daily.equals(b.daily)
    assert len(a.meta) == 3 + len(FAULTS)
    assert set(VARIABLES) <= set(a.daily.columns)
    assert a.daily.groupby("batch_id")["day"].apply(list).map(lambda d: d == list(range(15))).all()
