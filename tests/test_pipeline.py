from biotwin.optimize import best_setting, scan
from biotwin.pipeline import run


def test_quick_pipeline_runs_end_to_end(tmp_path):
    metrics = run(str(tmp_path), seed=0, quick=True)
    figures = {p.name for p in (tmp_path / "figures").iterdir()}
    assert {"fleet_profiles.png", "mspc_monitoring.png", "fault_contributions.png", "raman_spectra.png",
            "softsensor_tracking.png", "process_optimisation.png"} <= figures
    assert (tmp_path / "metrics.json").exists()
    assert metrics["soft_sensor"]["test_rmse_vs_true_mM"] < 2.0


def test_optimisation_finds_interior_trade_off():
    results = scan(shift_days=[None, 3, 5, 7, 9], feed_fracs=[0.02, 0.04, 0.06])
    best = best_setting(results, min_viability=70.0)
    assert best["temp_shift_day"] != "none"   # shifting temperature always helps here
    assert 0.02 < best["feed_frac"] < 0.06    # too little feed starves, too much is osmotic stress
    assert best["harvest_viability"] >= 70.0
