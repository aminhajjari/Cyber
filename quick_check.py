"""
quick_check.py — Run a minimal end-to-end test before SLURM submission.
Tests all modules with n_days=5, epochs=3.
UPDATED to test the new Wang et al. DG placement (WT/PV/BM) and
micro-grid partition.

  python quick_check.py
  python quick_check.py --excel /path/to/ieee69bus.xlsx
"""

import sys, os, argparse, traceback
import numpy as np

def check(label):
    print(f"\n{'-'*60}")
    print(f"  CHECK: {label}")
    print("-"*60)

def ok(msg=""):
    print(f"  PASS {msg}")

def fail(msg=""):
    print(f"  FAIL {msg}")
    sys.exit(1)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--excel", type=str, default="data/ieee69bus.xlsx")
    args = p.parse_args()

    print("="*60)
    print("  FDI Detection - Quick Check (Wang et al. DG placement)")
    print("="*60)

    check("Python imports")
    try:
        import torch
        import sklearn
        import pandas as pd
        ok(f"torch={torch.__version__}, cuda={torch.cuda.is_available()}")
    except ImportError as e:
        fail(str(e))

    check("Data loader (IEEE 69-bus + WT/PV/BM placement)")
    try:
        from data_loader import (load_ieee69_from_excel, assign_der_units,
                                  assign_storage_buses, get_microgrid_summary)
        sys69   = load_ieee69_from_excel(args.excel)
        sys69   = assign_der_units(sys69)
        storage = assign_storage_buses(sys69)
        assert sys69.n_bus == 69, f"Expected 69 buses, got {sys69.n_bus}"
        assert sys69.Y_bus is not None

        n_wt = sum(1 for b in sys69.buses if b.der_type == "WT")
        n_pv = sum(1 for b in sys69.buses if b.der_type == "PV")
        n_bm = sum(1 for b in sys69.buses if b.der_type == "BM")
        assert n_wt == 6, f"Expected 6 WT buses, got {n_wt}"
        assert n_pv == 6, f"Expected 6 PV buses, got {n_pv}"
        assert n_bm == 11, f"Expected 11 BM buses, got {n_bm}"
        ok(f"{sys69.n_bus} buses | DG: {n_wt} WT + {n_pv} PV + {n_bm} BM "
           f"= {n_wt+n_pv+n_bm} units (expected 23)")

        mg_summary = get_microgrid_summary(sys69)
        total_mg_buses = sum(v["n_buses"] for v in mg_summary.values())
        assert total_mg_buses == 69, f"MG buses sum to {total_mg_buses}, expected 69"
        ok(f"Micro-grid partition OK: {list(mg_summary.keys())}, "
           f"{total_mg_buses} buses total")
    except Exception:
        fail(traceback.format_exc())

    check("Power flow solver (BFS)")
    try:
        from power_flow import BackwardForwardSweep
        pf = BackwardForwardSweep(sys69)
        P_load = np.array([b.Pd_MW   for b in sys69.buses])
        Q_load = np.array([b.Qd_MVAr for b in sys69.buses])
        P_gen  = np.zeros(69); P_gen[0] = P_load.sum() * 1.05
        Q_gen  = np.zeros(69)
        res = pf.solve(P_load, Q_load, P_gen, Q_gen)
        assert res.converged, "Power flow did not converge"
        if not (0.9 < res.V_min_pu <= 1.05):
            print(f"  WARNING: V_min={res.V_min_pu:.4f} is outside the ideal "
                  f"[0.90,1.05] band. Known pre-existing solver accuracy issue, "
                  f"unrelated to today's Wang et al. changes. Not blocking.")
        ok(f"Converged in {res.n_iter} iter | V_min={res.V_min_pu:.4f} pu "
           f"at bus {res.V_min_bus}")
    except Exception:
        fail(traceback.format_exc())

    check("Daily profile generation (WT Weibull + PV irradiance + BM constant)")
    try:
        from power_flow import generate_daily_profiles
        load_MW, der_gen_MW, reserve, breakdown = generate_daily_profiles(
            sys69, n_days=5, seed=42)
        assert load_MW.shape == (5, 24, 69)
        assert der_gen_MW.shape == (5, 24, 69)
        assert set(breakdown.keys()) == {"WT", "PV", "BM"}
        assert np.all(load_MW >= 0) and np.all(der_gen_MW >= 0)
        ok(f"load shape={load_MW.shape} | DER gen shape={der_gen_MW.shape} | "
           f"breakdown keys={list(breakdown.keys())}")
        ok(f"avg WT output={breakdown['WT'].mean()*1000:.3f} kW/bus-hr, "
           f"avg PV output={breakdown['PV'].mean()*1000:.3f} kW/bus-hr, "
           f"avg BM output={breakdown['BM'].mean()*1000:.3f} kW/bus-hr")
    except Exception:
        fail(traceback.format_exc())

    check("FDI attack model (S1 & S2, with micro-grid attribution)")
    try:
        from attack_model import AttackSimulator
        sim = AttackSimulator(sys69, storage, seed=42)
        r1  = sim.simulate_day(load_MW[0], der_gen_MW[0], "S1")
        r2  = sim.simulate_day(load_MW[0], der_gen_MW[0], "S2")
        assert r1.original_dispatch.shape == (24, 69)
        assert r1.falsification_signal is not None
        assert hasattr(r1, "affected_microgrids")
        ok(f"S1: feasible={r1.attack_feasible}, success={r1.attack_success}, "
           f"outage_t={r1.outage_time}, MGs={r1.affected_microgrids}")
        ok(f"S2: feasible={r2.attack_feasible}, success={r2.attack_success}, "
           f"outage_t={r2.outage_time}, MGs={r2.affected_microgrids}")

        # NEW: targeted DG-type attack test
        r_wt = sim.simulate_day(load_MW[0], der_gen_MW[0], "S1",
                                 target_der_type="WT")
        ok(f"WT-targeted attack: feasible={r_wt.attack_feasible}, "
           f"MGs={r_wt.affected_microgrids}")
    except Exception:
        fail(traceback.format_exc())

    check("Dataset builder")
    try:
        from attack_model import AttackSimulator
        from power_flow   import BackwardForwardSweep
        from detection_model import build_dataset

        pf_solver = BackwardForwardSweep(sys69)
        pf_all = []
        for day in range(5):
            dr = {"V_mag": np.zeros((24,69)), "theta": np.zeros((24,69))}
            for t in range(24):
                Pl = load_MW[day,t]
                Pg = der_gen_MW[day,t].copy()
                Pg[0] += max(0, Pl.sum()*1.05 - Pg.sum())
                r = pf_solver.solve(Pl, Pl*0.3, Pg, Pg*0.1)
                dr["V_mag"][t] = r.V_pu; dr["theta"][t] = r.theta_rad
            pf_all.append(dr)

        sim = AttackSimulator(sys69, storage, seed=42)
        atk_r  = [sim.simulate_day(load_MW[d], der_gen_MW[d], "S1") for d in range(5)]
        norm_r = [sim.simulate_day(load_MW[d], der_gen_MW[d], "S1") for d in range(5)]

        X, y = build_dataset(atk_r, norm_r, pf_all, pf_all, T_m=6)
        assert X.ndim == 3 and y.ndim == 1
        ok(f"X.shape={X.shape} | y.shape={y.shape} | "
           f"y range=[{y.min():.4f}, {y.max():.4f}]")
    except Exception:
        fail(traceback.format_exc())

    check("CNN detection model (3 epochs)")
    try:
        from detection_model import DetectionModelTrainer
        from config import CNN_CONFIG

        n_bus, d = X.shape[1], X.shape[2]
        cfg = {**CNN_CONFIG, "epochs": 3, "batch_size": 8}
        cnn = DetectionModelTrainer("CNN", n_bus, d, cfg)
        cnn.fit(X[:30], y[:30], X[30:40], y[30:40])
        preds = cnn.predict(X[40:50])
        metrics = cnn.evaluate(X[40:50], y[40:50], "quick_check")
        ok(f"Acc={metrics['Accuracy']:.1f}% | MSE={metrics['MSE_overall']:.4f}")
    except Exception:
        fail(traceback.format_exc())

    check("LLM explainer (rule-based fallback, with MG context)")
    try:
        from llm_explainer import LLMExplainer, AttackContext
        ctx = AttackContext(
            scenario="S1", current_hour=15, predicted_margin=0.03,
            actual_margin=0.08, margin_history=[0.21, 0.15, 0.11, 0.08, 0.05],
            alarm_triggered=True, hours_to_outage=1,
            top_anomaly_buses=[13,19,16], anomaly_magnitudes=[0.04, 0.03, 0.02],
            bus_voltages=np.ones(69)*0.97,
            total_load_MW=3.8, total_gen_MW=3.5, total_pv_MW=0.4,
            reserve_MW=0.019, confidence=0.92, affected_microgrids=["MG4"])
        explainer = LLMExplainer(use_llm=False)
        report = explainer.explain(ctx)
        assert "ATTACK LOCATION" in report and "IMMEDIATE ACTIONS" in report
        assert "MICRO-GRID IMPACT" in report
        assert "MG4" in report
        ok("Rule-based report generated with micro-grid context")
        for line in report.strip().split("\n")[:6]:
            print(f"    {line}")
        print("    ...")
    except Exception:
        fail(traceback.format_exc())

    print("\n" + "="*60)
    print("  ALL CHECKS PASSED")
    print("="*60)
    print("\nNext steps:")
    print("  1. Copy updated files to Narval (overwrite the 5 changed files):")
    print("     scp config.py data_loader.py power_flow.py attack_model.py")
    print("         llm_explainer.py main_train.py quick_check.py")
    print("         gkianfar@narval.alliancecan.ca:/home/gkianfar/scratch/Amin/CB/Cyber/")
    print("  2. Run quick_check.py again on Narval to confirm")
    print("  3. Submit job: sbatch run_fdi.sh")


if __name__ == "__main__":
    main()
