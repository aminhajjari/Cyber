"""
quick_check.py — Run a minimal end-to-end test before SLURM submission.
Tests all modules with n_days=5, epochs=3.
Run this interactively on a login node or debug allocation:

  python quick_check.py
  python quick_check.py --excel /path/to/ieee69bus.xlsx
"""

import sys, os, argparse, traceback
import numpy as np

def check(label):
    print(f"\n{'─'*60}")
    print(f"  CHECK: {label}")
    print("─"*60)

def ok(msg=""):
    print(f"  ✓ PASS {msg}")

def fail(msg=""):
    print(f"  ✗ FAIL {msg}")
    sys.exit(1)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--excel", type=str, default="data/ieee69bus.xlsx")
    args = p.parse_args()

    print("="*60)
    print("  FDI Detection — Quick Check")
    print("="*60)

    # ── 1. Imports ──────────────────────────────────────────────────────
    check("Python imports")
    try:
        import torch
        import sklearn
        import pandas as pd
        ok(f"torch={torch.__version__}, cuda={torch.cuda.is_available()}")
    except ImportError as e:
        fail(str(e))

    # ── 2. Data loader ──────────────────────────────────────────────────
    check("Data loader (IEEE 69-bus)")
    try:
        from data_loader import load_ieee69_from_excel, assign_pv_buses, assign_storage_buses
        sys69   = load_ieee69_from_excel(args.excel)
        sys69   = assign_pv_buses(sys69)
        storage = assign_storage_buses(sys69)
        assert sys69.n_bus == 69, f"Expected 69 buses, got {sys69.n_bus}"
        assert sys69.Y_bus is not None
        ok(f"{sys69.n_bus} buses, {sys69.n_branch} branches, "
           f"load={sys69.total_load_MW*1000:.1f} kW")
    except Exception as e:
        fail(traceback.format_exc())

    # ── 3. Power flow ───────────────────────────────────────────────────
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
        assert 0.9 < res.V_min_pu <= 1.05, f"Unexpected V_min={res.V_min_pu:.4f}"
        ok(f"Converged in {res.n_iter} iter | V_min={res.V_min_pu:.4f} pu "
           f"at bus {res.V_min_bus}")
    except Exception as e:
        fail(traceback.format_exc())

    # ── 4. Daily profiles ───────────────────────────────────────────────
    check("Daily profile generation")
    try:
        from power_flow import generate_daily_profiles
        load_MW, pv_MW, reserve = generate_daily_profiles(sys69, n_days=5, seed=42)
        assert load_MW.shape == (5, 24, 69)
        assert pv_MW.shape  == (5, 24, 69)
        assert np.all(load_MW >= 0) and np.all(pv_MW >= 0)
        ok(f"load shape={load_MW.shape} | "
           f"avg load={load_MW.mean()*1000:.2f} kW/bus")
    except Exception as e:
        fail(traceback.format_exc())

    # ── 5. Attack model ─────────────────────────────────────────────────
    check("FDI attack model (S1 & S2)")
    try:
        from attack_model import AttackSimulator
        sim = AttackSimulator(sys69, storage, seed=42)
        r1  = sim.simulate_day(load_MW[0], pv_MW[0], "S1")
        r2  = sim.simulate_day(load_MW[0], pv_MW[0], "S2")
        assert r1.original_dispatch.shape == (24, 69)
        assert r1.falsification_signal is not None
        ok(f"S1: feasible={r1.attack_feasible}, success={r1.attack_success}, "
           f"outage_t={r1.outage_time}")
        ok(f"S2: feasible={r2.attack_feasible}, success={r2.attack_success}, "
           f"outage_t={r2.outage_time}")
    except Exception as e:
        fail(traceback.format_exc())

    # ── 6. Dataset builder ──────────────────────────────────────────────
    check("Dataset builder")
    try:
        from attack_model import AttackSimulator
        from power_flow   import BackwardForwardSweep
        from detection_model import build_dataset

        # Build mini pf_results (5 days)
        pf_solver = BackwardForwardSweep(sys69)
        pf_all = []
        for day in range(5):
            dr = {"V_mag": np.zeros((24,69)), "theta": np.zeros((24,69))}
            for t in range(24):
                Pl = load_MW[day,t]; Pg = np.zeros(69); Pg[0] = Pl.sum()*1.05
                r = pf_solver.solve(Pl, Pl*0.3, Pg, Pg*0.1)
                dr["V_mag"][t] = r.V_pu; dr["theta"][t] = r.theta_rad
            pf_all.append(dr)

        sim = AttackSimulator(sys69, storage, seed=42)
        atk_r  = [sim.simulate_day(load_MW[d], pv_MW[d], "S1") for d in range(5)]
        norm_r = [sim.simulate_day(load_MW[d], pv_MW[d], "S1") for d in range(5)]

        X, y = build_dataset(atk_r, norm_r, pf_all, pf_all, T_m=6)
        assert X.ndim == 3 and y.ndim == 1
        ok(f"X.shape={X.shape} | y.shape={y.shape} | "
           f"y range=[{y.min():.3f}, {y.max():.3f}]")
    except Exception as e:
        fail(traceback.format_exc())

    # ── 7. CNN model ────────────────────────────────────────────────────
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
    except Exception as e:
        fail(traceback.format_exc())

    # ── 8. LLM explainer (rule-based, no GPU) ──────────────────────────
    check("LLM explainer (rule-based fallback)")
    try:
        from llm_explainer import LLMExplainer, AttackContext
        ctx = AttackContext(
            scenario="S1", current_hour=15, predicted_margin=0.25,
            actual_margin=0.8, margin_history=[2.0, 1.5, 1.0, 0.6, 0.3],
            alarm_triggered=True, hours_to_outage=1,
            top_anomaly_buses=[11,21,33], anomaly_magnitudes=[0.4, 0.3, 0.2],
            bus_voltages=np.ones(69)*0.97,
            total_load_MW=3.8, total_gen_MW=3.5, total_pv_MW=0.4,
            reserve_MW=0.19, confidence=0.92)
        explainer = LLMExplainer(use_llm=False)
        report = explainer.explain(ctx)
        assert "ATTACK LOCATION" in report and "IMMEDIATE ACTIONS" in report
        ok("Rule-based report generated")
        # Print first 5 lines
        for line in report.strip().split("\n")[:5]:
            print(f"    {line}")
        print("    ...")
    except Exception as e:
        fail(traceback.format_exc())

    # ── Summary ─────────────────────────────────────────────────────────
    print("\n" + "="*60)
    print("  ALL CHECKS PASSED ✓")
    print("="*60)
    print("\nNext steps:")
    print("  1. Copy project to Narval:")
    print("     scp -r fdi_detection/ narval.alliancecan.ca:"
          "/home/gkianfar/scratch/Amin/")
    print("  2. Upload your Excel file to data/ieee69bus.xlsx")
    print("  3. Download Mistral-7B (see run_fdi.sh comments)")
    print("  4. Submit job: sbatch run_fdi.sh")


if __name__ == "__main__":
    main()
