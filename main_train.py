"""
Main Training & Evaluation Pipeline
Runs the complete FDI detection experiment on IEEE 69-bus.

Usage:
  python main_train.py --excel /path/to/ieee69bus.xlsx --scenario S1 --use_llm
  python main_train.py --scenario both --n_days 356
"""

import argparse, os, json
import numpy as np
from datetime import datetime

from config import (BUS_EXCEL_PATH, RESULTS_DIR, MODEL_DIR, N_DAYS,
                    RANDOM_SEED, CNN_CONFIG, T_MONITORING, SIGMA_LEVELS,
                    SECURITY_THRESHOLD_MW)
from data_loader     import load_ieee69_from_excel, assign_pv_buses, assign_storage_buses
from power_flow      import BackwardForwardSweep, generate_daily_profiles
from attack_model    import AttackSimulator
from detection_model import DetectionModelTrainer, SVRDetector, build_dataset
from llm_explainer   import LLMExplainer, build_attack_context


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--excel",      type=str, default=BUS_EXCEL_PATH)
    p.add_argument("--scenario",   type=str, default="both", choices=["S1","S2","both"])
    p.add_argument("--n_days",     type=int, default=N_DAYS)
    p.add_argument("--epochs",     type=int, default=CNN_CONFIG["epochs"])
    p.add_argument("--use_llm",    action="store_true")
    p.add_argument("--skip_train", action="store_true")
    p.add_argument("--sensitivity",action="store_true")
    p.add_argument("--seed",       type=int, default=RANDOM_SEED)
    return p.parse_args()


def run_power_flow_all(system, load_MW, pv_MW):
    """Power flow for all days/hours. Returns list of {V_mag, theta} per day."""
    pf_solver = BackwardForwardSweep(system)
    n_days, T, n = load_MW.shape
    results = []
    print("[PF] Running power flow...")
    for day in range(n_days):
        day_r = {"V_mag": np.zeros((T, n)), "theta": np.zeros((T, n))}
        for t in range(T):
            P_l = load_MW[day,t]; Q_l = P_l * 0.3
            P_g = pv_MW[day,t].copy()
            P_g[0] += max(0, P_l.sum()*1.05 - P_g.sum())
            Q_g = P_g * 0.1
            pf = pf_solver.solve(P_l, Q_l, P_g, Q_g)
            day_r["V_mag"][t] = pf.V_pu
            day_r["theta"][t] = pf.theta_rad
        results.append(day_r)
        if (day+1) % 50 == 0:
            print(f"  PF: {day+1}/{n_days} days")
    return results


def simulate_attacks(system, storage, load_MW, pv_MW, scenario, seed):
    simulator = AttackSimulator(system, storage, seed=seed)
    n = load_MW.shape[0]
    atk_r, norm_r = [], []
    for day in range(n):
        a = simulator.simulate_day(load_MW[day], pv_MW[day], scenario)
        a.day = day; atk_r.append(a)
        b = simulator.simulate_day(load_MW[day], pv_MW[day], scenario)
        # Make "normal": zero out falsification so margin is healthy
        b.falsification_signal = np.zeros_like(a.falsification_signal
            if a.falsification_signal is not None else a.original_dispatch)
        b.day = day; norm_r.append(b)
    feas = sum(r.attack_feasible for r in atk_r)/n
    succ = sum(r.attack_success  for r in atk_r)/n
    print(f"[Attack {scenario}] Feasibility={feas:.2%} | Success={succ:.2%}")
    return atk_r, norm_r, feas, succ


def print_table(metrics, scenario):
    print(f"\n{'─'*72}")
    print(f"  Detection Performance — Scenario {scenario}  (cf. Table II paper)")
    print(f"{'─'*72}")
    print(f"  {'Model':<8} {'Acc%':>8} {'Prec%':>8} {'TPR%':>7} {'FPR%':>7} {'MSE':>9}")
    print(f"{'─'*72}")
    for k, m in metrics.items():
        print(f"  {k:<8} {m['Accuracy']:>7.2f} {m['Precision']:>8.2f} "
              f"{m['TPR']:>7.2f} {m['FPR']:>7.2f} {m['MSE_overall']:>9.4f}")
    print(f"{'─'*72}")


def main():
    args = parse_args()
    results_dir = os.path.join(RESULTS_DIR, datetime.now().strftime("%Y%m%d_%H%M%S"))
    os.makedirs(results_dir, exist_ok=True); os.makedirs(MODEL_DIR, exist_ok=True)

    print("\n" + "="*70)
    print("  FDI Attack Detection on IEEE 69-Bus + LLM Explainability")
    print("  Based on: Wu et al., IEEE Trans. Smart Grid, Vol 16, Nov 2025")
    print("="*70)

    # ── System setup ───────────────────────────────────────────────────────
    system  = load_ieee69_from_excel(args.excel)
    system  = assign_pv_buses(system)
    storage = assign_storage_buses(system)

    # ── Profiles ───────────────────────────────────────────────────────────
    load_MW, der_gen_MW, reserve, der_breakdown = generate_daily_profiles(system, n_days=args.n_days, seed=args.seed)

    # ── Power flow ─────────────────────────────────────────────────────────
    pf_all = run_power_flow_all(system, load_MW, pv_MW)

    # ── LLM ────────────────────────────────────────────────────────────────
    explainer = LLMExplainer(use_llm=args.use_llm)

    all_results = {}
    scenarios = ["S1","S2"] if args.scenario == "both" else [args.scenario]

    for scen in scenarios:
        print(f"\n{'#'*70}\n  SCENARIO: {scen}\n{'#'*70}")

        # Attack simulation
        atk_r, norm_r, feas, succ = simulate_attacks(
            system, storage, load_MW, pv_MW, scen, args.seed)

        # Dataset
        X, y = build_dataset(atk_r, norm_r, pf_all, pf_all,
                               T_m=T_MONITORING, feature_set="full")
        rng  = np.random.default_rng(args.seed)
        idx  = rng.permutation(len(X))
        X, y = X[idx], y[idx]
        n    = len(X)
        n1, n2 = int(.70*n), int(.85*n)
        X_tr,y_tr = X[:n1], y[:n1]
        X_v, y_v  = X[n1:n2], y[n1:n2]
        X_te,y_te = X[n2:], y[n2:]

        n_bus, d = X_tr.shape[1], X_tr.shape[2]
        cfg = {**CNN_CONFIG, "epochs": args.epochs}
        metrics = {}

        # CNN
        cnn_path = os.path.join(MODEL_DIR, f"cnn_{scen}.pkl")
        if args.skip_train and os.path.exists(cnn_path):
            cnn = DetectionModelTrainer.load(cnn_path)
        else:
            cnn = DetectionModelTrainer("CNN", n_bus, d, cfg)
            cnn.fit(X_tr, y_tr, X_v, y_v)
            cnn.save(cnn_path)
        metrics["CNN"] = cnn.evaluate(X_te, y_te, f"CNN/{scen}")

        # MLP
        mlp_path = os.path.join(MODEL_DIR, f"mlp_{scen}.pkl")
        if args.skip_train and os.path.exists(mlp_path):
            mlp = DetectionModelTrainer.load(mlp_path)
        else:
            mlp = DetectionModelTrainer("MLP", n_bus, d, cfg)
            mlp.fit(X_tr, y_tr, X_v, y_v)
            mlp.save(mlp_path)
        metrics["MLP"] = mlp.evaluate(X_te, y_te, f"MLP/{scen}")

        # SVR
        svr = SVRDetector()
        svr.fit(X_tr, y_tr)
        metrics["SVR"] = svr.evaluate(X_te, y_te, f"SVR/{scen}")

        print_table(metrics, scen)

        # LLM explanations for 3 interesting attack cases
        interesting = [r for r in atk_r if r.attack_success][:3]
        reports = []
        for res in interesting:
            alert_h = max(0, (res.outage_time or 16) - 2)
            class _PF:
                V_pu = pf_all[res.day]["V_mag"][alert_h]
            ctx = build_attack_context(
                res, _PF(), alert_h,
                load_MW[res.day, alert_h], pv_MW[res.day, alert_h],
                confidence=0.92)
            print(f"\n[LLM] Day={res.day}, Hour={alert_h}, Scenario={scen}")
            report = explainer.explain(ctx)
            print(report)
            reports.append({"day": res.day, "hour": alert_h, "report": report})

        # Sensitivity: incomplete information
        if args.sensitivity:
            sens = {}
            for sigma in SIGMA_LEVELS:
                sim2 = AttackSimulator(system, storage, seed=args.seed)
                a2 = [sim2.simulate_day(load_MW[d], pv_MW[d], scen, noise_std=sigma)
                      for d in range(min(50, args.n_days))]
                m2 = len(a2)
                sens[sigma] = {
                    "feasibility": sum(r.attack_feasible for r in a2)/m2,
                    "success":     sum(r.attack_success  for r in a2)/m2,
                }
                print(f"[Sensitivity {scen}] σ={sigma:.1f} → "
                      f"feas={sens[sigma]['feasibility']:.2%}, "
                      f"succ={sens[sigma]['success']:.2%}")
        else:
            sens = {}

        # Save
        out = {"metrics": metrics, "feasibility": feas, "success_ratio": succ,
               "sensitivity": {str(k): v for k,v in sens.items()},
               "llm_reports": reports}
        with open(os.path.join(results_dir, f"results_{scen}.json"), "w") as f:
            json.dump(out, f, indent=2, default=str)

        all_results[scen] = out

    # Final summary
    print("\n" + "="*70 + "\n  EXPERIMENT COMPLETE\n" + "="*70)
    for scen, res in all_results.items():
        print(f"\n  {scen}: Feasibility={res['feasibility']:.2%} | "
              f"Success={res['success_ratio']:.2%}")
        for m_name, m in res["metrics"].items():
            print(f"    {m_name}: Acc={m['Accuracy']:.2f}% MSE={m['MSE_overall']:.4f}")

    print(f"\n  Results saved to: {results_dir}")


if __name__ == "__main__":
    main()
