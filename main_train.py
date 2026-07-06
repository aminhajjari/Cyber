"""
Main Training & Evaluation Pipeline
Runs the complete FDI detection experiment on IEEE 69-bus.

DG placement (WT/PV/BM, 23 units) and micro-grid partition (5 MGs) follow:
  Wang et al., Energy Reports 6 (2020) 1233-1249, Tables 1 & 3.

Usage:
  python main_train.py --excel /path/to/ieee69bus.xlsx --scenario S1 --use_llm
  python main_train.py --scenario both --n_days 356
"""

import argparse, os, json
import numpy as np
from datetime import datetime

from config import (BUS_EXCEL_PATH, RESULTS_DIR, MODEL_DIR, N_DAYS,
                    RANDOM_SEED, CNN_CONFIG, T_MONITORING, T_PRED_AHEAD, SIGMA_LEVELS,
                    SECURITY_THRESHOLD_MW, MICROGRID_MAP)
from data_loader     import (load_ieee69_from_excel, assign_der_units,
                              assign_storage_buses, get_microgrid_summary)
from power_flow      import BackwardForwardSweep, generate_daily_profiles
from attack_model    import AttackSimulator
from detection_model import DetectionModelTrainer, SVRDetector, build_dataset
from llm_explainer   import (LLMExplainer, build_attack_context,
                              parse_structured_report, grounding_score,
                              check_confidence_faithfulness)
from detection_model import build_input_tensor
from improvements    import (bus_saliency, localization_score, group_split_by_day,
                              deletion_insertion_score)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--excel",      type=str, default=BUS_EXCEL_PATH)
    p.add_argument("--scenario",   type=str, default="both", choices=["S1","S2","both"])
    p.add_argument("--n_days",     type=int, default=N_DAYS)
    p.add_argument("--epochs",     type=int, default=CNN_CONFIG["epochs"])
    p.add_argument("--use_llm",    action="store_true")
    p.add_argument("--skip_train", action="store_true")
    p.add_argument("--sensitivity",action="store_true")
    p.add_argument("--target_der", type=str, default=None,
                   choices=[None, "WT", "PV", "BM"],
                   help="Confine S1 attack to one DG technology (NEW)")
    p.add_argument("--feature_set", type=str, default="full",
                   choices=["PV", "PVtheta", "full", "network_only"],
                   help="'network_only' drops the predicted-vs-actual dispatch "
                        "pair and keeps only V_mag/theta -- use this to check "
                        "whether detection accuracy survives without the "
                        "near-trivial mismatch signal (NEW, ablation).")
    p.add_argument("--seed",       type=int, default=RANDOM_SEED)
    return p.parse_args()


def run_power_flow_all(system, load_MW, der_gen_MW):
    """Power flow for all days/hours. Returns list of {V_mag, theta} per day."""
    pf_solver = BackwardForwardSweep(system)
    n_days, T, n = load_MW.shape
    results = []
    print("[PF] Running power flow...")
    for day in range(n_days):
        day_r = {"V_mag": np.zeros((T, n)), "theta": np.zeros((T, n))}
        for t in range(T):
            P_l = load_MW[day,t]; Q_l = P_l * 0.3
            P_g = der_gen_MW[day,t].copy()
            P_g[0] += max(0, P_l.sum()*1.05 - P_g.sum())
            Q_g = P_g * 0.1
            pf = pf_solver.solve(P_l, Q_l, P_g, Q_g)
            day_r["V_mag"][t] = pf.V_pu
            day_r["theta"][t] = pf.theta_rad
        results.append(day_r)
        if (day+1) % 50 == 0:
            print(f"  PF: {day+1}/{n_days} days")
    return results


def simulate_attacks(system, storage, load_MW, der_gen_MW, scenario, seed,
                      target_der_type=None):
    simulator = AttackSimulator(system, storage, seed=seed)
    n = load_MW.shape[0]
    atk_r, norm_r = [], []
    for day in range(n):
        a = simulator.simulate_day(load_MW[day], der_gen_MW[day], scenario,
                                    target_der_type=target_der_type)
        a.day = day; atk_r.append(a)
        b = simulator.simulate_day(load_MW[day], der_gen_MW[day], scenario)
        b.falsification_signal = np.zeros_like(a.falsification_signal
            if a.falsification_signal is not None else a.original_dispatch)
        b.day = day; norm_r.append(b)
    feas = sum(r.attack_feasible for r in atk_r)/n
    succ = sum(r.attack_success  for r in atk_r)/n
    print(f"[Attack {scenario}] Feasibility={feas:.2%} | Success={succ:.2%}")
    return atk_r, norm_r, feas, succ


def print_table(metrics, scenario):
    print(f"\n{'-'*72}")
    print(f"  Detection Performance — Scenario {scenario}  (cf. Table II, Wu et al.)")
    print(f"{'-'*72}")
    print(f"  {'Model':<8} {'Acc%':>8} {'Prec%':>8} {'TPR%':>7} {'FPR%':>7} {'MSE':>9}")
    print(f"{'-'*72}")
    for k, m in metrics.items():
        print(f"  {k:<8} {m['Accuracy']:>7.2f} {m['Precision']:>8.2f} "
              f"{m['TPR']:>7.2f} {m['FPR']:>7.2f} {m['MSE_overall']:>9.4f}")
    print(f"{'-'*72}")


def print_microgrid_summary(system):
    """NEW: print the Wang et al. micro-grid breakdown at experiment start."""
    print(f"\n{'='*70}")
    print("  MICRO-GRID PARTITION (Wang et al., Table 3)")
    print(f"{'='*70}")
    summary = get_microgrid_summary(system)
    for mg, info in summary.items():
        print(f"  {mg}: {info['n_buses']:>2} buses | "
              f"load={info['total_load_MW']*1000:>7.1f} kW | "
              f"DER={info['total_der_MW']*1000:>6.1f} kW | "
              f"units={len(info['der_units'])}")


def main():
    args = parse_args()
    results_dir = os.path.join(RESULTS_DIR, datetime.now().strftime("%Y%m%d_%H%M%S"))
    os.makedirs(results_dir, exist_ok=True); os.makedirs(MODEL_DIR, exist_ok=True)

    print("\n" + "="*70)
    print("  FDI Attack Detection on IEEE 69-Bus + LLM Explainability")
    print("  Attack/Detection: Wu et al., IEEE Trans. Smart Grid, 2025")
    print("  DG placement & MG partition: Wang et al., Energy Reports, 2020")
    print("="*70)

    # ── System setup ───────────────────────────────────────────────────────
    system  = load_ieee69_from_excel(args.excel)
    system  = assign_der_units(system)          # CHANGED from assign_pv_buses
    storage = assign_storage_buses(system)

    print_microgrid_summary(system)             # NEW

    # ── Profiles ───────────────────────────────────────────────────────────
    # CHANGED: now returns 4 values (load, combined DER gen, reserve, breakdown)
    load_MW, der_gen_MW, reserve, der_breakdown = generate_daily_profiles(
        system, n_days=args.n_days, seed=args.seed)

    # ── Power flow ─────────────────────────────────────────────────────────
    pf_all = run_power_flow_all(system, load_MW, der_gen_MW)

    # ── LLM ────────────────────────────────────────────────────────────────
    explainer = LLMExplainer(use_llm=args.use_llm)

    all_results = {}
    scenarios = ["S1","S2"] if args.scenario == "both" else [args.scenario]

    for scen in scenarios:
        print(f"\n{'#'*70}\n  SCENARIO: {scen}"
              f"{f' (targeting {args.target_der} only)' if args.target_der else ''}"
              f"\n{'#'*70}")

        atk_r, norm_r, feas, succ = simulate_attacks(
            system, storage, load_MW, der_gen_MW, scen, args.seed,
            target_der_type=args.target_der)

        X, y, lbl, day = build_dataset(atk_r, norm_r, pf_all, pf_all,
                                       T_m=T_MONITORING, feature_set=args.feature_set)
        # leakage-free split: all windows of a Monte-Carlo day stay together
        itr, iva, ite = group_split_by_day(day, 0.70, 0.15, seed=args.seed)
        X_tr, y_tr, lbl_tr = X[itr], y[itr], lbl[itr]
        X_v,  y_v,  lbl_v  = X[iva], y[iva], lbl[iva]
        X_te, y_te, lbl_te = X[ite], y[ite], lbl[ite]

        n_bus, d = X_tr.shape[1], X_tr.shape[2]
        cfg = {**CNN_CONFIG, "epochs": args.epochs}
        metrics = {}

        cnn_path = os.path.join(MODEL_DIR, f"cnn_{scen}.pkl")
        if args.skip_train and os.path.exists(cnn_path):
            cnn = DetectionModelTrainer.load(cnn_path)
        else:
            cnn = DetectionModelTrainer("CNN", n_bus, d, cfg)
            cnn.fit(X_tr, y_tr, X_v, y_v, lbl_train=lbl_tr, lbl_val=lbl_v)
            cnn.save(cnn_path)
        metrics["CNN"] = cnn.evaluate(X_te, y_te, f"CNN/{scen}", lbl_true=lbl_te)

        mlp_path = os.path.join(MODEL_DIR, f"mlp_{scen}.pkl")
        if args.skip_train and os.path.exists(mlp_path):
            mlp = DetectionModelTrainer.load(mlp_path)
        else:
            mlp = DetectionModelTrainer("MLP", n_bus, d, cfg)
            mlp.fit(X_tr, y_tr, X_v, y_v, lbl_train=lbl_tr, lbl_val=lbl_v)
            mlp.save(mlp_path)
        metrics["MLP"] = mlp.evaluate(X_te, y_te, f"MLP/{scen}", lbl_true=lbl_te)

        svr = SVRDetector()
        svr.fit(X_tr, y_tr, lbl_train=lbl_tr)
        metrics["SVR"] = svr.evaluate(X_te, y_te, f"SVR/{scen}", lbl_true=lbl_te)

        print_table(metrics, scen)

        # ── LLM interpretability on the DETECTOR'S decision ──────────────────
        # Pick 3 cases the CLASSIFIER itself actually flagged (attack_prob
        # above 0.5), not just days where the attack succeeded physically.
        # These are NOT the same thing: a day can be a "successful" attack
        # (margin exhausted) while the classifier still assigns near-zero
        # probability at the specific hour we sample, which would have us
        # asking the LLM to narrate a confident incident report for an hour
        # the model itself saw nothing anomalous in -- exactly the mismatch
        # check_confidence_faithfulness() below is there to catch, but better
        # to select genuine detections in the first place so this demo
        # actually reflects the model's own decisions.
        tmin = T_MONITORING + T_PRED_AHEAD

        def _clf_hour_and_prob(res):
            """Return (best_hour, best_prob) among candidate alert hours for res,
            or (None, 0.0) if no valid monitoring window exists."""
            best_h, best_p = None, -1.0
            for h in range(tmin, len(res.system_margin_true)):
                zeros = np.zeros_like(res.original_dispatch)
                xs = build_input_tensor(
                    gen_dispatch_hat=res.original_dispatch,
                    gen_dispatch_meas=res.falsified_dispatch,
                    curtail_hat=zeros, curtail_meas=zeros,
                    stor_hat=zeros, stor_meas=zeros,
                    V_mag=pf_all[res.day]["V_mag"], theta=pf_all[res.day]["theta"],
                    t_pred=h, T_m=T_MONITORING, feature_set=args.feature_set)
                if xs is None:
                    continue
                p = float(cnn.predict_proba(xs[None, ...])[0])
                if p > best_p:
                    best_h, best_p = h, p
            return best_h, best_p

        candidates = []
        for i, r in enumerate(atk_r):
            h, p = _clf_hour_and_prob(r)
            if h is not None:
                candidates.append((i, h, p))
        candidates.sort(key=lambda t: t[2], reverse=True)
        detected = [(i, h, p) for i, h, p in candidates if p > 0.5][:3]
        if len(detected) < 3:
            print(f"[Interpretability/{scen}] WARNING: only {len(detected)}/3 "
                  f"sampled days had ANY hour where the classifier's own "
                  f"probability exceeded 0.5 -- falling back to the "
                  f"highest-probability hours available even though the "
                  f"detector itself was not confident there either. This is "
                  f"itself worth reporting, not hiding.")
            detected = candidates[:3]
        interesting = [(atk_r[i], h) for i, h, _ in detected]

        reports, loc_scores = [], []
        for res, alert_h in interesting:
            zeros = np.zeros_like(res.original_dispatch)
            x_sample = build_input_tensor(
                gen_dispatch_hat=res.original_dispatch,
                gen_dispatch_meas=res.falsified_dispatch,
                curtail_hat=zeros, curtail_meas=zeros,
                stor_hat=zeros, stor_meas=zeros,
                V_mag=pf_all[res.day]["V_mag"], theta=pf_all[res.day]["theta"],
                t_pred=alert_h, T_m=T_MONITORING, feature_set=args.feature_set)
            if x_sample is None:
                continue

            # detector's OWN outputs + saliency (uses the trained CNN)
            X1 = x_sample[None, ...]
            pred_margin = float(cnn.predict_margin(X1)[0])
            atk_prob    = float(cnn.predict_proba(X1)[0])
            sal   = bus_saliency(cnn.model, x_sample, cnn.scaler_X, head="cls")
            order = np.argsort(sal)[::-1]
            sal_buses = [int(b) + 1 for b in order[:5]]
            sal_mags  = [float(sal[b]) for b in order[:5]]

            # explanation faithfulness vs the true attack (evaluation metric)
            true_atk = [int(i) + 1 for i in
                        np.where(np.abs(res.falsification_signal).sum(0) > 1e-6)[0]]
            loc = localization_score(sal, true_atk)
            loc_scores.append(loc)
            # Ground-truth-FREE faithfulness (also computable at deployment,
            # unlike loc above which needs the true attacked buses):
            di = deletion_insertion_score(cnn.model, x_sample, sal, cnn.scaler_X,
                                          head="cls", device=None)

            class _PF:
                V_pu = pf_all[res.day]["V_mag"][alert_h]
            ctx = build_attack_context(
                res, _PF(), alert_h,
                load_MW[res.day, alert_h], der_gen_MW[res.day, alert_h],
                model_pred_margin=pred_margin, attack_prob=atk_prob,
                saliency_buses=sal_buses, saliency_mags=sal_mags)

            print(f"\n[LLM] Day={res.day} Hour={alert_h} {scen} | "
                  f"P(attack)={atk_prob:.2f} pred_margin={pred_margin:+.4f} | "
                  f"saliency->MGs={ctx.affected_microgrids} | "
                  f"localization P@k={loc['precision@k']:.2f} | "
                  f"deletion_drop={di['drop_frac']:.2f} insertion_recov={di['recovered_frac']:.2f}")
            report = explainer.explain(ctx)
            print(report)

            structured = parse_structured_report(report)
            ground = grounding_score(structured, sal_buses, true_atk)
            faith  = check_confidence_faithfulness(structured, ctx)
            print(f"[Grounding] parsed={ground['parsed']} "
                  f"vs_saliency(Jaccard)={ground['vs_saliency']:.2f} "
                  f"vs_ground_truth(Jaccard)={ground['vs_ground_truth']:.2f}")
            if faith["mismatch"]:
                print(f"[FAITHFULNESS WARNING] LLM stated confidence "
                      f"{faith['llm_confidence']:.2f} contradicts detector's "
                      f"own probability {faith['detector_confidence']:.2f} "
                      f"-- treat the detector's number as ground truth, not "
                      f"the LLM's narrative.")

            reports.append({"day": res.day, "hour": alert_h, "report": report,
                            "attack_prob": atk_prob, "pred_margin": pred_margin,
                            "saliency_buses": sal_buses, "localization": loc,
                            "deletion_insertion": di,
                            "structured": structured, "grounding": ground,
                            "faithfulness": faith,
                            "affected_microgrids": ctx.affected_microgrids})

        mean_loc = {m: (float(np.nanmean([s[m] for s in loc_scores]))
                        if loc_scores else float("nan"))
                    for m in ("precision@k", "recall@k", "IoU")}
        print(f"[Interpretability/{scen}] mean localization over "
              f"{len(loc_scores)} cases: {mean_loc}")

        ground_scores = [r["grounding"] for r in reports]
        mean_ground = {
            "parse_rate": float(np.mean([g["parsed"] for g in ground_scores]))
                          if ground_scores else float("nan"),
            "vs_saliency": float(np.nanmean([g["vs_saliency"] for g in ground_scores]))
                          if ground_scores else float("nan"),
            "vs_ground_truth": float(np.nanmean([g["vs_ground_truth"] for g in ground_scores]))
                          if ground_scores else float("nan"),
        }
        print(f"[Grounding/{scen}] mean over {len(ground_scores)} LLM reports: {mean_ground}")

        if args.sensitivity:
            sens = {}
            for sigma in SIGMA_LEVELS:
                sim2 = AttackSimulator(system, storage, seed=args.seed)
                a2 = [sim2.simulate_day(load_MW[d], der_gen_MW[d], scen, noise_std=sigma)
                      for d in range(min(50, args.n_days))]
                m2 = len(a2)
                sens[sigma] = {
                    "feasibility": sum(r.attack_feasible for r in a2)/m2,
                    "success":     sum(r.attack_success  for r in a2)/m2,
                }
                print(f"[Sensitivity {scen}] sigma={sigma:.1f} -> "
                      f"feas={sens[sigma]['feasibility']:.2%}, "
                      f"succ={sens[sigma]['success']:.2%}")
        else:
            sens = {}

        out = {"metrics": metrics, "feasibility": feas, "success_ratio": succ,
               "sensitivity": {str(k): v for k,v in sens.items()},
               "llm_reports": reports,
               "localization_mean": mean_loc,
               "grounding_mean": mean_ground,
               "target_der": args.target_der,
               "feature_set": args.feature_set}
        with open(os.path.join(results_dir, f"results_{scen}.json"), "w") as f:
            json.dump(out, f, indent=2, default=str)

        all_results[scen] = out

    print("\n" + "="*70 + "\n  EXPERIMENT COMPLETE\n" + "="*70)
    for scen, res in all_results.items():
        print(f"\n  {scen}: Feasibility={res['feasibility']:.2%} | "
              f"Success={res['success_ratio']:.2%}")
        for m_name, m in res["metrics"].items():
            print(f"    {m_name}: Acc={m['Accuracy']:.2f}% MSE={m['MSE_overall']:.4f}")

    print(f"\n  Results saved to: {results_dir}")


if __name__ == "__main__":
    main()
