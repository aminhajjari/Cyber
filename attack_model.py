"""
FDI Attack Model for IEEE 69-Bus System
Implements: Dispatch Prediction (eq. 1) + Dispatch Falsification (eq. 2)
Base framework: Wu et al., IEEE Trans. Smart Grid, 2025
DG placement (WT/PV/BM at 23 buses): Wang et al., Energy Reports 2020

Attack Scenarios:
  S1 - Generation dispatch falsification (now targets WT/PV/BM DG units)
  S2 - Load curtailment falsification
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass

from data_loader import IEEE69BusSystem
from power_flow  import BackwardForwardSweep, PowerFlowResult, generate_daily_profiles
from config import (
    S_BASE_MVA, T_INTERVALS, ATTACK_START_H, ATTACK_END_H,
    ATTACK_MAGNITUDE, EPSILON_ATTACK, RHO_SMOOTH,
    RESERVE_FRACTION, SECURITY_THRESHOLD_MW, RANDOM_SEED,
    BUS_TO_MICROGRID, SLACK_CAPACITY_MW,
)


@dataclass
class AttackResult:
    scenario:           str
    day:                int
    attack_window:      List[int]
    original_dispatch:  np.ndarray
    falsified_dispatch: np.ndarray
    falsification_signal: np.ndarray
    system_margin_true: np.ndarray
    system_margin_monitored: np.ndarray
    attack_feasible:    bool
    attack_success:     bool
    outage_time:        Optional[int]
    # NEW: which micro-grid(s) are impacted by this attack
    affected_microgrids: List[str] = None


class DispatchPredictor:
    """
    Attacker's dispatch prediction model (eq. 1 in Wu et al.).

    CHANGED: generator buses are now derived from the 23 DG units
    (WT + PV + BM) placed per Wang et al. Table 1, instead of a generic
    "non-PV PV-type bus" search. Bus 1 (slack) still picks up residual.
    """

    def __init__(self, system: IEEE69BusSystem,
                 storage_config: dict,
                 noise_std: float = 0.0):
        self.system         = system
        self.storage_config = storage_config
        self.noise_std      = noise_std
        self.pf_solver      = BackwardForwardSweep(system)
        self.n_bus          = system.n_bus
        self.T              = T_INTERVALS

        # CHANGED: dispatchable generator buses = slack + all DG-bearing buses
        self.der_buses = [b.bus_id - 1 for b in system.buses
                           if b.der_type != "none"]
        self.gen_buses = [0] + self.der_buses   # 0 = slack (bus 1, 0-indexed)

        # Keep separate lists per DG type for attack targeting / reporting
        self.wt_buses = [b.bus_id-1 for b in system.buses if b.der_type == "WT"]
        self.pv_buses = [b.bus_id-1 for b in system.buses if b.der_type == "PV"]
        self.bm_buses = [b.bus_id-1 for b in system.buses if b.der_type == "BM"]

        self.stor_buses = list(storage_config.keys())

    def predict_dispatch(self,
                         load_MW:    np.ndarray,
                         der_gen_MW: np.ndarray,   # renamed from pv_MW: combined WT+PV+BM
                         reserve_MW: np.ndarray,
                         rng: np.random.Generator = None
                         ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Predict optimal dispatch for T time intervals.

        Parameters
        ----------
        load_MW    : (T, n_bus)
        der_gen_MW : (T, n_bus) combined WT+PV+BM generation forecast
        reserve_MW : (T,)
        rng        : random generator (imperfect info noise)

        Returns
        -------
        gen_dispatch : (T, n_bus) generator dispatch MW (slack + DG units)
        stor_dispatch: (T, n_bus)
        curtailment  : (T, n_bus)
        """
        T, n = self.T, self.n_bus

        gen_dispatch  = np.zeros((T, n))
        stor_dispatch = np.zeros((T, n))
        curtailment   = np.zeros((T, n))
        soc           = {k: cfg["soc_init"] * cfg["capacity_MWh"]
                         for k, cfg in self.storage_config.items()}

        for t in range(T):
            # DG units dispatch at their (forecast) available output
            gen_dispatch[t, self.der_buses] = der_gen_MW[t, self.der_buses]

            # Residual load after DG contribution
            der_total = der_gen_MW[t, self.der_buses].sum()
            total_load = load_MW[t].sum()
            residual = max(0, total_load - der_total)

            # FIX: slack bus covers ONLY the residual load it needs to serve.
            # Previously this line was `residual + reserve_MW[t]`, which
            # baked the required reserve directly into the dispatch target,
            # making true system margin (= gen - load - reserve) collapse to
            # ~0 MW EVERY hour by construction (only floating point noise
            # around zero) -- so any nonzero SECURITY_THRESHOLD_MW would
            # trigger a false "outage" at hour 0, before the attack window
            # even starts. Reserve is now genuine UNUSED capacity: the
            # substation/slack has a fixed import limit (SLACK_CAPACITY_MW,
            # representing the upstream transmission interconnection), and
            # margin = how much of that capacity remains unused. This
            # matches Wu et al.'s definition of system margin as "remaining
            # operational reserve," not "deviation from an exact target."
            gen_dispatch[t, 0] = residual

            if rng is not None and self.noise_std > 0:
                noise = rng.normal(0, self.noise_std, n)
                gen_dispatch[t] *= (1 + noise)

            is_peak = ATTACK_START_H <= t <= ATTACK_END_H
            for k, cfg in self.storage_config.items():
                kb = k - 1
                if is_peak and soc[k] > cfg["soc_min"] * cfg["capacity_MWh"]:
                    discharge = min(cfg["power_MW"],
                                    (soc[k] - cfg["soc_min"]*cfg["capacity_MWh"])
                                    * cfg["eta_dis"])
                    stor_dispatch[t, kb] = discharge
                    soc[k] -= discharge / cfg["eta_dis"]
                elif not is_peak and soc[k] < cfg["soc_max"] * cfg["capacity_MWh"]:
                    charge = min(cfg["power_MW"],
                                 (cfg["soc_max"]*cfg["capacity_MWh"] - soc[k])
                                 / cfg["eta_ch"])
                    stor_dispatch[t, kb] = -charge
                    soc[k] += charge * cfg["eta_ch"]

        return gen_dispatch, stor_dispatch, curtailment

    def compute_reserve(self, gen_dispatch, stor_dispatch, curtailment, load_MW):
        T = self.T
        margin = np.zeros(T)
        for t in range(T):
            total_gen = gen_dispatch[t].sum() + stor_dispatch[t].sum()
            total_load = load_MW[t].sum() - curtailment[t].sum()
            required_reserve = total_load * RESERVE_FRACTION
            margin[t] = total_gen - total_load - required_reserve
        return margin


class DispatchFalsifier:
    """Attacker's dispatch falsification model (eq. 2 in Wu et al.)."""

    def __init__(self, attack_magnitude: float = ATTACK_MAGNITUDE,
                 epsilon: float = EPSILON_ATTACK,
                 rho: float = RHO_SMOOTH):
        self.Ka      = attack_magnitude
        self.epsilon = epsilon
        self.rho     = rho

    def falsify_generation(self, gen_dispatch, reserve_MW, attack_window, rng,
                            target_buses: List[int] = None):
        """
        S1: Falsify generation dispatch signals.

        CHANGED: accepts optional `target_buses` so the attack can be
        confined to a specific DG type or micro-grid (e.g., only WT buses,
        or only buses inside MG4) — useful for studying targeted attacks
        on the Wang et al. micro-grid partition.
        """
        T, n = gen_dispatch.shape
        delta = np.zeros((T, n))
        mask = np.zeros(n, dtype=bool)
        if target_buses is not None:
            mask[target_buses] = True
        else:
            mask[:] = True

        prev_delta = np.zeros(n)
        for t in attack_window:
            target_reduction = self.Ka * reserve_MW[t]

            raw_delta = np.zeros(n)
            raw_delta[mask] = -gen_dispatch[t, mask] * self.epsilon

            total_raw = np.abs(raw_delta).sum()
            if total_raw > 1e-8:
                scale = min(1.0, target_reduction / total_raw)
                raw_delta = raw_delta * scale

            if self.rho > 0 and t > attack_window[0]:
                raw_delta = (raw_delta + self.rho * prev_delta) / (1 + self.rho)

            lower = -self.epsilon * np.abs(gen_dispatch[t])
            upper =  self.epsilon * np.abs(gen_dispatch[t])
            raw_delta = np.clip(raw_delta, lower, upper)
            raw_delta = np.maximum(raw_delta, -gen_dispatch[t])

            delta[t]   = raw_delta
            prev_delta = raw_delta

        return delta

    def falsify_load_curtailment(self, curtailment, load_MW, reserve_MW,
                                  attack_window, rng, affected_buses=None):
        T, n = curtailment.shape
        delta = np.zeros((T, n))

        if affected_buses is None:
            affected_buses = list(range(49, 69))

        prev_delta = np.zeros(n)
        for t in attack_window:
            target_impact = self.Ka * reserve_MW[t]
            raw_delta = np.zeros(n)
            for b in affected_buses:
                if b < n:
                    raw_delta[b] = load_MW[t, b] * self.epsilon

            total_raw = raw_delta.sum()
            if total_raw > 1e-8:
                raw_delta *= min(1.0, target_impact / total_raw)

            if self.rho > 0 and t > attack_window[0]:
                raw_delta = (raw_delta + self.rho * prev_delta) / (1 + self.rho)

            raw_delta = np.clip(raw_delta, 0, self.epsilon * load_MW[t])
            delta[t]   = raw_delta
            prev_delta = raw_delta

        return delta


class AttackSimulator:
    """Orchestrates the full FDI attack simulation and system margin tracking."""

    def __init__(self, system: IEEE69BusSystem,
                 storage_config: dict,
                 seed: int = RANDOM_SEED):
        self.system    = system
        self.storage   = storage_config
        self.rng       = np.random.default_rng(seed)
        self.predictor = DispatchPredictor(system, storage_config)
        self.falsifier = DispatchFalsifier()
        self.pf_solver = BackwardForwardSweep(system)

        self.attack_window = list(range(ATTACK_START_H, ATTACK_END_H + 1))

    def simulate_day(self,
                     load_MW:  np.ndarray,
                     der_gen_MW: np.ndarray,   # renamed from pv_MW
                     scenario: str = "S1",
                     noise_std: float = 0.0,
                     target_der_type: str = None  # NEW: "WT"|"PV"|"BM"|None(=all)
                     ) -> AttackResult:
        """
        Simulate one day of FDI attack.

        NEW parameter `target_der_type` lets you confine an S1 attack to a
        specific DG technology (e.g., attack only the wind turbines) to
        study which DG type's compromise is most damaging — a natural
        extension enabled by the Wang et al. multi-DG-type placement.
        """
        T = T_INTERVALS

        reserve_MW = np.array([load_MW[t].sum() * RESERVE_FRACTION
                                for t in range(T)])

        rng_attack = self.rng if noise_std > 0 else None
        gen_disp, stor_disp, curtail = self.predictor.predict_dispatch(
            load_MW, der_gen_MW, reserve_MW, rng_attack)

        target_buses = None
        if target_der_type == "WT":
            target_buses = self.predictor.wt_buses
        elif target_der_type == "PV":
            target_buses = self.predictor.pv_buses
        elif target_der_type == "BM":
            target_buses = self.predictor.bm_buses

        if scenario == "S1":
            delta = self.falsifier.falsify_generation(
                gen_disp, reserve_MW, self.attack_window, self.rng,
                target_buses=target_buses)
            falsified_gen  = gen_disp  + delta
            falsified_curt = curtail.copy()
        else:  # S2
            delta = self.falsifier.falsify_load_curtailment(
                curtail, load_MW, reserve_MW, self.attack_window, self.rng)
            falsified_gen  = gen_disp.copy()
            falsified_curt = curtail + delta

        true_margin = np.zeros(T)
        monitored_margin = np.zeros(T)
        outage_time = None
        for t in range(T):
            # FIX: system margin = remaining UNUSED slack/substation import
            # capacity, not "gen - load - reserve" (which was tautologically
            # ~0 by construction of the old dispatch formula). The slack bus
            # was dispatched to cover only the (pre-attack) residual load;
            # under attack, falsified generation/curtailment changes how
            # much the slack would ACTUALLY need to supply to keep the
            # system balanced. We recompute that required slack draw here
            # and compare it against the fixed substation capacity.
            non_slack_gen_true = (falsified_gen[t, 1:].sum()
                                   + stor_disp[t, 1:].sum())
            true_load  = load_MW[t].sum() - falsified_curt[t].sum()
            required_slack_true = max(0.0, true_load - non_slack_gen_true)
            true_resv  = reserve_MW[t]
            true_margin[t] = SLACK_CAPACITY_MW - required_slack_true - true_resv

            non_slack_gen_mon = (gen_disp[t, 1:].sum()
                                  + stor_disp[t, 1:].sum())
            mon_load = load_MW[t].sum() - curtail[t].sum()
            required_slack_mon = max(0.0, mon_load - non_slack_gen_mon)
            monitored_margin[t] = SLACK_CAPACITY_MW - required_slack_mon - true_resv

            if outage_time is None and true_margin[t] < SECURITY_THRESHOLD_MW:
                outage_time = t

        attack_feasible = np.any(
            np.abs(delta[self.attack_window]).sum(axis=1) > 0.001)
        attack_success = outage_time is not None

        original_dispatch  = gen_disp + stor_disp
        falsified_dispatch = falsified_gen + stor_disp

        # NEW: identify which micro-grid(s) the attacked buses belong to
        attacked_bus_idx = np.where(np.abs(delta).sum(axis=0) > 1e-6)[0]
        affected_mgs = sorted(set(
            BUS_TO_MICROGRID.get(idx + 1, "UNASSIGNED")
            for idx in attacked_bus_idx
        ))

        return AttackResult(
            scenario=scenario,
            day=0,
            attack_window=self.attack_window,
            original_dispatch=original_dispatch,
            falsified_dispatch=falsified_dispatch,
            falsification_signal=delta,
            system_margin_true=true_margin,
            system_margin_monitored=monitored_margin,
            attack_feasible=attack_feasible,
            attack_success=attack_success,
            outage_time=outage_time,
            affected_microgrids=affected_mgs,
        )

    def compute_attack_rates(self, load_MW_all, der_gen_MW_all,
                              scenario="S1", noise_std=0.0):
        n_days = load_MW_all.shape[0]
        results = []
        feasible = success = 0

        for day in range(n_days):
            res = self.simulate_day(load_MW_all[day], der_gen_MW_all[day],
                                    scenario, noise_std)
            res.day = day
            results.append(res)
            if res.attack_feasible:
                feasible += 1
            if res.attack_success:
                success += 1

        feas_ratio = feasible / n_days
        succ_ratio = success  / n_days
        print(f"[Attack] Scenario {scenario}, sigma={noise_std:.1f}: "
              f"Feasibility={feas_ratio:.2f}, Success={succ_ratio:.2f}")
        return {"feasibility_ratio": feas_ratio,
                "success_ratio": succ_ratio, "results": results}


if __name__ == "__main__":
    from data_loader import load_ieee69_from_excel, assign_der_units, assign_storage_buses
    from config import BUS_EXCEL_PATH
    from power_flow import generate_daily_profiles

    sys69   = load_ieee69_from_excel(BUS_EXCEL_PATH)
    sys69   = assign_der_units(sys69)
    storage = assign_storage_buses(sys69)

    load_MW, der_gen_MW, reserve, breakdown = generate_daily_profiles(sys69, n_days=10)

    simulator = AttackSimulator(sys69, storage)
    result = simulator.simulate_day(load_MW[0], der_gen_MW[0], scenario="S1")

    print(f"\nAttack feasible: {result.attack_feasible}")
    print(f"Attack success:  {result.attack_success}")
    print(f"Outage at hour:  {result.outage_time}")
    print(f"Affected micro-grids: {result.affected_microgrids}")
    print(f"True margin (16-20h): {result.system_margin_true[16:21].round(4)}")
    print(f"EMS margin  (16-20h): {result.system_margin_monitored[16:21].round(4)}")

    # NEW: targeted attack example — attack only wind turbines
    result_wt = simulator.simulate_day(load_MW[0], der_gen_MW[0],
                                        scenario="S1", target_der_type="WT")
    print(f"\n[WT-targeted attack] Affected MGs: {result_wt.affected_microgrids}")
