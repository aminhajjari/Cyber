"""
FDI Attack Model for IEEE 69-Bus System
Implements: Dispatch Prediction (eq. 1) + Dispatch Falsification (eq. 2)
Based on: Wu et al., IEEE Trans. Smart Grid, 2025

Attack Scenarios:
  S1 - Generation dispatch falsification
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
    RESERVE_FRACTION, SECURITY_THRESHOLD_MW, RANDOM_SEED
)


@dataclass
class AttackResult:
    scenario:           str          # "S1" or "S2"
    day:                int
    attack_window:      List[int]    # time indices of attack
    original_dispatch:  np.ndarray   # (T, n_bus) original scheduled dispatch
    falsified_dispatch: np.ndarray   # (T, n_bus) falsified dispatch
    falsification_signal: np.ndarray # (T, n_bus) = falsified - original
    system_margin_true: np.ndarray   # (T,) actual system margin
    system_margin_monitored: np.ndarray  # (T,) what EMS sees (falsified)
    attack_feasible:    bool
    attack_success:     bool         # margin exhausted below threshold
    outage_time:        Optional[int]


class DispatchPredictor:
    """
    Attacker's dispatch prediction model (eq. 1 in paper).
    Simplified convex approximation for distribution systems.
    Uses DistFlow-based linear OPF (avoids full SOCP for tractability).
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

        # Generator buses (non-PV, non-slack dispatchable)
        self.gen_buses = [b.bus_id-1 for b in system.buses
                          if b.bus_type == 2 and not b.is_pv_bus]
        # Add slack as main generator
        self.gen_buses = [0] + self.gen_buses

        # PV buses
        self.pv_buses  = [b.bus_id-1 for b in system.buses if b.is_pv_bus]

        # Storage buses
        self.stor_buses = list(storage_config.keys())

    def predict_dispatch(self,
                         load_MW:    np.ndarray,
                         pv_MW:      np.ndarray,
                         reserve_MW: np.ndarray,
                         rng: np.random.Generator = None
                         ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Predict optimal dispatch for T time intervals.

        Parameters
        ----------
        load_MW    : (T, n_bus) load forecast
        pv_MW      : (T, n_bus) PV generation forecast
        reserve_MW : (T,)       required reserve
        rng        : random generator (for imperfect info noise)

        Returns
        -------
        gen_dispatch : (T, n_bus) generator dispatch MW
        stor_dispatch: (T, n_bus) storage dispatch MW (+ = discharge, - = charge)
        curtailment  : (T, n_bus) load curtailment MW
        """
        T, n = self.T, self.n_bus

        gen_dispatch  = np.zeros((T, n))
        stor_dispatch = np.zeros((T, n))
        curtailment   = np.zeros((T, n))
        soc           = {k: cfg["soc_init"] * cfg["capacity_MWh"]
                         for k, cfg in self.storage_config.items()}

        for t in range(T):
            # Effective load = load - PV generation
            effective_load = np.maximum(0, load_MW[t] - pv_MW[t])
            total_eff_load = effective_load.sum()

            # Simple economic dispatch: slack bus picks up residual
            # (In full paper this is SOCP; here we use proportional dispatch)
            gen_dispatch[t, 0] = total_eff_load + reserve_MW[t]

            # Apply noise if attacker has imperfect information
            if rng is not None and self.noise_std > 0:
                noise = rng.normal(0, self.noise_std, n)
                gen_dispatch[t] *= (1 + noise)

            # Storage dispatch: charge during low net load, discharge during peak
            # Evening ramp period (attack window) → discharge storage
            is_peak = ATTACK_START_H <= t <= ATTACK_END_H
            for k, cfg in self.storage_config.items():
                kb = k - 1  # 0-indexed
                if is_peak and soc[k] > cfg["soc_min"] * cfg["capacity_MWh"]:
                    # Discharge
                    discharge = min(cfg["power_MW"],
                                    (soc[k] - cfg["soc_min"]*cfg["capacity_MWh"])
                                    * cfg["eta_dis"])
                    stor_dispatch[t, kb] = discharge
                    soc[k] -= discharge / cfg["eta_dis"]
                elif not is_peak and soc[k] < cfg["soc_max"] * cfg["capacity_MWh"]:
                    # Charge
                    charge = min(cfg["power_MW"],
                                 (cfg["soc_max"]*cfg["capacity_MWh"] - soc[k])
                                 / cfg["eta_ch"])
                    stor_dispatch[t, kb] = -charge
                    soc[k] += charge * cfg["eta_ch"]

        return gen_dispatch, stor_dispatch, curtailment

    def compute_reserve(self,
                        gen_dispatch:   np.ndarray,
                        stor_dispatch:  np.ndarray,
                        curtailment:    np.ndarray,
                        load_MW:        np.ndarray
                        ) -> np.ndarray:
        """
        Compute system upward/downward reserve margin at each time step.
        Returns (T,) array of system margin MW.
        """
        T = self.T
        margin = np.zeros(T)
        for t in range(T):
            total_gen = gen_dispatch[t].sum() + stor_dispatch[t].sum()
            total_load = load_MW[t].sum() - curtailment[t].sum()
            required_reserve = total_load * RESERVE_FRACTION
            margin[t] = total_gen - total_load - required_reserve
        return margin


class DispatchFalsifier:
    """
    Attacker's dispatch falsification model (eq. 2 in paper).
    Generates stealthy falsification signals Δx_ot.

    Objective: minimize signal magnitude + temporal smoothness
    Subject to:
      - attack impact ≥ K_t × total reserve (depletes margin)
      - |Δx| ≤ ε × |x_hat| (stealthiness bound)
      - dispatch stays within operational limits
    """

    def __init__(self, attack_magnitude: float = ATTACK_MAGNITUDE,
                 epsilon: float = EPSILON_ATTACK,
                 rho: float = RHO_SMOOTH):
        self.Ka      = attack_magnitude
        self.epsilon = epsilon
        self.rho     = rho

    def falsify_generation(self,
                           gen_dispatch: np.ndarray,
                           reserve_MW:   np.ndarray,
                           attack_window: List[int],
                           rng: np.random.Generator
                           ) -> np.ndarray:
        """
        S1: Falsify generation dispatch signals.
        Returns falsification signal Δg (T, n_bus) — negative values reduce output.
        """
        T, n = gen_dispatch.shape
        delta = np.zeros((T, n))

        prev_delta = np.zeros(n)
        for t in attack_window:
            # Target: reduce generation to exhaust reserve
            target_reduction = self.Ka * reserve_MW[t]

            # Candidate falsification (negative = reduce generation)
            raw_delta = -gen_dispatch[t] * self.epsilon

            # Normalize to achieve target reduction
            total_raw = np.abs(raw_delta).sum()
            if total_raw > 1e-8:
                scale = min(1.0, target_reduction / total_raw)
                raw_delta = raw_delta * scale

            # Temporal smoothness penalty: blend with previous
            if self.rho > 0 and t > attack_window[0]:
                raw_delta = (raw_delta + self.rho * prev_delta) / (1 + self.rho)

            # Enforce bounds: -ε × x̂ ≤ Δx ≤ ε × x̂
            lower = -self.epsilon * np.abs(gen_dispatch[t])
            upper =  self.epsilon * np.abs(gen_dispatch[t])
            raw_delta = np.clip(raw_delta, lower, upper)

            # Operational bounds: dispatch + delta ≥ 0
            raw_delta = np.maximum(raw_delta, -gen_dispatch[t])

            delta[t]   = raw_delta
            prev_delta = raw_delta

        return delta

    def falsify_load_curtailment(self,
                                  curtailment:   np.ndarray,
                                  load_MW:       np.ndarray,
                                  reserve_MW:    np.ndarray,
                                  attack_window: List[int],
                                  rng: np.random.Generator,
                                  affected_buses: List[int] = None
                                  ) -> np.ndarray:
        """
        S2: Falsify load curtailment signals (cancel curtailments).
        Returns falsification signal Δd_curt (T, n_bus) — positive values
        reduce effective curtailment (increase effective load).
        Includes geographic regularization for stealthiness.
        """
        T, n = curtailment.shape
        delta = np.zeros((T, n))

        if affected_buses is None:
            # Use lower half of network (sub-b in paper → buses 50-69)
            affected_buses = list(range(49, 69))

        prev_delta = np.zeros(n)
        for t in attack_window:
            target_impact = self.Ka * reserve_MW[t]

            # Reduce curtailment (makes load increase)
            raw_delta = np.zeros(n)
            for b in affected_buses:
                if b < n:
                    # Proportional to local load (geographic smoothness)
                    raw_delta[b] = load_MW[t, b] * self.epsilon

            # Scale to target
            total_raw = raw_delta.sum()
            if total_raw > 1e-8:
                raw_delta *= min(1.0, target_impact / total_raw)

            # Temporal + geographic smoothness
            if self.rho > 0 and t > attack_window[0]:
                raw_delta = (raw_delta + self.rho * prev_delta) / (1 + self.rho)

            # Bound: 0 ≤ delta ≤ ε × load (can only reduce curtailment, not add)
            raw_delta = np.clip(raw_delta, 0, self.epsilon * load_MW[t])

            delta[t]   = raw_delta
            prev_delta = raw_delta

        return delta


class AttackSimulator:
    """
    Orchestrates the full FDI attack simulation and system margin tracking.
    """

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
                     pv_MW:    np.ndarray,
                     scenario: str = "S1",
                     noise_std: float = 0.0
                     ) -> AttackResult:
        """
        Simulate one day of FDI attack.

        Parameters
        ----------
        load_MW  : (T, n_bus)
        pv_MW    : (T, n_bus)
        scenario : "S1" (generation) or "S2" (curtailment)
        noise_std: attacker information error level σ

        Returns
        -------
        AttackResult
        """
        T = T_INTERVALS

        # Reserve array (T,)
        reserve_MW = np.array([load_MW[t].sum() * RESERVE_FRACTION
                                for t in range(T)])

        # 1. Attacker predicts dispatch
        rng_attack = self.rng if noise_std > 0 else None
        gen_disp, stor_disp, curtail = self.predictor.predict_dispatch(
            load_MW, pv_MW, reserve_MW, rng_attack)

        # 2. Generate falsification signals
        if scenario == "S1":
            delta = self.falsifier.falsify_generation(
                gen_disp, reserve_MW, self.attack_window, self.rng)
            falsified_gen  = gen_disp  + delta
            falsified_curt = curtail.copy()
        else:  # S2
            delta = self.falsifier.falsify_load_curtailment(
                curtail, load_MW, reserve_MW, self.attack_window, self.rng)
            falsified_gen  = gen_disp.copy()
            falsified_curt = curtail + delta

        # 3. Compute true system margin (what actually happens)
        true_margin = np.zeros(T)
        monitored_margin = np.zeros(T)  # what EMS sees (falsified uplink)

        outage_time = None
        for t in range(T):
            # True: actual generation is reduced/load is increased
            true_gen   = falsified_gen[t].sum()  + stor_disp[t].sum()
            true_load  = load_MW[t].sum() - falsified_curt[t].sum()
            true_resv  = reserve_MW[t]
            true_margin[t] = true_gen - true_load - true_resv

            # Monitored: EMS sees original dispatch (attack also falsifies uplink)
            mon_gen  = gen_disp[t].sum() + stor_disp[t].sum()
            mon_load = load_MW[t].sum() - curtail[t].sum()
            monitored_margin[t] = mon_gen - mon_load - true_resv

            if outage_time is None and true_margin[t] < SECURITY_THRESHOLD_MW:
                outage_time = t

        attack_feasible = np.any(
            np.abs(delta[self.attack_window]).sum(axis=1) > 0.01)
        attack_success = outage_time is not None

        # Combine dispatch into single array for return (gen + stor)
        original_dispatch  = gen_disp + stor_disp
        falsified_dispatch = falsified_gen + stor_disp

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
        )

    def compute_attack_rates(self,
                              load_MW_all: np.ndarray,
                              pv_MW_all:   np.ndarray,
                              scenario:    str = "S1",
                              noise_std:   float = 0.0
                              ) -> Dict:
        """
        Compute attack feasibility and success ratios over all days.
        Returns dict with feasibility_ratio, success_ratio, results.
        """
        n_days = load_MW_all.shape[0]
        results = []
        feasible = 0
        success  = 0

        for day in range(n_days):
            res = self.simulate_day(load_MW_all[day], pv_MW_all[day],
                                    scenario, noise_std)
            res.day = day
            results.append(res)
            if res.attack_feasible:
                feasible += 1
            if res.attack_success:
                success += 1

        feas_ratio  = feasible / n_days
        succ_ratio  = success  / n_days

        print(f"[Attack] Scenario {scenario}, σ={noise_std:.1f}: "
              f"Feasibility={feas_ratio:.2f}, Success={succ_ratio:.2f}")

        return {
            "feasibility_ratio": feas_ratio,
            "success_ratio":     succ_ratio,
            "results":           results,
        }


if __name__ == "__main__":
    from data_loader import load_ieee69_from_excel, assign_pv_buses, assign_storage_buses
    from config import BUS_EXCEL_PATH
    from power_flow import generate_daily_profiles

    sys69   = load_ieee69_from_excel(BUS_EXCEL_PATH)
    sys69   = assign_pv_buses(sys69)
    storage = assign_storage_buses(sys69)

    load_MW, pv_MW, reserve = generate_daily_profiles(sys69, n_days=10)

    simulator = AttackSimulator(sys69, storage)
    result = simulator.simulate_day(load_MW[0], pv_MW[0], scenario="S1")

    print(f"\nAttack feasible: {result.attack_feasible}")
    print(f"Attack success:  {result.attack_success}")
    print(f"Outage at hour:  {result.outage_time}")
    print(f"True margin (16-20h): {result.system_margin_true[16:21].round(3)}")
    print(f"EMS margin  (16-20h): {result.system_margin_monitored[16:21].round(3)}")
