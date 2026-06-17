"""
Power Flow Solver for IEEE 69-Bus Radial Distribution System
Method: Backward-Forward Sweep (BFS) — efficient for radial/tree networks

DG generation models (WT/PV/BM) follow Wang et al. (2020):
  - Wind turbine: Weibull wind speed distribution + piecewise power curve (eq. 1-2)
  - PV: radiation-based piecewise power model (eq. 3)
  - Biomass: constant dispatchable output (always available, per paper assumption)
"""

import numpy as np
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
from collections import deque

from data_loader import IEEE69BusSystem, BranchData
from config import (V_BASE_KV, S_BASE_MVA, V_MIN_PU, V_MAX_PU,
                    WT_V_CI, WT_V_R, WT_V_CO, WT_WEIBULL_SHAPE, WT_WEIBULL_SCALE,
                    PV_R_STD, PV_R_C)


@dataclass
class PowerFlowResult:
    converged:      bool
    n_iter:         int
    V_pu:           np.ndarray
    theta_rad:      np.ndarray
    P_inj_MW:       np.ndarray
    Q_inj_MVAr:     np.ndarray
    P_loss_MW:      float
    Q_loss_MVAr:    float
    V_min_pu:       float
    V_max_pu:       float
    V_min_bus:      int
    branch_flows:   dict
    system_margin:  float = 0.0


class BackwardForwardSweep:
    """BFS power flow for radial IEEE 69-bus distribution network."""

    def __init__(self, system: IEEE69BusSystem, max_iter=100, tol=1e-6):
        self.system   = system
        self.max_iter = max_iter
        self.tol      = tol
        self.n_bus    = system.n_bus
        self._build_tree()

    def _build_tree(self):
        self.children   = {i: [] for i in range(1, self.n_bus + 1)}
        self.parent     = {}
        self.branch_map = {}

        for br in self.system.branches:
            self.children[br.from_bus].append(br.to_bus)
            self.parent[br.to_bus] = br.from_bus
            self.branch_map[(br.from_bus, br.to_bus)] = br

        self.bfs_order = []
        visited = set()
        q = deque([1])
        while q:
            node = q.popleft()
            if node in visited:
                continue
            visited.add(node)
            self.bfs_order.append(node)
            for child in self.children[node]:
                q.append(child)

    def solve(self, P_load_MW, Q_load_MVAr,
              P_gen_MW, Q_gen_MVAr, V_slack_pu=1.0) -> PowerFlowResult:
        n = self.n_bus
        P_net_pu = (P_gen_MW   - P_load_MW)   / S_BASE_MVA
        Q_net_pu = (Q_gen_MVAr - Q_load_MVAr) / S_BASE_MVA

        V = np.ones(n + 1, dtype=complex)
        V[1] = complex(V_slack_pu, 0.0)

        S_branch = {}
        converged = False

        for iteration in range(self.max_iter):
            V_prev = V.copy()

            P_s = {bus: -P_net_pu[bus-1] for bus in range(1, n+1)}
            Q_s = {bus: -Q_net_pu[bus-1] for bus in range(1, n+1)}

            for bus in reversed(self.bfs_order[1:]):
                par = self.parent[bus]
                br  = self.branch_map[(par, bus)]
                R, X = br.R_pu, br.X_pu
                P_flow, Q_flow = P_s[bus], Q_s[bus]
                V_bus2 = max(abs(V[bus])**2, 1e-8)
                I2 = (P_flow**2 + Q_flow**2) / V_bus2
                P_s[par] += P_flow + R * I2
                Q_s[par] += Q_flow + X * I2
                S_branch[(par, bus)] = complex(P_flow, Q_flow)

            for bus in self.bfs_order[1:]:
                par = self.parent[bus]
                br  = self.branch_map[(par, bus)]
                R, X = br.R_pu, br.X_pu
                P_flow = S_branch[(par, bus)].real
                Q_flow = S_branch[(par, bus)].imag
                V_par  = max(abs(V[par]), 1e-6)
                dV = (R * P_flow + X * Q_flow) / V_par
                dT = (X * P_flow - R * Q_flow) / V_par
                V[bus] = max(V_par - dV, 0.001) * np.exp(1j*(np.angle(V[par]) - dT))

            if np.max(np.abs(np.abs(V[1:]) - np.abs(V_prev[1:]))) < self.tol:
                converged = True
                break

        V_mag   = np.array([abs(V[b])       for b in range(1, n+1)])
        V_theta = np.array([np.angle(V[b])  for b in range(1, n+1)])

        branch_flows = {}
        P_loss_total = Q_loss_total = 0.0
        for (fr, to), S in S_branch.items():
            branch_flows[(fr, to)] = (S.real * S_BASE_MVA, S.imag * S_BASE_MVA)
            br = self.branch_map[(fr, to)]
            V2 = max(abs(V[fr])**2, 1e-8)
            I2 = (S.real**2 + S.imag**2) / V2
            P_loss_total += br.R_pu * I2 * S_BASE_MVA
            Q_loss_total += br.X_pu * I2 * S_BASE_MVA

        return PowerFlowResult(
            converged=converged, n_iter=iteration+1,
            V_pu=V_mag, theta_rad=V_theta,
            P_inj_MW=P_gen_MW - P_load_MW,
            Q_inj_MVAr=Q_gen_MVAr - Q_load_MVAr,
            P_loss_MW=P_loss_total, Q_loss_MVAr=Q_loss_total,
            V_min_pu=float(np.min(V_mag)), V_max_pu=float(np.max(V_mag)),
            V_min_bus=int(np.argmin(V_mag))+1,
            branch_flows=branch_flows,
        )


# ═══════════════════════════════════════════════════════════════════════════
# NEW: Wind turbine and PV generation models (Wang et al. eq. 1-3)
# ═══════════════════════════════════════════════════════════════════════════

def sample_wind_speed(rng: np.random.Generator, n_samples: int,
                       shape: float = WT_WEIBULL_SHAPE,
                       scale: float = WT_WEIBULL_SCALE) -> np.ndarray:
    """
    Sample wind speed from Weibull distribution (Wang et al. eq. 1).
    f(v) = (beta/alpha) * (v/alpha)^(beta-1) * exp(-(v/alpha)^beta)
    """
    return rng.weibull(shape, n_samples) * scale


def wind_turbine_power(v: np.ndarray, p_rated_MW: float,
                        v_ci: float = WT_V_CI, v_r: float = WT_V_R,
                        v_co: float = WT_V_CO) -> np.ndarray:
    """
    Wind turbine power curve (Wang et al. eq. 2):
      P_WT = 0                                  if v < v_ci or v > v_co
      P_WT = p_rated * (v - v_ci)/(v_r - v_ci)   if v_ci <= v < v_r
      P_WT = p_rated                             if v_r <= v <= v_co
    """
    v = np.atleast_1d(v)
    P = np.zeros_like(v, dtype=float)

    ramp_mask = (v >= v_ci) & (v < v_r)
    P[ramp_mask] = p_rated_MW * (v[ramp_mask] - v_ci) / (v_r - v_ci)

    rated_mask = (v >= v_r) & (v <= v_co)
    P[rated_mask] = p_rated_MW

    return P


def pv_power(R: np.ndarray, p_rs_MW: float,
             R_std: float = PV_R_STD, R_c: float = PV_R_C) -> np.ndarray:
    """
    PV output power vs. solar irradiance (Wang et al. eq. 3):
      P(R) = p_rs * R^2/(R_std*R_c)   if 0 <= R <= R_c
      P(R) = p_rs * R/R_std            if R_c <= R <= R_std
      P(R) = p_rs                      if R_std <= R
    """
    R = np.atleast_1d(R)
    P = np.zeros_like(R, dtype=float)

    low_mask = (R >= 0) & (R <= R_c)
    P[low_mask] = p_rs_MW * (R[low_mask]**2) / (R_std * R_c)

    mid_mask = (R > R_c) & (R <= R_std)
    P[mid_mask] = p_rs_MW * R[mid_mask] / R_std

    high_mask = R > R_std
    P[high_mask] = p_rs_MW

    return P


def generate_daily_profiles(system: IEEE69BusSystem,
                             n_days=356, seed=42, noise_std=0.02):
    """
    Generate synthetic daily load and DG generation profiles.

    CHANGED: now generates separate WT (Weibull wind + power curve),
    PV (radiation-based), and BM (constant dispatchable) output per bus,
    instead of only PV. Returns combined `pv_MW` (renamed conceptually to
    "DER output") for backward compatibility with the rest of the pipeline,
    plus a breakdown dict for diagnostics / LLM context.

    Returns
    -------
    load_MW    : (n_days, T, n_bus)  hourly load at each bus
    gen_MW     : (n_days, T, n_bus)  hourly TOTAL DER output (WT+PV+BM) at each bus
                  (kept as `pv_MW` name in call sites for backward compat)
    reserve    : (n_days, T)         system reserve each hour
    breakdown  : dict with 'WT', 'PV', 'BM' -> (n_days, T, n_bus) arrays
    """
    rng = np.random.default_rng(seed)
    T, n = 24, system.n_bus
    hour = np.arange(T)

    # Load shape: morning rise, midday plateau, evening peak
    base_load_shape = np.clip(
        0.4 + 0.3*np.exp(-((hour-8)**2)/8) + 0.4*np.exp(-((hour-18)**2)/8),
        0.35, 1.0)

    # Solar irradiance shape (bell curve, peak at solar noon ~12:00)
    # Scaled to W/m^2 so it interacts correctly with PV_R_STD / PV_R_C
    irradiance_shape = np.maximum(0, np.exp(-((hour-12)**2)/8))  # 0..1

    # Wind speed has a mild diurnal pattern (slightly higher in afternoon)
    # but is dominated by the Weibull stochastic draw each hour
    wind_diurnal_factor = 1.0 + 0.15*np.sin((hour-6)*np.pi/12)

    bus_loads = np.array([b.Pd_MW for b in system.buses])

    wt_buses = {b.bus_id-1: b.der_capacity_MW for b in system.buses if b.der_type == "WT"}
    pv_buses = {b.bus_id-1: b.der_capacity_MW for b in system.buses if b.der_type == "PV"}
    bm_buses = {b.bus_id-1: b.der_capacity_MW for b in system.buses if b.der_type == "BM"}

    load_MW   = np.zeros((n_days, T, n))
    wt_out_MW = np.zeros((n_days, T, n))
    pv_out_MW = np.zeros((n_days, T, n))
    bm_out_MW = np.zeros((n_days, T, n))
    reserve   = np.zeros((n_days, T))

    for day in range(n_days):
        day_scale  = 1.0 + rng.normal(0, 0.05)
        is_cloudy  = rng.random() < 0.3
        irr_day_scale = rng.uniform(0.3, 0.6) if is_cloudy else rng.uniform(0.7, 1.0)

        for t in range(T):
            # ── Load ──────────────────────────────────────────────────────
            noise = rng.normal(0, noise_std, n)
            load_MW[day,t,:] = np.maximum(
                0, bus_loads * base_load_shape[t] * day_scale * (1+noise))

            # ── Wind turbines (Weibull speed -> power curve) ───────────────
            for bidx, p_rated in wt_buses.items():
                v = sample_wind_speed(rng, 1)[0] * wind_diurnal_factor[t]
                wt_out_MW[day, t, bidx] = wind_turbine_power(
                    np.array([v]), p_rated)[0]

            # ── PV (irradiance -> power curve) ──────────────────────────────
            R_t = irradiance_shape[t] * irr_day_scale * PV_R_STD
            R_t *= (1 + rng.normal(0, 0.05 if is_cloudy else 0.02))
            R_t = max(0, R_t)
            for bidx, p_rs in pv_buses.items():
                pv_out_MW[day, t, bidx] = pv_power(np.array([R_t]), p_rs)[0]

            # ── Biomass (constant dispatchable, small noise) ───────────────
            for bidx, p_bm in bm_buses.items():
                bm_out_MW[day, t, bidx] = p_bm * (1 + rng.normal(0, 0.01))

            reserve[day,t] = load_MW[day,t,:].sum() * 0.05

    gen_MW = wt_out_MW + pv_out_MW + bm_out_MW   # combined DER output

    print(f"[PowerFlow] Generated {n_days} days x {T}h profiles for {n} buses")
    print(f"[PowerFlow]   WT capacity: {sum(wt_buses.values())*1000:.0f} kW "
          f"across {len(wt_buses)} buses")
    print(f"[PowerFlow]   PV capacity: {sum(pv_buses.values())*1000:.0f} kW "
          f"across {len(pv_buses)} buses")
    print(f"[PowerFlow]   BM capacity: {sum(bm_buses.values())*1000:.0f} kW "
          f"across {len(bm_buses)} buses")

    breakdown = {"WT": wt_out_MW, "PV": pv_out_MW, "BM": bm_out_MW}
    return load_MW, gen_MW, reserve, breakdown
