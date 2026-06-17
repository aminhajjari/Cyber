"""
Power Flow Solver for IEEE 69-Bus Radial Distribution System
Method: Backward-Forward Sweep (BFS) — efficient for radial/tree networks
"""

import numpy as np
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
from collections import deque

from data_loader import IEEE69BusSystem, BranchData
from config import V_BASE_KV, S_BASE_MVA, V_MIN_PU, V_MAX_PU


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

            # Backward sweep
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

            # Forward sweep
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


def generate_daily_profiles(system: IEEE69BusSystem,
                             n_days=356, seed=42, noise_std=0.02):
    """
    Generate synthetic daily load and PV generation profiles.
    Returns: load_MW (n_days,T,n_bus), pv_MW (n_days,T,n_bus), reserve (n_days,T)
    """
    rng = np.random.default_rng(seed)
    T, n = 24, system.n_bus
    hour = np.arange(T)

    base_load_shape = np.clip(
        0.4 + 0.3*np.exp(-((hour-8)**2)/8) + 0.4*np.exp(-((hour-18)**2)/8),
        0.35, 1.0)
    pv_shape = np.maximum(0, np.exp(-((hour-12)**2)/8))

    bus_loads = np.array([b.Pd_MW  for b in system.buses])
    pv_caps   = np.array([b.Pv_MW  for b in system.buses])
    pv_buses  = [b.bus_id-1 for b in system.buses if b.is_pv_bus]

    load_MW = np.zeros((n_days, T, n))
    pv_MW   = np.zeros((n_days, T, n))
    reserve = np.zeros((n_days, T))

    for day in range(n_days):
        day_scale    = 1.0 + rng.normal(0, 0.05)
        is_cloudy    = rng.random() < 0.3
        pv_day_scale = rng.uniform(0.3, 0.6) if is_cloudy else rng.uniform(0.7, 1.0)

        for t in range(T):
            noise = rng.normal(0, noise_std, n)
            load_MW[day,t,:] = np.maximum(0, bus_loads * base_load_shape[t] * day_scale * (1+noise))

            for pb in pv_buses:
                pv_n = rng.normal(0, 0.05 if is_cloudy else 0.02)
                pv_MW[day,t,pb] = max(0, pv_caps[pb]*pv_shape[t]*pv_day_scale*(1+pv_n))

            reserve[day,t] = load_MW[day,t,:].sum() * 0.05

    print(f"[PowerFlow] Generated {n_days} days × {T}h profiles for {n} buses")
    return load_MW, gen_MW, reserve, breakdown
