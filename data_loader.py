"""
IEEE 69-Bus System Data Loader
Reads bus and branch data from your Excel file.
Falls back to hardcoded IEEE 69-bus standard data if Excel not found.
"""

import numpy as np
import pandas as pd
import os
import warnings
from dataclasses import dataclass, field
from typing import List, Tuple, Optional

from config import (
    BUS_EXCEL_PATH, BUS_DATA_SHEET, BRANCH_DATA_SHEET,
    S_BASE_MVA, V_BASE_KV, V_MIN_PU, V_MAX_PU
)


@dataclass
class BusData:
    bus_id:     int
    bus_type:   int     # 1=PQ load, 2=PV gen, 3=slack
    Pd_MW:      float   # Active load MW
    Qd_MVAr:    float   # Reactive load MVAr
    Pg_MW:      float = 0.0   # Active generation MW
    Qg_MW:      float = 0.0
    Vbase_kV:   float = V_BASE_KV
    V_pu:       float = 1.0
    theta_rad:  float = 0.0
    is_pv_bus:  bool  = False
    Pv_MW:      float = 0.0   # PV generation capacity MW


@dataclass
class BranchData:
    from_bus:   int
    to_bus:     int
    R_pu:       float   # Resistance p.u.
    X_pu:       float   # Reactance p.u.
    B_pu:       float = 0.0
    rating_MVA: float = 10.0


@dataclass
class IEEE69BusSystem:
    buses:          List[BusData]
    branches:       List[BranchData]
    n_bus:          int = 69
    n_branch:       int = 68
    slack_bus:      int = 1
    total_load_MW:  float = 0.0
    total_load_MVAr: float = 0.0

    # Incidence matrix and admittance matrix (built after loading)
    Y_bus:      Optional[np.ndarray] = field(default=None, repr=False)
    Z_base:     float = 0.0


def load_ieee69_from_excel(excel_path: str) -> IEEE69BusSystem:
    """
    Load IEEE 69-bus data from Excel file.

    Expected sheets:
      BusData   : Bus | Type | Pd_kW | Qd_kVAr | Vbase_kV
      BranchData: From | To | R_ohm | X_ohm | B_S | RatingMVA
    """
    if not os.path.exists(excel_path):
        warnings.warn(
            f"Excel file not found at: {excel_path}\n"
            "Falling back to built-in IEEE 69-bus standard data.",
            UserWarning
        )
        return load_ieee69_standard()

    print(f"[DataLoader] Loading IEEE 69-bus from: {excel_path}")

    # ── Bus Data ──────────────────────────────────────────────────────────────
    try:
        df_bus = pd.read_excel(excel_path, sheet_name=BUS_DATA_SHEET)
        df_bus.columns = [c.strip().lower() for c in df_bus.columns]
    except Exception as e:
        warnings.warn(f"Could not read bus sheet: {e}. Using standard data.")
        return load_ieee69_standard()

    # Flexible column name mapping
    col_map_bus = {
        "bus":      ["bus", "bus_id", "node", "bus no", "busno"],
        "type":     ["type", "bus_type", "bustype"],
        "pd_kw":    ["pd_kw", "pd(kw)", "p_kw", "pd", "pload"],
        "qd_kvar":  ["qd_kvar", "qd(kvar)", "q_kvar", "qd", "qload"],
        "vbase_kv": ["vbase_kv", "vbase", "kv", "voltage_kv"],
    }

    def find_col(df, candidates):
        for c in candidates:
            if c in df.columns:
                return c
        return None

    bus_col  = find_col(df_bus, col_map_bus["bus"])
    type_col = find_col(df_bus, col_map_bus["type"])
    pd_col   = find_col(df_bus, col_map_bus["pd_kw"])
    qd_col   = find_col(df_bus, col_map_bus["qd_kvar"])

    buses = []
    for _, row in df_bus.iterrows():
        bus_id   = int(row[bus_col]) if bus_col else int(row.iloc[0])
        bus_type = int(row[type_col]) if type_col else 1
        Pd_kW    = float(row[pd_col])  if pd_col  else 0.0
        Qd_kVAr  = float(row[qd_col])  if qd_col  else 0.0
        buses.append(BusData(
            bus_id   = bus_id,
            bus_type = bus_type,
            Pd_MW    = Pd_kW  / 1000.0,
            Qd_MVAr  = Qd_kVAr / 1000.0,
        ))

    # ── Branch Data ───────────────────────────────────────────────────────────
    try:
        df_br = pd.read_excel(excel_path, sheet_name=BRANCH_DATA_SHEET)
        df_br.columns = [c.strip().lower() for c in df_br.columns]
    except Exception as e:
        warnings.warn(f"Could not read branch sheet: {e}. Using standard data.")
        return load_ieee69_standard()

    col_map_br = {
        "from":    ["from", "from_bus", "fbus", "sending"],
        "to":      ["to",   "to_bus",   "tbus", "receiving"],
        "r_ohm":   ["r_ohm", "r(ohm)", "resistance", "r"],
        "x_ohm":   ["x_ohm", "x(ohm)", "reactance",  "x"],
        "b_s":     ["b_s",   "b(s)",   "susceptance", "b"],
        "rating":  ["ratingmva", "rating_mva", "rating", "mva"],
    }

    from_col   = find_col(df_br, col_map_br["from"])
    to_col     = find_col(df_br, col_map_br["to"])
    r_col      = find_col(df_br, col_map_br["r_ohm"])
    x_col      = find_col(df_br, col_map_br["x_ohm"])
    b_col      = find_col(df_br, col_map_br["b_s"])
    rate_col   = find_col(df_br, col_map_br["rating"])

    # Base impedance for p.u. conversion
    Z_base = (V_BASE_KV ** 2) / S_BASE_MVA   # ohms

    branches = []
    for _, row in df_br.iterrows():
        R_pu = float(row[r_col]) / Z_base if r_col else 0.01
        X_pu = float(row[x_col]) / Z_base if x_col else 0.01
        B_pu = float(row[b_col]) * Z_base  if b_col else 0.0
        rating = float(row[rate_col]) if rate_col else S_BASE_MVA
        branches.append(BranchData(
            from_bus   = int(row[from_col]) if from_col else int(row.iloc[0]),
            to_bus     = int(row[to_col])   if to_col   else int(row.iloc[1]),
            R_pu       = R_pu,
            X_pu       = X_pu,
            B_pu       = B_pu,
            rating_MVA = rating,
        ))

    system = IEEE69BusSystem(buses=buses, branches=branches,
                              n_bus=len(buses), n_branch=len(branches))
    system.total_load_MW   = sum(b.Pd_MW  for b in buses)
    system.total_load_MVAr = sum(b.Qd_MVAr for b in buses)
    system.Z_base = Z_base
    _build_ybus(system)

    print(f"[DataLoader] Loaded {system.n_bus} buses, {system.n_branch} branches")
    print(f"[DataLoader] Total load: {system.total_load_MW:.3f} MW, "
          f"{system.total_load_MVAr:.3f} MVAr")
    return system


def load_ieee69_standard() -> IEEE69BusSystem:
    """
    Hardcoded IEEE 69-bus standard data.
    Source: Baran & Wu (1989), widely used benchmark.
    Units: kW, kVAr for loads; ohms for impedances.
    """
    print("[DataLoader] Using built-in IEEE 69-bus standard data.")

    # Format: (bus_id, type, Pd_kW, Qd_kVAr)
    # Bus 1 = slack (substation), type 3
    bus_raw = [
        (1,  3,  0.0,   0.0),
        (2,  1,  0.0,   0.0),
        (3,  1,  0.0,   0.0),
        (4,  1,  0.0,   0.0),
        (5,  1,  0.0,   0.0),
        (6,  1, 2.6,   2.2),
        (7,  1, 40.0,   30.0),
        (8,  1, 75.0,   54.0),
        (9,  1, 30.0,   22.0),
        (10, 1, 28.0,   19.0),
        (11, 1, 145.0, 104.0),
        (12, 1, 145.0, 104.0),
        (13, 1,  8.0,   5.0),
        (14, 1,  8.0,   5.0),
        (15, 1,  0.0,   0.0),
        (16, 1, 45.5,  30.0),
        (17, 1, 60.0,  35.0),
        (18, 1, 60.0,  35.0),
        (19, 1,  0.0,   0.0),
        (20, 1,  1.0,   0.6),
        (21, 1,114.0,  81.0),
        (22, 1,  5.0,   3.5),
        (23, 1,  0.0,   0.0),
        (24, 1, 28.0,  20.0),
        (25, 1,  0.0,   0.0),
        (26, 1, 14.0,  10.0),
        (27, 1, 14.0,  10.0),
        (28, 1, 26.0,  18.6),
        (29, 1, 26.0,  18.6),
        (30, 1,  0.0,   0.0),
        (31, 1,  0.0,   0.0),
        (32, 1,  0.0,   0.0),
        (33, 1, 14.0,  10.0),
        (34, 1, 19.5,  14.0),
        (35, 1,  6.0,   4.0),
        (36, 1, 26.0,  18.55),
        (37, 1, 26.0,  18.55),
        (38, 1,  0.0,   0.0),
        (39, 1, 24.0,  17.0),
        (40, 1, 24.0,  17.0),
        (41, 1,  1.2,   1.0),
        (42, 1,  0.0,   0.0),
        (43, 1,  6.0,   4.3),
        (44, 1,  0.0,   0.0),
        (45, 1, 39.22, 26.3),
        (46, 1, 39.22, 26.3),
        (47, 1,  0.0,   0.0),
        (48, 1, 79.0,  56.4),
        (49, 1,384.7, 274.5),
        (50, 1,384.7, 274.5),
        (51, 1, 40.5,  28.3),
        (52, 1,  3.6,   2.7),
        (53, 1,  4.35,  3.5),
        (54, 1, 26.4,  19.0),
        (55, 1, 24.0,  17.2),
        (56, 1,  0.0,   0.0),
        (57, 1,  0.0,   0.0),
        (58, 1,  0.0,   0.0),
        (59, 1,100.0,  72.0),
        (60, 1,  0.0,   0.0),
        (61, 1,  1244.0, 888.0),
        (62, 1, 32.0,  23.0),
        (63, 1,  0.0,   0.0),
        (64, 1,227.0, 162.0),
        (65, 1, 59.0,  42.0),
        (66, 1, 18.0,  13.0),
        (67, 1, 18.0,  13.0),
        (68, 1, 28.0,  20.0),
        (69, 1, 28.0,  20.0),
    ]

    buses = [BusData(bus_id=b[0], bus_type=b[1],
                     Pd_MW=b[2]/1000.0, Qd_MVAr=b[3]/1000.0)
             for b in bus_raw]

    # Format: (from, to, R_ohm, X_ohm)  — IEEE 69-bus standard impedances
    branch_raw = [
        (1,2,  0.0005, 0.0012), (2,3,  0.0005, 0.0012), (3,4,  0.0015, 0.0036),
        (4,5,  0.0251, 0.0294), (5,6,  0.3660, 0.1864), (6,7,  0.3810, 0.1941),
        (7,8,  0.0922, 0.0470), (8,9,  0.0493, 0.0251), (9,10, 0.8190, 0.2707),
        (10,11,0.1872, 0.0619), (11,12,0.7114, 0.2351), (12,13,1.0300, 0.3400),
        (13,14,1.0440, 0.3450), (14,15,1.0580, 0.3496), (15,16,0.1966, 0.0650),
        (16,17,0.3744, 0.1238), (17,18,0.0047, 0.0016), (18,19,0.3276, 0.1083),
        (19,20,0.2106, 0.0690), (20,21,0.3416, 0.1129), (21,22,0.0140, 0.0046),
        (22,23,0.1591, 0.0526), (23,24,0.3463, 0.1145), (24,25,0.7488, 0.2475),
        (25,26,0.3089, 0.1021), (26,27,0.1732, 0.0572), (3,28,  0.0044, 0.0108),
        (28,29,0.0640, 0.1565), (29,30,0.3978, 0.1315), (30,31,0.0702, 0.0232),
        (31,32,0.3510, 0.1160), (32,33,0.8390, 0.2816), (33,34,1.7080, 0.5646),
        (34,35,1.4740, 0.4873), (35,36,0.7570, 0.2500), (36,37,0.7570, 0.2500),
        (3,38,  0.0015, 0.0036), (38,39,0.3180, 0.0845), (39,40,1.0278, 0.3596),
        (40,41,0.2290, 0.0755), (41,42,0.3378, 0.1115), (42,43,0.1546, 0.0515),
        (43,44,0.1626, 0.0535), (44,45,0.1730, 0.0572), (45,46,0.2030, 0.0675),
        (46,47,0.2842, 0.0938), (47,48,0.2813, 0.0930), (48,49,1.5900, 0.5337),
        (49,50,0.7837, 0.2630), (8,51,  0.4512, 0.3083), (51,52,0.6981, 0.2083),
        (9,53,  0.8980, 0.7091), (53,54,0.8960, 0.7011), (54,55,0.2030, 0.1034),
        (55,56,0.2842, 0.1447), (56,57,1.0590, 0.9338), (57,58,0.7796, 0.6269),
        (58,59,1.4706, 1.1551), (59,60,0.4556, 0.3588), (60,61,0.7152, 0.5825),
        (61,62,0.6989, 0.5819), (62,63,0.9500, 0.8209), (63,64,1.3254, 1.0550),
        (64,65,0.6027, 0.4856), (65,66,0.7006, 0.5765), (66,67,1.9290, 1.7342),
        (67,68,0.9103, 0.8265), (68,69,0.4100, 0.3750),
    ]

    Z_base = (V_BASE_KV ** 2) / S_BASE_MVA
    branches = []
    for fr, to, r, x in branch_raw:
        branches.append(BranchData(
            from_bus=fr, to_bus=to,
            R_pu=r/Z_base, X_pu=x/Z_base,
            B_pu=0.0, rating_MVA=S_BASE_MVA
        ))

    system = IEEE69BusSystem(buses=buses, branches=branches,
                              n_bus=69, n_branch=len(branches))
    system.total_load_MW   = sum(b.Pd_MW  for b in buses)
    system.total_load_MVAr = sum(b.Qd_MVAr for b in buses)
    system.Z_base = Z_base
    _build_ybus(system)

    print(f"[DataLoader] IEEE 69-bus: {system.n_bus} buses, "
          f"{system.n_branch} branches, "
          f"Total load: {system.total_load_MW*1000:.1f} kW")
    return system


def _build_ybus(system: IEEE69BusSystem):
    """Build the Y-bus admittance matrix (complex, n_bus × n_bus)."""
    n = system.n_bus
    Y = np.zeros((n, n), dtype=complex)

    for br in system.branches:
        i = br.from_bus - 1   # 0-indexed
        j = br.to_bus   - 1
        y_ij = 1.0 / complex(br.R_pu, br.X_pu)
        b_sh  = 1j * br.B_pu / 2.0

        Y[i, i] += y_ij + b_sh
        Y[j, j] += y_ij + b_sh
        Y[i, j] -= y_ij
        Y[j, i] -= y_ij

    system.Y_bus = Y


def assign_pv_buses(system: IEEE69BusSystem,
                    pv_bus_ids: List[int] = None,
                    scale_factor: float = 4.0) -> IEEE69BusSystem:
    """
    Assign PV generation to selected buses.
    Default: buses 11, 21, 33, 49, 62 (spread across feeders).
    """
    if pv_bus_ids is None:
        # Representative buses across the three main feeders
        pv_bus_ids = [11, 21, 33, 49, 62]

    for bus in system.buses:
        if bus.bus_id in pv_bus_ids:
            # PV capacity = scale_factor × local load (as in paper, 4x)
            bus.is_pv_bus = True
            bus.Pv_MW     = bus.Pd_MW * scale_factor
            bus.bus_type  = 2   # PV → PV bus

    print(f"[DataLoader] PV assigned to buses: {pv_bus_ids}")
    return system


def assign_storage_buses(system: IEEE69BusSystem,
                         storage_bus_ids: List[int] = None,
                         capacity_MWh: float = 0.5,
                         power_MW: float = 0.2) -> dict:
    """
    Assign energy storage parameters to selected buses.
    Returns storage config dict used by the dispatch model.
    """
    if storage_bus_ids is None:
        storage_bus_ids = [6, 25, 50]

    storage = {}
    for bid in storage_bus_ids:
        storage[bid] = {
            "capacity_MWh": capacity_MWh,
            "power_MW":     power_MW,
            "soc_min":      0.10,
            "soc_max":      0.90,
            "eta_ch":       0.95,
            "eta_dis":      0.95,
            "soc_init":     0.50,
        }
    print(f"[DataLoader] Storage at buses: {storage_bus_ids}")
    return storage


if __name__ == "__main__":
    # Quick test
    sys69 = load_ieee69_from_excel(BUS_EXCEL_PATH)
    sys69 = assign_pv_buses(sys69)
    storage = assign_storage_buses(sys69)
    print(f"\nY-bus shape: {sys69.Y_bus.shape}")
    print(f"Sample Y-bus diagonal (bus 1): {sys69.Y_bus[0,0]:.4f}")
