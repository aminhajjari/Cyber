"""
IEEE 69-Bus System Data Loader
Reads bus and branch data from your Excel file.
Falls back to hardcoded IEEE 69-bus standard data if Excel not found.

DG placement (WT/PV/Biomass) and micro-grid partition follow:
  Wang et al., "Optimal planning of multi-micro grids based-on networks
  reliability," Energy Reports 6 (2020) 1233-1249, Table 1 & Table 3.
"""

import numpy as np
import pandas as pd
import os
import warnings
from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Dict

from config import (
    BUS_EXCEL_PATH, BUS_DATA_SHEET, BRANCH_DATA_SHEET,
    S_BASE_MVA, V_BASE_KV, V_MIN_PU, V_MAX_PU,
    WT_BUSES, WT_CAPACITY_KW, WT_COST_USD_MWH,
    PV_BUSES, PV_CAPACITY_KW, PV_COST_USD_MWH,
    BM_BUSES, BM_CAPACITY_KW, BM_COST_USD_MWH,
    MICROGRID_MAP, BUS_TO_MICROGRID,
    BESS_BUSES, BESS_POWER_KW, BESS_CAPACITY_KWH,
    BESS_ETA_CH, BESS_ETA_DIS, BESS_SOC_MIN, BESS_SOC_MAX, BESS_SOC_INIT,
    EV_BUSES, EV_N_VEHICLES_PER_BUS, EV_CHARGER_KW,
    EV_CONTROLLABLE_FRACTION, EV_CHARGE_PROB_BY_HOUR,
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

    # DER flags (CHANGED: replaced single is_pv_bus with explicit DER type)
    der_type:   str   = "none"   # "none" | "WT" | "PV" | "BM"
    der_capacity_MW: float = 0.0 # rated capacity of the DER at this bus

    # Backward-compat property: old code checks `is_pv_bus`
    @property
    def is_pv_bus(self) -> bool:
        return self.der_type == "PV"

    @property
    def Pv_MW(self) -> float:
        """Backward-compat alias: renewable (PV or WT) capacity at this bus."""
        return self.der_capacity_MW if self.der_type in ("PV", "WT") else 0.0

    @property
    def microgrid(self) -> str:
        return BUS_TO_MICROGRID.get(self.bus_id, "UNASSIGNED")


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
    Load IEEE 69-bus data from the user's Excel file.

    CHANGED (topology-fidelity fix): this function previously expected a
    generic "BusData"/"BranchData" sheet layout with kW/ohm units, and
    silently fell back to load_ieee69_standard() (a DIFFERENT, hardcoded
    69-bus branch topology) whenever the user's actual file didn't match
    that layout -- which it never did, since the real uploaded file is a
    MATPOWER-style case export with sheets named "Bus Data" / "Line data",
    a few header/comment rows before the real column row, loads already in
    MW (not kW), and impedances already in PER-UNIT (not ohms).

    We now parse the user's actual file directly. This matters because a
    direct branch-by-branch comparison showed the previous hardcoded
    fallback topology (load_ieee69_standard) differs from this file in 5
    branches -- e.g. bus 36 branches directly off bus 3 in the real data,
    but the old hardcoded version chained it as 35->36 instead. Using the
    user's real file here removes that topology-fidelity gap for the base
    network (DG/BESS/EV bus placements from Wang et al. are layered on top
    of whichever topology is loaded, and are unaffected by this fix).

    Expected sheet layout (MATPOWER case export):
      "Bus Data"  : header row contains bus_i, type, Pd, Qd, ... (Pd/Qd in MW/MVAr)
      "Line data" : header row contains fbus, tbus, r, x, b, ... (r/x in p.u.)
    """
    if not os.path.exists(excel_path):
        warnings.warn(
            f"Excel file not found at: {excel_path}\n"
            "Falling back to built-in IEEE 69-bus standard data.",
            UserWarning
        )
        return load_ieee69_standard()

    print(f"[DataLoader] Loading IEEE 69-bus from: {excel_path}")

    def _find_header_row(excel_path, sheet_name, key_col_name, max_scan=6):
        """Scan the first few rows to find which row holds the real
        column headers (MATPOWER exports often have comment/blank rows
        before the actual header)."""
        raw = pd.read_excel(excel_path, sheet_name=sheet_name, header=None,
                             nrows=max_scan)
        for i in range(len(raw)):
            row_vals = [str(v).strip().lower() for v in raw.iloc[i].tolist()]
            if key_col_name in row_vals:
                return i
        return 0  # fallback: assume first row is header

    # ── Bus Data ──────────────────────────────────────────────────────────
    try:
        bus_sheet_name = None
        xl = pd.ExcelFile(excel_path)
        for s in xl.sheet_names:
            if "bus" in s.lower():
                bus_sheet_name = s
                break
        if bus_sheet_name is None:
            bus_sheet_name = BUS_DATA_SHEET

        hdr_row = _find_header_row(excel_path, bus_sheet_name, "bus_i")
        df_bus = pd.read_excel(excel_path, sheet_name=bus_sheet_name, header=hdr_row)
        df_bus.columns = [str(c).strip().lower() for c in df_bus.columns]
        df_bus = df_bus[pd.to_numeric(df_bus.iloc[:, 0], errors="coerce").notna()]
    except Exception as e:
        warnings.warn(f"Could not read bus sheet: {e}. Using standard data.")
        return load_ieee69_standard()

    def find_col(df, candidates):
        for c in candidates:
            if c in df.columns:
                return c
        return None

    col_map_bus = {
        "bus":  ["bus_i", "bus", "bus_id", "node", "bus no", "busno"],
        "type": ["type", "bus_type", "bustype"],
        "pd":   ["pd", "pd_kw", "pd(kw)", "p_kw", "pload"],
        "qd":   ["qd", "qd_kvar", "qd(kvar)", "q_kvar", "qload"],
        "basekv": ["basekv", "vbase_kv", "vbase", "kv"],
    }
    bus_col   = find_col(df_bus, col_map_bus["bus"])
    type_col  = find_col(df_bus, col_map_bus["type"])
    pd_col    = find_col(df_bus, col_map_bus["pd"])
    qd_col    = find_col(df_bus, col_map_bus["qd"])
    basekv_col = find_col(df_bus, col_map_bus["basekv"])

    # Unit auto-detection: MATPOWER case files store Pd in MW already.
    # If pd values look implausibly large for MW (e.g. >50 on average for a
    # 69-bus feeder), they are probably actually in kW -- guard against
    # double conversion errors either way.
    raw_pd_vals = pd.to_numeric(df_bus[pd_col], errors="coerce").fillna(0) if pd_col else None
    pd_is_kw = False
    if raw_pd_vals is not None and len(raw_pd_vals) > 0:
        nonzero = raw_pd_vals[raw_pd_vals > 0]
        if len(nonzero) > 0 and nonzero.mean() > 50:
            pd_is_kw = True   # values like 75, 145 etc. with no decimal -> likely kW

    buses = []
    for _, row in df_bus.iterrows():
        bus_id   = int(row[bus_col]) if bus_col else int(row.iloc[0])
        bus_type = int(row[type_col]) if type_col else 1
        Pd_raw   = float(row[pd_col])  if pd_col  else 0.0
        Qd_raw   = float(row[qd_col])  if qd_col  else 0.0
        Pd_MW    = Pd_raw / 1000.0 if pd_is_kw else Pd_raw
        Qd_MVAr  = Qd_raw / 1000.0 if pd_is_kw else Qd_raw
        buses.append(BusData(
            bus_id   = bus_id,
            bus_type = bus_type,
            Pd_MW    = Pd_MW,
            Qd_MVAr  = Qd_MVAr,
        ))

    base_kv_used = V_BASE_KV
    if basekv_col is not None:
        kv_vals = pd.to_numeric(df_bus[basekv_col], errors="coerce").dropna()
        if len(kv_vals) > 0 and kv_vals.iloc[0] > 0:
            base_kv_used = float(kv_vals.iloc[0])

    # ── Branch ("Line") Data ─────────────────────────────────────────────
    try:
        line_sheet_name = None
        for s in xl.sheet_names:
            if "line" in s.lower() or "branch" in s.lower():
                line_sheet_name = s
                break
        if line_sheet_name is None:
            line_sheet_name = BRANCH_DATA_SHEET

        hdr_row_br = _find_header_row(excel_path, line_sheet_name, "fbus")
        df_br = pd.read_excel(excel_path, sheet_name=line_sheet_name, header=hdr_row_br)
        df_br.columns = [str(c).strip().lower() for c in df_br.columns]
        df_br = df_br[pd.to_numeric(df_br.iloc[:, 0], errors="coerce").notna()]
    except Exception as e:
        warnings.warn(f"Could not read branch sheet: {e}. Using standard data.")
        return load_ieee69_standard()

    col_map_br = {
        "from":   ["fbus", "from", "from_bus", "sending"],
        "to":     ["tbus", "to",   "to_bus",   "receiving"],
        "r":      ["r", "r_ohm", "r(ohm)", "resistance"],
        "x":      ["x", "x_ohm", "x(ohm)", "reactance"],
        "b":      ["b", "b_s", "b(s)", "susceptance"],
        "rating": ["ratea", "ratingmva", "rating_mva", "rating", "mva"],
    }
    from_col = find_col(df_br, col_map_br["from"])
    to_col   = find_col(df_br, col_map_br["to"])
    r_col    = find_col(df_br, col_map_br["r"])
    x_col    = find_col(df_br, col_map_br["x"])
    b_col    = find_col(df_br, col_map_br["b"])
    rate_col = find_col(df_br, col_map_br["rating"])

    Z_base = (base_kv_used ** 2) / S_BASE_MVA   # ohms

    # Unit auto-detection for r/x: MATPOWER case files store these already
    # in per-unit (e.g. 0.0000310 for the first 69-bus feeder segment).
    # Values that small can ONLY be p.u. -- raw ohms for a 12.66kV feeder
    # segment are always >> 0.001. If avg |r| < 0.01 -> already p.u.
    r_is_pu = True
    if r_col is not None:
        r_vals = pd.to_numeric(df_br[r_col], errors="coerce").dropna()
        if len(r_vals) > 0 and r_vals.abs().mean() > 0.05:
            r_is_pu = False   # values like 0.3, 1.7 etc -> likely ohms

    branches = []
    for _, row in df_br.iterrows():
        r_raw = float(row[r_col]) if r_col else 0.0001
        x_raw = float(row[x_col]) if x_col else 0.0001
        R_pu = r_raw if r_is_pu else r_raw / Z_base
        X_pu = x_raw if r_is_pu else x_raw / Z_base
        B_pu = float(row[b_col]) if b_col else 0.0
        rating = float(row[rate_col]) if (rate_col and not pd.isna(row[rate_col]) and row[rate_col] != 0) else S_BASE_MVA
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

    print(f"[DataLoader] Loaded {system.n_bus} buses, {system.n_branch} branches "
          f"from user's Excel file (sheets: '{bus_sheet_name}', '{line_sheet_name}')")
    print(f"[DataLoader] Total load: {system.total_load_MW:.4f} MW, "
          f"{system.total_load_MVAr:.4f} MVAr | base_kV={base_kv_used} | "
          f"r/x units detected as: {'p.u.' if r_is_pu else 'ohms'}")
    return system


def load_ieee69_standard() -> IEEE69BusSystem:
    """
    Hardcoded IEEE 69-bus standard data.
    Source: Baran & Wu (1989), widely used benchmark.
    Units: kW, kVAr for loads; ohms for impedances.
    """
    print("[DataLoader] Using built-in IEEE 69-bus standard data.")

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
        i = br.from_bus - 1
        j = br.to_bus   - 1
        y_ij = 1.0 / complex(br.R_pu, br.X_pu)
        b_sh  = 1j * br.B_pu / 2.0

        Y[i, i] += y_ij + b_sh
        Y[j, j] += y_ij + b_sh
        Y[i, j] -= y_ij
        Y[j, i] -= y_ij

    system.Y_bus = Y


# ═══════════════════════════════════════════════════════════════════════════
# CHANGED: assign_pv_buses() replaced by assign_der_units()
# Now places WT + PV + Biomass exactly as in Wang et al. Table 1
# (23 DG units total, instead of just 5 PV buses)
# ═══════════════════════════════════════════════════════════════════════════

def assign_der_units(system: IEEE69BusSystem) -> IEEE69BusSystem:
    """
    Assign DG units (Wind Turbine, PV, Biomass) to buses exactly as specified
    in Wang et al. (2020), Table 1:
      - 6 Wind Turbines  @ buses [52,43,35,19,16,13], 110 kW each
      - 6 PV systems     @ buses [62,58,56,50,36,30], 150 kW each
      - 11 Biomass units @ buses [68,57,54,45,42,38,33,27,21,15,6],
                            capacities [75,75,75,50,50,50,75,50,25,50,25,75] kW

    Total: 23 DG units, matching the paper's stated count exactly.
    """
    bus_map = {b.bus_id: b for b in system.buses}

    # Wind turbines
    for bid in WT_BUSES:
        if bid in bus_map:
            bus_map[bid].der_type = "WT"
            bus_map[bid].der_capacity_MW = WT_CAPACITY_KW / 1000.0
            bus_map[bid].bus_type = 2  # treat as generator bus

    # Photovoltaic
    for bid in PV_BUSES:
        if bid in bus_map:
            bus_map[bid].der_type = "PV"
            bus_map[bid].der_capacity_MW = PV_CAPACITY_KW / 1000.0
            bus_map[bid].bus_type = 2

    # Biomass (dispatchable, paired with per-bus capacity list)
    for bid, cap_kw in zip(BM_BUSES, BM_CAPACITY_KW):
        if bid in bus_map:
            bus_map[bid].der_type = "BM"
            bus_map[bid].der_capacity_MW = cap_kw / 1000.0
            bus_map[bid].bus_type = 2

    n_wt = len(WT_BUSES)
    n_pv = len(PV_BUSES)
    n_bm = len(BM_BUSES)
    print(f"[DataLoader] DG units assigned: {n_wt} WT + {n_pv} PV + {n_bm} BM "
          f"= {n_wt+n_pv+n_bm} total (Wang et al. Table 1)")
    print(f"[DataLoader]   WT buses: {WT_BUSES}")
    print(f"[DataLoader]   PV buses: {PV_BUSES}")
    print(f"[DataLoader]   BM buses: {BM_BUSES}")
    return system


def assign_pv_buses(system: IEEE69BusSystem,
                    pv_bus_ids: List[int] = None,
                    scale_factor: float = 1.0) -> IEEE69BusSystem:
    """
    DEPRECATED — kept only for backward compatibility with old scripts.
    New code should call assign_der_units(system) instead, which places
    WT + PV + BM exactly per Wang et al. Table 1.
    """
    warnings.warn(
        "assign_pv_buses() is deprecated; use assign_der_units() to match "
        "Wang et al. (2020) Table 1 DG placement.", DeprecationWarning)
    return assign_der_units(system)


def get_microgrid_for_bus(bus_id: int) -> str:
    """Return the micro-grid name ('MG1'..'MG5') that a bus belongs to."""
    return BUS_TO_MICROGRID.get(bus_id, "UNASSIGNED")


def get_microgrid_summary(system: IEEE69BusSystem) -> Dict[str, dict]:
    """
    Summarize each micro-grid: bus count, total load, DG units inside it.
    Useful for LLM context and for attack-impact analysis per MG.
    """
    summary = {}
    for mg_name, bus_ids in MICROGRID_MAP.items():
        mg_buses = [b for b in system.buses if b.bus_id in bus_ids]
        ders = [(b.bus_id, b.der_type, b.der_capacity_MW)
                for b in mg_buses if b.der_type != "none"]
        summary[mg_name] = {
            "n_buses":       len(mg_buses),
            "bus_ids":       bus_ids,
            "total_load_MW": sum(b.Pd_MW for b in mg_buses),
            "der_units":     ders,
            "total_der_MW":  sum(d[2] for d in ders),
        }
    return summary


def assign_storage_buses(system: IEEE69BusSystem,
                         storage_bus_ids: List[int] = None,
                         capacity_MWh: float = None,
                         power_MW: float = None) -> dict:
    """
    Assign Battery Energy Storage System (BESS) units to specific buses.

    CHANGED: previously this used arbitrary placeholder buses [6, 25, 50]
    with a single uniform size for all units. Neither source paper specifies
    a BESS model (Wang et al. has no storage at all; Wu et al. references
    "storage" generically without siting it). This is now a deliberately
    designed addition: one BESS per micro-grid (Wang et al. Table 3),
    sited at each MG's largest DG-hosting bus, sized to ~2h of that MG's
    DER capacity (see config.py BESS_* block for the documented rationale).

    If storage_bus_ids/capacity_MWh/power_MW are explicitly passed, they
    override the config.py defaults (kept for flexibility/backward compat),
    otherwise the Wang-et-al-aligned per-MG siting in config.py is used.
    """
    if storage_bus_ids is None:
        storage_bus_ids = BESS_BUSES
        power_list = [p/1000.0 for p in BESS_POWER_KW]
        cap_list   = [c/1000.0 for c in BESS_CAPACITY_KWH]
    else:
        # Caller supplied a custom bus list -> use uniform fallback sizing
        power_list = [power_MW or 0.02] * len(storage_bus_ids)
        cap_list   = [capacity_MWh or 0.05] * len(storage_bus_ids)

    bus_map = {b.bus_id: b for b in system.buses}
    storage = {}
    for bid, p_mw, e_mwh in zip(storage_bus_ids, power_list, cap_list):
        storage[bid] = {
            "capacity_MWh": e_mwh,
            "power_MW":     p_mw,
            "soc_min":      BESS_SOC_MIN,
            "soc_max":      BESS_SOC_MAX,
            "eta_ch":       BESS_ETA_CH,
            "eta_dis":      BESS_ETA_DIS,
            "soc_init":     BESS_SOC_INIT,
        }
        # Tag the bus so DER-type reporting / LLM context can see it
        if bid in bus_map and bus_map[bid].der_type == "none":
            bus_map[bid].der_type = "BESS_host"   # informational only;
            # NOTE: BESS is NOT counted as a generation DER (it's net-zero
            # over a cycle), so der_capacity_MW is intentionally left at 0
            # here to avoid double counting it in DG totals/Table-1 counts.

    print(f"[DataLoader] BESS units assigned: {len(storage)} units at "
          f"buses {storage_bus_ids} (total power="
          f"{sum(power_list)*1000:.0f} kW, total capacity="
          f"{sum(cap_list)*1000:.0f} kWh) [project addition, see config.py]")
    return storage


def assign_ev_units(system: IEEE69BusSystem) -> Dict[int, dict]:
    """
    Assign Electric Vehicle (EV) charging load to selected buses.

    NEW — project addition. Neither Wu et al. nor Wang et al. models EVs.
    EV charging is represented as a controllable, time-varying ADDITIVE
    load (it stacks on top of the bus's existing Pd_MW from the base case),
    with a per-bus fleet size and a per-vehicle charger rating. A fraction
    of this load (EV_CONTROLLABLE_FRACTION) is flagged as demand-response
    capable, making it a natural extension target for an S2-style (load
    curtailment) FDI attack alongside the existing curtailable loads.

    Returns
    -------
    dict[bus_id] -> {
        "n_vehicles": int,
        "charger_kw": float,
        "max_load_MW": float,          # n_vehicles * charger_kw, if all charge at once
        "controllable_fraction": float,
    }
    """
    ev_config = {}
    bus_map = {b.bus_id: b for b in system.buses}

    for bid, n_veh in zip(EV_BUSES, EV_N_VEHICLES_PER_BUS):
        max_load_MW = (n_veh * EV_CHARGER_KW) / 1000.0
        ev_config[bid] = {
            "n_vehicles":   n_veh,
            "charger_kw":   EV_CHARGER_KW,
            "max_load_MW":  max_load_MW,
            "controllable_fraction": EV_CONTROLLABLE_FRACTION,
        }
        if bid in bus_map:
            # Mark informationally; EV load is a DEMAND not a DG, so it does
            # not change der_type/der_capacity_MW (those remain reserved for
            # generation-side DG units per Wang et al. Table 1).
            pass

    total_ev_MW = sum(c["max_load_MW"] for c in ev_config.values())
    print(f"[DataLoader] EV charging assigned: {len(ev_config)} buses, "
          f"{sum(EV_N_VEHICLES_PER_BUS)} vehicles total, "
          f"max simultaneous load={total_ev_MW*1000:.1f} kW "
          f"[project addition, see config.py]")
    return ev_config


if __name__ == "__main__":
    sys69 = load_ieee69_from_excel(BUS_EXCEL_PATH)
    sys69 = assign_der_units(sys69)
    storage = assign_storage_buses(sys69)
    ev_units = assign_ev_units(sys69)

    print(f"\nY-bus shape: {sys69.Y_bus.shape}")

    print("\n--- Micro-Grid Summary ---")
    summary = get_microgrid_summary(sys69)
    for mg, info in summary.items():
        print(f"{mg}: {info['n_buses']} buses, "
              f"load={info['total_load_MW']*1000:.1f} kW, "
              f"DER={info['total_der_MW']*1000:.1f} kW "
              f"({len(info['der_units'])} units)")

    print("\n--- BESS Summary ---")
    for bid, cfg in storage.items():
        print(f"  Bus {bid}: {cfg['power_MW']*1000:.0f} kW / "
              f"{cfg['capacity_MWh']*1000:.0f} kWh")

    print("\n--- EV Summary ---")
    for bid, cfg in ev_units.items():
        print(f"  Bus {bid}: {cfg['n_vehicles']} vehicles, "
              f"max load {cfg['max_load_MW']*1000:.1f} kW "
              f"({cfg['controllable_fraction']*100:.0f}% controllable)")
