"""
Configuration for FDI Detection on IEEE 69-Bus System
Base attack/detection framework: Wu et al., IEEE Trans. Smart Grid, 2025
DG placement & micro-grid topology: Wang et al., Energy Reports 6 (2020) 1233-1249
Extended with LLM Explainability
"""

import os

# ─── Paths ────────────────────────────────────────────────────────────────────
BASE_DIR        = os.path.dirname(os.path.abspath(__file__))
DATA_DIR        = os.path.join(BASE_DIR, "data")
RESULTS_DIR     = "/home/gkianfar/scratch/Amin/CB/results"
MODEL_DIR       = "/home/gkianfar/scratch/Amin/CB/models"
LLM_DIR         = os.path.join(BASE_DIR, "llm")

#  IEEE 69-bus Excel file path — update this
BUS_EXCEL_PATH  = "/home/gkianfar/scratch/Amin/CB/IEEE 69 bus.xlsx"

# Sheet names in your Excel (update to match your file)
BUS_DATA_SHEET      = "BusData"       # columns: Bus, Type, Pd(kW), Qd(kVAr), Vbase(kV)
BRANCH_DATA_SHEET   = "BranchData"    # columns: From, To, R(ohm), X(ohm), B(S), RatingMVA

# ─── Power System Parameters ──────────────────────────────────────────────────
# NOTE: Wang et al. states base power=1MW / base voltage=12kV in their paper,
# but their ACTUAL network impedances (same topology as Baran & Wu standard
# 69-bus) require a 10 MVA base to keep per-unit impedance consistent with
# the standard R/X values in ohms used by load_ieee69_standard(). Using the
# paper's literal 1 MW base together with these impedances causes p.u.
# overflow (V_min collapses to ~0.78). We therefore keep S_BASE_MVA=10 for
# the power-flow/p.u. calculations (consistent with the underlying Baran &
# Wu impedance data) while still using the paper's DG placement, capacities,
# and micro-grid partition, which are independent of the MVA base choice.
S_BASE_MVA      = 10.0      # System base MVA (kept consistent with branch Z data)
V_BASE_KV       = 12.66     # Base voltage kV (standard IEEE 69-bus value)
V_MIN_PU        = 0.95      # Wang et al. constraint: [0.95, 1.05]
V_MAX_PU        = 1.05
FREQ_HZ         = 60.0      # System frequency

# ─── DG Unit Placement (Wang et al., Energy Reports 2020, Table 1) ───────────
# 23 DG units total on the standard IEEE 69-bus radial system:
#   6 Wind Turbines + 6 Photovoltaic + 11 Biomass

WT_BUSES        = [52, 43, 35, 19, 16, 13]
WT_CAPACITY_KW  = 110              # per unit (rated power, Table 1)
WT_COST_USD_MWH = 96.8

PV_BUSES        = [62, 58, 56, 50, 36, 30]
PV_CAPACITY_KW  = 150              # per unit (rated power, Table 1)
PV_COST_USD_MWH = 156.9

BM_BUSES        = [68, 57, 54, 45, 42, 38, 33, 27, 21, 15, 6]
BM_CAPACITY_KW  = [75, 75, 75, 50, 50, 50, 75, 50, 25, 50, 25, 75]  # per-bus, Table 1
BM_COST_USD_MWH = 120.2

# Wind turbine power curve (Wang et al. eq. 2)
WT_V_CI           = 3.0     # cut-in speed, m/s
WT_V_R            = 12.0    # rated speed, m/s
WT_V_CO           = 25.0    # cut-out speed, m/s
WT_WEIBULL_SHAPE  = 2.0     # beta (shape parameter of Weibull wind speed PDF)
WT_WEIBULL_SCALE  = 6.0     # alpha (scale parameter, m/s)

# PV radiation model (Wang et al. eq. 3)
PV_R_STD = 1000.0   # W/m^2, standard environment radiation
PV_R_C   = 150.0    # W/m^2, threshold radiation level

# Power factor for all DG buses (Wang et al. assumption)
DG_POWER_FACTOR = 0.85

# Backward-compatible aliases (old code references these names)
PV_SCALE_FACTOR = 1.0
PV_PENETRATION  = 0.15

# ─── Micro-Grid Partition (Wang et al., Table 3) ─────────────────────────────
# The 69-bus network is partitioned into 5 autonomous micro-grids.
MICROGRID_MAP = {
    "MG1": [49, 50, 51, 52, 53, 54],
    "MG2": [28, 29, 30, 31, 32, 33, 34, 35],
    "MG3": [18, 19, 20, 21, 22, 23, 24, 25, 26, 27],
    "MG4": [6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 40, 41, 42, 43, 44, 45,
            46, 47, 48, 55, 56, 57, 58],
    "MG5": [1, 2, 3, 4, 5, 36, 37, 38, 39, 59, 60, 61, 62, 63, 64, 65, 66, 67,
            68, 69],
}
# Reverse lookup: bus_id -> microgrid name
BUS_TO_MICROGRID = {bus: mg for mg, buses in MICROGRID_MAP.items() for bus in buses}

# ─── Time Settings ────────────────────────────────────────────────────────────
T_INTERVALS     = 24        # 24 hourly intervals per day
T_MONITORING    = 6         # |T_m|: monitoring window (hours)
T_PRED_AHEAD    = 2         # T_pred: predict this many hours ahead
T_PREP          = 1         # T_prep: required operator response window (hours)
N_DAYS          = 356       # Training days

# ─── Attack Parameters ────────────────────────────────────────────────────────
# Attack scenarios
ATTACK_SCENARIO_1 = "generation_dispatch"   # S1
ATTACK_SCENARIO_2 = "load_curtailment"      # S2

# Attack window: evening ramp-up (14:00–20:00 index = 14 to 20)
ATTACK_START_H  = 14
ATTACK_END_H    = 20

# Attack magnitude K_t^a (paper: 1.6 uniformly)
ATTACK_MAGNITUDE = 1.6

# Falsification bound epsilon^a (fraction of original dispatch)
EPSILON_ATTACK  = 0.30      # ±30% of original dispatch

# Smoothness regularization rho in falsification model
RHO_SMOOTH      = 0.5

# System reserve requirement (fraction of total load)
RESERVE_FRACTION = 0.05     # 5% reserve

# Security threshold (minimum operating reserve MW)
SECURITY_THRESHOLD_MW = 0.3  # ~5% of typical 6 MW total load, reasonable margin floor

# ─── Demand / Generation Uncertainty ─────────────────────────────────────────
DEMAND_NOISE_STD = 0.02     # 2% std Gaussian demand forecast error
GEN_COST_NOISE_STD  = 0.0   # Perfect info (relaxed in sensitivity analysis)

# ─── CNN Detection Model ──────────────────────────────────────────────────────
CNN_CONFIG = {
    "conv1d_filters":   128,
    "conv1d_kernel":    3,
    "pool_size":        1,       # max-pool
    "fc_layers":        [128, 64],
    "activation":       "relu",
    "dropout":          0.2,
    "batch_size":       32,
    "epochs":           800,
    "learning_rate":    1e-3,    # Adam default
    "train_ratio":      0.7,
    "val_ratio":        0.15,
    "test_ratio":       0.15,
}

# Feature sets (matching Table II of paper)
FEATURE_SET_1 = ["P", "V_mag"]                  # (P_i, |V_i|)
FEATURE_SET_2 = ["P", "V_mag", "theta"]         # (P_i, |V_i|, θ_i)  ← best in paper

# ─── LLM Explainability ───────────────────────────────────────────────────────
# Recommended for Narval A100: Mistral-7B-Instruct (fits in ~14GB VRAM)
LLM_MODEL_NAME  = "mistralai/Mistral-7B-Instruct-v0.3"
LLM_CACHE_DIR   = "/home/gkianfar/scratch/Amin/CB/llm_cache"
LLM_MAX_TOKENS  = 512
LLM_TEMPERATURE = 0.2       # Low temp for factual explanations

# ─── Simulation ───────────────────────────────────────────────────────────────
RANDOM_SEED     = 42
N_SIMULATIONS   = 500       # Monte Carlo runs for attack/detection evaluation

# Incomplete information sensitivity (sigma levels)
SIGMA_LEVELS    = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]
