"""
Configuration for FDI Detection on IEEE 69-Bus System
Based on: Wu et al., IEEE Trans. Smart Grid, 2025
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
S_BASE_MVA      = 1.0       # System base MVA
V_BASE_KV       = 12.0      # Base voltage kV (IEEE 69-bus standard)
V_MIN_PU        = 0.90      # Minimum voltage (p.u.)
V_MAX_PU        = 1.05      # Maximum voltage (p.u.)
FREQ_HZ         = 60.0      # System frequency

# PV penetration scaling (paper uses 4x to reach ~15%)
# ─── DG Unit Placement (Wang et al., Energy Reports 2020, Table 1) ───────────
# 23 DG units total: 6 WT + 6 PV + 11 Biomass, on the standard IEEE 69-bus system

WT_BUSES        = [52, 43, 35, 19, 16, 13]
WT_CAPACITY_KW  = 110              # per unit
WT_COST_USD_MWH = 96.8

PV_BUSES        = [62, 58, 56, 50, 36, 30]
PV_CAPACITY_KW  = 150              # per unit
PV_COST_USD_MWH = 156.9

BM_BUSES        = [68, 57, 54, 45, 42, 38, 33, 27, 21, 15, 6]
BM_CAPACITY_KW  = [75, 75, 75, 50, 50, 50, 75, 50, 25, 50, 25, 75]  # per-bus capacity
BM_COST_USD_MWH = 120.2

# Wind turbine power curve (Wang et al. eq. 2)
WT_V_CI           = 3.0     # cut-in speed, m/s
WT_V_R             = 12.0    # rated speed, m/s
WT_V_CO            = 25.0    # cut-out speed, m/s
WT_WEIBULL_SHAPE  = 2.0     # beta (shape parameter)
WT_WEIBULL_SCALE  = 6.0     # alpha (scale parameter, m/s)

# PV radiation model (Wang et al. eq. 3)
PV_R_STD = 1000.0   # W/m^2 standard environment radiation
PV_R_C   = 150.0    # W/m^2 threshold radiation

# Keep for backward compatibility with any old code paths
PV_SCALE_FACTOR = 1.0
PV_PENETRATION  = 0.15

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
SECURITY_THRESHOLD_MW = 0.5  # Scaled for 69-bus (paper uses 1MW for 187-bus)

# ─── Demand / Generation Uncertainty ─────────────────────────────────────────
DEMAND_NOISE_STD = 0.02     # 2% std Gaussian demand forecast error
GEN_COST_NOISE_STD  = 0.0   # Perfect info (relaxed in sensitivity analysis)


# ─── Micro-Grid Partition (Wang et al., Table 3) ─────────────────────────────
MICROGRID_MAP = {
    "MG1": [49, 50, 51, 52, 53, 54],
    "MG2": [28, 29, 30, 31, 32, 33, 34, 35],
    "MG3": [18, 19, 20, 21, 22, 23, 24, 25, 26, 27],
    "MG4": [6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 40, 41, 42, 43, 44, 45,
            46, 47, 48, 55, 56, 57, 58],
    "MG5": [1, 2, 3, 4, 5, 36, 37, 38, 39, 59, 60, 61, 62, 63, 64, 65, 66, 67,
            68, 69],
}
BUS_TO_MICROGRID = {bus: mg for mg, buses in MICROGRID_MAP.items() for bus in buses}

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
LLM_CACHE_DIR   = "/home/gkianfar/scratch/Amin/llm_cache"   # Narval scratch
LLM_MAX_TOKENS  = 512
LLM_TEMPERATURE = 0.2       # Low temp for factual explanations

# ─── Simulation ───────────────────────────────────────────────────────────────
RANDOM_SEED     = 42
N_SIMULATIONS   = 500       # Monte Carlo runs for attack/detection evaluation

# Incomplete information sensitivity (sigma levels)
SIGMA_LEVELS    = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]
