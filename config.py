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
S_BASE_MVA      = 10.0      # System base MVA
V_BASE_KV       = 12.66     # Base voltage kV (IEEE 69-bus standard)
V_MIN_PU        = 0.90      # Minimum voltage (p.u.)
V_MAX_PU        = 1.05      # Maximum voltage (p.u.)
FREQ_HZ         = 60.0      # System frequency

# PV penetration scaling (paper uses 4x to reach ~15%)
PV_SCALE_FACTOR = 4.0
PV_PENETRATION  = 0.15      # Target solar penetration

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
