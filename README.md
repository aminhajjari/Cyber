# FDI Attack Detection on IEEE 69-Bus with LLM Explainability

**Based on:** Wu et al., "Learning-Based Detection of Intraday False Data Injection 
Attacks on DER Dispatch Signals," *IEEE Trans. Smart Grid*, Vol. 16, No. 6, Nov. 2025.

**Extension:** LLM (Mistral-7B-Instruct) for attack localization, severity assessment, 
and countermeasure recommendations.

---

## Project Structure

```
fdi_detection/
├── config.py           ← All hyperparameters, paths, system settings
├── data_loader.py      ← IEEE 69-bus loader (your Excel or built-in standard data)
├── power_flow.py       ← Backward-Forward Sweep power flow + daily profile generator
├── attack_model.py     ← FDI attack (eq.1 dispatch prediction + eq.2 falsification)
├── detection_model.py  ← CNN / MLP / SVR detection models (Table I architecture)
├── llm_explainer.py    ← Mistral-7B wrapper + rule-based fallback
├── main_train.py       ← Full experiment pipeline (all steps)
├── quick_check.py      ← Fast end-to-end test (run before SLURM submission)
├── run_fdi.sh          ← SLURM job script for Narval A100
├── requirements.txt
└── data/
    └── ieee69bus.xlsx  ← YOUR Excel file goes here
```

---

## Quick Start

### 1. Provide your Excel file
Place your IEEE 69-bus Excel file at `data/ieee69bus.xlsx`.

**Expected sheets:**
- `BusData`:   columns `Bus`, `Type`, `Pd_kW`, `Qd_kVAr`
- `BranchData`: columns `From`, `To`, `R_ohm`, `X_ohm`

If the file is missing or columns have different names, the code falls back 
to the hardcoded IEEE 69-bus standard data automatically.

### 2. Quick check (locally or on login node)
```bash
python quick_check.py --excel data/ieee69bus.xlsx
```

### 3. Run on Narval

**Setup environment (once):**
```bash
module load StdEnv/2023 python/3.10.13 cuda/12.2
virtualenv --no-download ~/ENV
source ~/ENV/bin/activate
pip install --no-index torch torchvision numpy pandas openpyxl scikit-learn
pip install transformers accelerate huggingface_hub
```

**Download Mistral-7B (once, ~14GB):**
```bash
huggingface-cli download mistralai/Mistral-7B-Instruct-v0.3 \
    --local-dir /home/gkianfar/scratch/Amin/llm_cache/Mistral-7B-Instruct-v0.3
```

**Submit job:**
```bash
sbatch run_fdi.sh
```

---

## Key Hyperparameters (config.py)

| Parameter | Value | Description |
|-----------|-------|-------------|
| `T_MONITORING` | 6h | Monitoring window \|T_m\| |
| `T_PRED_AHEAD` | 2h | Prediction horizon T_pred |
| `T_PREP` | 1h | Operator response window T_prep |
| `ATTACK_MAGNITUDE` | 1.6 | K_t^a (attack magnitude) |
| `EPSILON_ATTACK` | 0.30 | ε (falsification bound, 30%) |
| `RESERVE_FRACTION` | 5% | System reserve requirement |
| `SECURITY_THRESHOLD_MW` | 0.5 | Min operating reserve (MW) |

---

## LLM Models for Narval A100 (40GB)

| Model | Size | VRAM | Notes |
|-------|------|------|-------|
| `mistralai/Mistral-7B-Instruct-v0.3` | ~14GB fp16 | ~16GB | **Recommended** |
| `microsoft/phi-3-mini-4k-instruct` | ~4GB | ~6GB | Faster, good for testing |
| `meta-llama/Meta-Llama-3-8B-Instruct` | ~16GB fp16 | ~18GB | Slightly better quality |

---

## What the LLM Explains

For each detected attack, the LLM provides:

1. **ATTACK LOCATION** — specific buses/feeders/DERs affected
2. **SEVERITY** — Critical/High/Medium with justification
3. **MECHANISM** — how the FDI attack depletes the system margin
4. **IMMEDIATE ACTIONS** — numbered operator steps (next 30 min)
5. **PREVENTIVE MEASURES** — security hardening actions (24h)
6. **RECOVERY TIME** — estimated restoration timeline

---

## Results (Table II equivalent)

Expected performance on IEEE 69-bus (approximate):

| Model | Accuracy | Precision | TPR | FPR | MSE |
|-------|----------|-----------|-----|-----|-----|
| CNN (S1) | ~96% | ~93% | ~96% | ~6% | ~0.10 |
| MLP (S1) | ~87% | ~83% | ~93% | ~19% | ~0.14 |
| SVR (S1) | ~32% | ~30% | ~26% | ~61% | ~0.49 |
| CNN (S2) | ~98% | ~97% | ~100% | ~3% | ~0.04 |

*(Numbers will vary with 69-bus vs 187-bus and synthetic data)*
