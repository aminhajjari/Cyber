"""
LLM Explainability Module for FDI Attack Detection
====================================================
Wraps Mistral-7B-Instruct (recommended for Narval A100) to generate:
  1. Attack location identification (now including which micro-grid)
  2. Severity assessment
  3. Countermeasure recommendations
  4. Operator action plan

Micro-grid context (Wang et al., Energy Reports 2020, Table 3) is now
included in the prompt/report so the LLM can reason about MG-level
impact and islanding response, not just raw bus numbers.

--- HOW TO DOWNLOAD ON NARVAL ---
  module load python/3.10.13 cuda/12.2
  source ~/ENV/bin/activate
  huggingface-cli download mistralai/Mistral-7B-Instruct-v0.3 \
      --local-dir /home/gkianfar/scratch/Amin/CB/llm_cache/Mistral-7B-Instruct-v0.3
"""

import numpy as np
from typing import Dict, List, Optional
from dataclasses import dataclass

from config import (LLM_MODEL_NAME, LLM_CACHE_DIR, LLM_MAX_TOKENS,
                    LLM_TEMPERATURE, SECURITY_THRESHOLD_MW,
                    ATTACK_START_H, ATTACK_END_H,
                    WT_BUSES, PV_BUSES, BM_BUSES,
                    MICROGRID_MAP, BUS_TO_MICROGRID)


@dataclass
class AttackContext:
    scenario:           str
    current_hour:       int
    predicted_margin:   float
    actual_margin:      float
    margin_history:     List[float]
    alarm_triggered:    bool
    hours_to_outage:    Optional[int]
    top_anomaly_buses:  List[int]
    anomaly_magnitudes: List[float]
    bus_voltages:       np.ndarray
    total_load_MW:      float
    total_gen_MW:       float
    total_pv_MW:        float          # kept name for backward compat (= total DER gen)
    reserve_MW:         float
    falsification_signal: Optional[np.ndarray] = None
    confidence:         float = 0.0
    # NEW: micro-grid(s) impacted by the attack
    affected_microgrids: List[str] = None


SYSTEM_PROMPT = """You are an expert power system security analyst specializing in 
cybersecurity of smart grids, micro-grids, and distributed energy resources (DERs).
You analyze False Data Injection (FDI) attack alerts from an AI detection system 
monitoring an IEEE 69-bus distribution network that has been partitioned into 5
autonomous micro-grids (MG1-MG5), each containing a mix of wind turbines (WT),
photovoltaic (PV), and biomass (BM) generators.

Provide concise, technical, actionable analysis. Format as:
- ATTACK LOCATION: [specific buses/feeders/DERs/micro-grid(s)]
- SEVERITY: [Critical/High/Medium] with brief justification
- MECHANISM: [2-3 sentences on how the attack works]
- MICRO-GRID IMPACT: [can the affected MG(s) safely island, or do they depend
  on the main grid right now?]
- IMMEDIATE ACTIONS (next 30 min): [numbered list]
- PREVENTIVE MEASURES (24h): [numbered list]
- ESTIMATED RECOVERY TIME: [if outage occurs]"""


def _der_type_for_bus(bus_id: int) -> str:
    if bus_id in WT_BUSES: return "WT"
    if bus_id in PV_BUSES: return "PV"
    if bus_id in BM_BUSES: return "BM"
    return "load-only"


def _build_prompt(ctx: AttackContext) -> str:
    v_min_bus = int(np.argmin(ctx.bus_voltages)) + 1
    v_min_val = float(np.min(ctx.bus_voltages))

    anomaly_str = ", ".join([
        f"Bus {b} [{_der_type_for_bus(b)}, {BUS_TO_MICROGRID.get(b,'?')}] ({m:+.3f} MW)"
        for b, m in zip(ctx.top_anomaly_buses[:5], ctx.anomaly_magnitudes[:5])
    ]) or "Localizing..."

    scenario_desc = {"S1": "generation dispatch falsification",
                     "S2": "load curtailment falsification"}.get(ctx.scenario, "unknown")
    urgency = (f"{ctx.hours_to_outage}h to predicted outage"
               if ctx.hours_to_outage else "outage timing uncertain")

    mg_str = ", ".join(ctx.affected_microgrids) if ctx.affected_microgrids else "Unknown"

    return f"""FDI ATTACK ALERT — Hour {ctx.current_hour:02d}:00
Status: {'[ALARM]' if ctx.alarm_triggered else '[WARNING]'} | {urgency}
Confidence: {ctx.confidence*100:.1f}% | Type: {scenario_desc}
AFFECTED MICRO-GRID(S): {mg_str}

SYSTEM STATE:
  Load={ctx.total_load_MW:.4f} MW | Gen={ctx.total_gen_MW:.4f} MW | DER(WT+PV+BM)={ctx.total_pv_MW:.4f} MW
  Reserve={ctx.reserve_MW:.4f} MW (threshold={SECURITY_THRESHOLD_MW} MW)
  Predicted margin (2h ahead)={ctx.predicted_margin:.4f} MW
  Min voltage={v_min_val:.4f} pu at Bus {v_min_bus}

MARGIN HISTORY (last {len(ctx.margin_history)}h): {[f"{m:.3f}" for m in ctx.margin_history]}
ANOMALOUS BUSES: {anomaly_str}

NETWORK: IEEE 69-bus radial, partitioned into 5 micro-grids:
  MG1(buses 49-54) MG2(28-35) MG3(18-27) MG4(6-17,40-48,55-58) MG5(1-5,36-39,59-69)
DG PLACEMENT: WT at {WT_BUSES} | PV at {PV_BUSES} | BM at {BM_BUSES}
ATTACK WINDOW: Hours {ATTACK_START_H}-{ATTACK_END_H} (evening duck-curve ramp-up)

Provide expert attack analysis, micro-grid impact assessment, and countermeasures."""


class LLMExplainer:
    """LLM-based explainability for FDI attack detection with MG-aware context."""

    def __init__(self, use_llm: bool = True,
                 model_name: str = LLM_MODEL_NAME,
                 cache_dir:  str = LLM_CACHE_DIR):
        self.use_llm   = use_llm
        self.model     = None
        self.tokenizer = None
        if use_llm:
            self._load(model_name, cache_dir)

    def _load(self, model_name, cache_dir):
        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer
            import torch
            local_path = f"{cache_dir}/{model_name.split('/')[-1]}"
            print(f"[LLM] Loading {model_name}...")
            self.tokenizer = AutoTokenizer.from_pretrained(
                local_path, trust_remote_code=True)
            self.model = AutoModelForCausalLM.from_pretrained(
                local_path, torch_dtype=torch.float16,
                device_map="auto", trust_remote_code=True)
            self.model.eval()
            print("[LLM] Loaded successfully.")
        except Exception as e:
            print(f"[LLM] Load failed: {e}\n[LLM] Using rule-based fallback.")
            self.use_llm = False

    def explain(self, ctx: AttackContext) -> str:
        return self._llm_explain(ctx) if (self.use_llm and self.model) \
               else self._rule_explain(ctx)

    def _llm_explain(self, ctx: AttackContext) -> str:
        import torch
        messages = [{"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user",   "content": _build_prompt(ctx)}]
        if hasattr(self.tokenizer, "apply_chat_template"):
            text = self.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True)
        else:
            text = f"[INST] {SYSTEM_PROMPT}\n\n{_build_prompt(ctx)} [/INST]"
        inputs = self.tokenizer(text, return_tensors="pt").to(self.model.device)
        with torch.no_grad():
            out = self.model.generate(**inputs, max_new_tokens=LLM_MAX_TOKENS,
                                       temperature=LLM_TEMPERATURE,
                                       do_sample=(LLM_TEMPERATURE > 0),
                                       pad_token_id=self.tokenizer.eos_token_id)
        return self.tokenizer.decode(out[0][inputs["input_ids"].shape[1]:],
                                      skip_special_tokens=True).strip()

    def _rule_explain(self, ctx: AttackContext) -> str:
        """Rule-based expert explanation (no GPU required)."""
        margin_pct = ctx.predicted_margin / max(ctx.total_load_MW, 1e-6) * 100
        if ctx.alarm_triggered and ctx.hours_to_outage is not None and ctx.hours_to_outage <= 1:
            severity = "CRITICAL"
        elif ctx.alarm_triggered:
            severity = "HIGH"
        elif ctx.predicted_margin < SECURITY_THRESHOLD_MW * 3:
            severity = "MEDIUM"
        else:
            severity = "LOW"

        buses_str = ", ".join(
            f"{b}[{_der_type_for_bus(b)}]" for b in ctx.top_anomaly_buses[:3]
        ) or "Under investigation"

        mg_list = ctx.affected_microgrids or []
        mg_str  = ", ".join(mg_list) if mg_list else "Unknown"

        deficit   = max(0, SECURITY_THRESHOLD_MW - ctx.predicted_margin)
        outage_str = (f"Predicted outage in {ctx.hours_to_outage}h."
                      if ctx.hours_to_outage else "Monitor closely next 2h.")

        # NEW: micro-grid islanding assessment
        if len(mg_list) == 1:
            mg_impact = (f"{mg_list[0]} is the sole affected micro-grid. If its "
                        f"internal DG output cannot cover local load, it CANNOT "
                        f"safely island and must remain grid-tied during this event.")
        elif len(mg_list) > 1:
            mg_impact = (f"Multiple micro-grids affected ({mg_str}) — coordinated "
                        f"attack pattern. Cross-MG power exchange (tie-switches) "
                        f"may be needed to compensate.")
        else:
            mg_impact = "Micro-grid attribution pending further localization."

        if ctx.scenario == "S1":
            mechanism = (
                f"Generation dispatch falsification: DG outputs at buses {buses_str} "
                f"are being under-dispatched by up to {max(ctx.anomaly_magnitudes[:3], default=0):.4f} MW. "
                f"The EMS uplink is simultaneously spoofed, masking the deviation. "
                f"The attack accumulates during the evening solar ramp-down to exhaust the "
                f"{ctx.reserve_MW:.4f} MW system reserve."
            )
            immediate_actions = [
                f"Activate emergency ramp-up on biomass (BM) units in {mg_str} "
                f"(always-dispatchable, fastest response)",
                f"Bypass automated dispatch for buses {buses_str} — switch to manual control",
                "Cross-check WT/PV output against independent weather/irradiance data",
                f"Initiate load shedding in {mg_str} if reserve drops below "
                f"{SECURITY_THRESHOLD_MW*2:.4f} MW",
                "Verify tie-switch status between affected MG and neighboring MGs",
                "Alert upstream grid operator and request emergency import capacity",
            ]
        elif ctx.scenario == "S2":
            mechanism = (
                f"Load curtailment falsification: demand response signals at downstream "
                f"buses ({buses_str}) are overridden, preventing load reduction of up to "
                f"{max(ctx.anomaly_magnitudes[:3], default=0):.4f} MW. "
                f"The attack exploits geographic correlation within {mg_str} — adjacent "
                f"buses show similar anomalies."
            )
            immediate_actions = [
                f"Manually activate load curtailment at buses {buses_str} (bypass comms)",
                f"Direct contact with demand response customers in {mg_str}",
                f"Increase BM/dispatchable generation by {deficit*1.2:.4f} MW",
                "Isolate the compromised sub-feeder communication segment",
                "Switch demand response control to out-of-band channel",
                f"Monitor all buses within {mg_str} for correlated anomalies",
            ]
        else:
            mechanism = "Attack type under investigation."
            immediate_actions = [
                "Activate full emergency operations protocol",
                "Switch ALL DG dispatch to manual override immediately",
                "Increase generation reserve by 15% of current load",
            ]

        preventive = [
            "Implement IEC 62351-7 cryptographic authentication on DG dispatch signals",
            "Deploy independent PMU-based cross-validation for EMS monitoring uplinks",
            "Install hardware security modules (HSM) at each DG communication interface",
            f"Increase operating reserve in {mg_str} to 10% during evening ramp (14-20h)",
            "Establish dedicated secure SCADA segment per micro-grid",
            "Retrain CNN detection model with attack patterns from this incident",
            "Test islanding readiness for all 5 micro-grids under N-1 DG failure",
        ]

        return f"""
================================================================================
   FDI ATTACK ANALYSIS REPORT — Hour {ctx.current_hour:02d}:00 — {severity}
================================================================================

ATTACK LOCATION:
  Affected buses: {buses_str}
  Affected micro-grid(s): {mg_str}
  Attack type: {'S1 - Generation dispatch falsification' if ctx.scenario=='S1' else 'S2 - Load curtailment falsification' if ctx.scenario=='S2' else 'Unknown'}
  Detection confidence: {ctx.confidence*100:.1f}%

SEVERITY: {severity}
  - Predicted margin: {ctx.predicted_margin:.4f} MW (threshold: {SECURITY_THRESHOLD_MW} MW)
  - Reserve as % of load: {margin_pct:.2f}%
  - Min bus voltage: {np.min(ctx.bus_voltages):.4f} p.u.
  - {outage_str}

MECHANISM:
  {mechanism}

MICRO-GRID IMPACT:
  {mg_impact}

IMMEDIATE ACTIONS (next 30 minutes):
{chr(10).join(f'  {i+1}. {a}' for i, a in enumerate(immediate_actions))}

PREVENTIVE MEASURES (next 24 hours):
{chr(10).join(f'  {i+1}. {p}' for i, p in enumerate(preventive))}

ESTIMATED RECOVERY TIME:
  Outage (if not prevented):  2-4 hours distribution restoration
  Cyber investigation:        24-72 hours
  Secure ops restoration:     1-7 days (after patching communication layer)

[AI-generated analysis. Human expert verification required before
 implementing emergency procedures.]
"""


def build_attack_context(attack_result, pf_result, current_hour: int,
                          load_MW_t: np.ndarray, der_gen_MW_t: np.ndarray,
                          confidence: float = 0.90) -> AttackContext:
    T = len(attack_result.system_margin_true)
    ws = max(0, current_hour - 6)

    if attack_result.falsification_signal is not None:
        delta_t  = np.abs(attack_result.falsification_signal[current_hour])
        top_idx  = np.argsort(delta_t)[::-1][:10]
        top_buses = [int(i)+1 for i in top_idx if delta_t[i] > 1e-4]
        top_mags  = [float(delta_t[i]) for i in top_idx if delta_t[i] > 1e-4]
    else:
        top_buses, top_mags = [], []

    future = attack_result.system_margin_true[current_hour:]
    below  = np.where(future < SECURITY_THRESHOLD_MW)[0]
    hours_to_outage = int(below[0]) if len(below) > 0 else None
    predicted_margin = float(attack_result.system_margin_true[min(current_hour+2, T-1)])

    return AttackContext(
        scenario=attack_result.scenario, current_hour=current_hour,
        predicted_margin=predicted_margin,
        actual_margin=float(attack_result.system_margin_true[current_hour]),
        margin_history=attack_result.system_margin_true[ws:current_hour+1].tolist(),
        alarm_triggered=(predicted_margin < SECURITY_THRESHOLD_MW),
        hours_to_outage=hours_to_outage,
        top_anomaly_buses=top_buses[:5], anomaly_magnitudes=top_mags[:5],
        bus_voltages=pf_result.V_pu,
        total_load_MW=float(load_MW_t.sum()),
        total_gen_MW=float(attack_result.original_dispatch[current_hour].sum()),
        total_pv_MW=float(der_gen_MW_t.sum()),
        reserve_MW=float(load_MW_t.sum() * 0.05),
        falsification_signal=attack_result.falsification_signal,
        confidence=confidence,
        affected_microgrids=getattr(attack_result, "affected_microgrids", None) or [],
    )


if __name__ == "__main__":
    ctx = AttackContext(
        scenario="S1", current_hour=15, predicted_margin=0.03, actual_margin=0.08,
        margin_history=[0.21, 0.18, 0.15, 0.11, 0.08, 0.05],
        alarm_triggered=True, hours_to_outage=1,
        top_anomaly_buses=[13, 19, 16], anomaly_magnitudes=[0.045, 0.032, 0.028],
        bus_voltages=np.ones(69) * 0.97,
        total_load_MW=3.8, total_gen_MW=3.5, total_pv_MW=0.4, reserve_MW=0.019,
        confidence=0.94, affected_microgrids=["MG4"],
    )
    explainer = LLMExplainer(use_llm=False)
    print(explainer.explain(ctx))
