"""
LLM Explainability Module for FDI Attack Detection
====================================================
Wraps Mistral-7B-Instruct (recommended for Narval A100) to generate:
  1. Attack location identification
  2. Severity assessment
  3. Countermeasure recommendations
  4. Operator action plan

--- HOW TO DOWNLOAD ON NARVAL ---
  module load python/3.10.13 cuda/12.2
  source ~/ENV/bin/activate
  pip install huggingface_hub --no-index 2>/dev/null || pip install huggingface_hub
  huggingface-cli download mistralai/Mistral-7B-Instruct-v0.3 \
      --local-dir /home/gkianfar/scratch/Amin/llm_cache/Mistral-7B-Instruct-v0.3

  Alternative (smaller ~4GB): microsoft/phi-3-mini-4k-instruct
  Alternative (larger, more capable): meta-llama/Meta-Llama-3-8B-Instruct
"""

import numpy as np
from typing import Dict, List, Optional
from dataclasses import dataclass

from config import (LLM_MODEL_NAME, LLM_CACHE_DIR, LLM_MAX_TOKENS,
                    LLM_TEMPERATURE, SECURITY_THRESHOLD_MW,
                    ATTACK_START_H, ATTACK_END_H)


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
    total_pv_MW:        float
    reserve_MW:         float
    falsification_signal: Optional[np.ndarray] = None
    confidence:         float = 0.0


SYSTEM_PROMPT = """You are an expert power system security analyst specializing in 
cybersecurity of smart grids and distributed energy resources (DERs).
You analyze False Data Injection (FDI) attack alerts from an AI detection system 
monitoring an IEEE 69-bus distribution network with solar PV and energy storage.

Provide concise, technical, actionable analysis. Format as:
- ATTACK LOCATION: [specific buses/feeders/DERs]
- SEVERITY: [Critical/High/Medium] with brief justification
- MECHANISM: [2-3 sentences on how the attack works]
- IMMEDIATE ACTIONS (next 30 min): [numbered list]
- PREVENTIVE MEASURES (24h): [numbered list]
- ESTIMATED RECOVERY TIME: [if outage occurs]"""


def _build_prompt(ctx: AttackContext) -> str:
    v_min_bus = int(np.argmin(ctx.bus_voltages)) + 1
    v_min_val = float(np.min(ctx.bus_voltages))
    anomaly_str = ", ".join([
        f"Bus {b} ({m:+.3f} MW)"
        for b, m in zip(ctx.top_anomaly_buses[:5], ctx.anomaly_magnitudes[:5])
    ]) or "Localizing..."
    scenario_desc = {"S1": "generation dispatch falsification",
                     "S2": "load curtailment falsification"}.get(ctx.scenario, "unknown")
    urgency = (f"{ctx.hours_to_outage}h to predicted outage"
               if ctx.hours_to_outage else "outage timing uncertain")

    return f"""FDI ATTACK ALERT — Hour {ctx.current_hour:02d}:00
Status: {'⚠ ALARM' if ctx.alarm_triggered else '⚡ WARNING'} | {urgency}
Confidence: {ctx.confidence*100:.1f}% | Type: {scenario_desc}

SYSTEM STATE:
  Load={ctx.total_load_MW:.3f} MW | Gen={ctx.total_gen_MW:.3f} MW | PV={ctx.total_pv_MW:.3f} MW
  Reserve={ctx.reserve_MW:.3f} MW (threshold={SECURITY_THRESHOLD_MW} MW)
  Predicted margin (2h ahead)={ctx.predicted_margin:.3f} MW
  Min voltage={v_min_val:.4f} pu at Bus {v_min_bus}

MARGIN HISTORY (last {len(ctx.margin_history)}h): {[f"{m:.2f}" for m in ctx.margin_history]}
ANOMALOUS BUSES: {anomaly_str}

NETWORK: IEEE 69-bus radial, PV at buses 11,21,33,49,62, storage at 6,25,50
ATTACK WINDOW: Hours {ATTACK_START_H}-{ATTACK_END_H} (evening duck-curve ramp-up)

Provide expert attack analysis and countermeasures."""


class LLMExplainer:
    """
    LLM-based explainability for FDI attack detection.
    Falls back to rule-based explanation if model unavailable.
    """

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

        buses_str = ", ".join(map(str, ctx.top_anomaly_buses[:3])) or "Under investigation"
        deficit   = max(0, SECURITY_THRESHOLD_MW - ctx.predicted_margin)
        outage_str = (f"Predicted outage in {ctx.hours_to_outage}h."
                      if ctx.hours_to_outage else "Monitor closely next 2h.")

        if ctx.scenario == "S1":
            mechanism = (
                f"Generation dispatch falsification: DER outputs at buses {buses_str} "
                f"are being under-dispatched by up to {max(ctx.anomaly_magnitudes[:3], default=0):.3f} MW. "
                f"The EMS uplink is simultaneously spoofed, masking the deviation. "
                f"The attack accumulates during the evening solar ramp-down to exhaust the "
                f"{ctx.reserve_MW:.3f} MW system reserve."
            )
            immediate_actions = [
                f"Activate emergency ramp-up on all dispatchable generators (+{deficit*1.2:.2f} MW minimum)",
                f"Bypass automated dispatch for buses {buses_str} — switch to manual control",
                "Discharge all available energy storage at maximum rated power",
                f"Initiate load shedding if reserve drops below {SECURITY_THRESHOLD_MW*2:.2f} MW",
                "Cross-check generation PMU readings against EMS telemetry for discrepancies",
                "Alert transmission operator and request emergency import capacity",
            ]
        elif ctx.scenario == "S2":
            mechanism = (
                f"Load curtailment falsification: demand response signals at downstream buses "
                f"({buses_str}) are overridden, preventing load reduction of up to "
                f"{max(ctx.anomaly_magnitudes[:3], default=0):.3f} MW. "
                f"The attack exploits geographic correlation — adjacent buses show similar anomalies. "
                f"Effective system load exceeds scheduled values, depleting the reserve margin."
            )
            immediate_actions = [
                f"Manually activate load curtailment at buses {buses_str} (bypass comms)",
                f"Direct telephone contact with demand response customers for manual reduction",
                f"Increase generation by {deficit*1.2:.2f} MW from fast-start reserves",
                "Isolate the compromised sub-feeder communication segment",
                "Switch demand response control to out-of-band channel",
                "Monitor adjacent buses for correlated anomalies (geographic spread of attack)",
            ]
        else:
            mechanism = ("Attack type under investigation. Both generation dispatch and "
                         "load curtailment channels may be compromised simultaneously.")
            immediate_actions = [
                "Activate full emergency operations protocol",
                "Switch ALL DER dispatch to manual override immediately",
                "Increase generation reserve by 15% of current load",
                "Isolate all automated dispatch communication channels",
            ]

        preventive = [
            "Implement IEC 62351-7 cryptographic authentication on DER dispatch signals",
            "Deploy independent PMU-based cross-validation for EMS monitoring uplinks",
            "Install hardware security modules (HSM) at each DER communication interface",
            "Increase operating reserve to 10% during evening ramp (hours 14-20)",
            "Establish dedicated secure SCADA network segment for emergency dispatch",
            f"Retrain CNN detection model with attack patterns from this incident",
            "Conduct red-team penetration test on communication infrastructure",
        ]

        return f"""
╔═══════════════════════════════════════════════════════════════════════════╗
║     FDI ATTACK ANALYSIS REPORT — Hour {ctx.current_hour:02d}:00 — {severity:8s}         ║
╚═══════════════════════════════════════════════════════════════════════════╝

ATTACK LOCATION:
  Affected buses: {buses_str}
  Attack type: {'S1 – Generation dispatch falsification' if ctx.scenario=='S1' else 'S2 – Load curtailment falsification' if ctx.scenario=='S2' else 'Unknown'}
  Detection confidence: {ctx.confidence*100:.1f}%

SEVERITY: {severity}
  • Predicted margin: {ctx.predicted_margin:.4f} MW (threshold: {SECURITY_THRESHOLD_MW} MW)
  • Reserve as % of load: {margin_pct:.2f}%
  • Min bus voltage: {np.min(ctx.bus_voltages):.4f} p.u.
  • {outage_str}

MECHANISM:
  {mechanism}

IMMEDIATE ACTIONS (next 30 minutes):
{chr(10).join(f'  {i+1}. {a}' for i, a in enumerate(immediate_actions))}

PREVENTIVE MEASURES (next 24 hours):
{chr(10).join(f'  {i+1}. {p}' for i, p in enumerate(preventive))}

ESTIMATED RECOVERY TIME:
  Outage (if not prevented):  2–4 hours distribution restoration
  Cyber investigation:        24–72 hours
  Secure ops restoration:     1–7 days (after patching communication layer)

⚠ AI-generated analysis. Human expert verification required before 
  implementing emergency procedures.
"""


def build_attack_context(attack_result, pf_result, current_hour: int,
                          load_MW_t: np.ndarray, pv_MW_t: np.ndarray,
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
        total_pv_MW=float(pv_MW_t.sum()),
        reserve_MW=float(load_MW_t.sum() * 0.05),
        falsification_signal=attack_result.falsification_signal,
        confidence=confidence,
    )


if __name__ == "__main__":
    ctx = AttackContext(
        scenario="S1", current_hour=15, predicted_margin=0.3, actual_margin=0.8,
        margin_history=[2.1, 1.8, 1.5, 1.1, 0.8, 0.5],
        alarm_triggered=True, hours_to_outage=1,
        top_anomaly_buses=[11, 21, 33], anomaly_magnitudes=[0.45, 0.32, 0.28],
        bus_voltages=np.ones(69) * 0.97,
        total_load_MW=3.8, total_gen_MW=3.5, total_pv_MW=0.4, reserve_MW=0.19,
        confidence=0.94,
    )
    explainer = LLMExplainer(use_llm=False)
    print(explainer.explain(ctx))
