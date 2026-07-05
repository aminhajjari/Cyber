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
    # NEW: detector's own attack probability for THIS sample (0-1). When set,
    # the report is grounded in what the MODEL decided, not the ground truth.
    attack_probability: Optional[float] = None
    # NEW: whether top_anomaly_buses came from model saliency (True) or from
    # the ground-truth falsification vector (False, legacy fallback).
    buses_from_saliency: bool = False


SYSTEM_PROMPT = """You are an expert power system security analyst specializing in 
cybersecurity of smart grids, micro-grids, and distributed energy resources (DERs).
You analyze False Data Injection (FDI) attack alerts from an AI detection system 
monitoring an IEEE 69-bus distribution network that has been partitioned into 5
autonomous micro-grids (MG1-MG5), each containing a mix of wind turbines (WT),
photovoltaic (PV), and biomass (BM) generators.

Respond in TWO parts, in this exact order:

PART 1 — a single JSON object on its own line, with EXACTLY these keys:
  "attack_location_buses": [list of int bus numbers you believe are compromised],
  "affected_microgrids": [list of MG strings like "MG4"],
  "severity": one of "Critical", "High", "Medium",
  "mechanism_summary": one sentence,
  "confidence_0to1": float
Only include buses/MGs you can justify from the data given to you below — do not
guess buses that were not mentioned in the detector output.

PART 2 — the human-readable report, formatted as:
- ATTACK LOCATION: [specific buses/feeders/DERs/micro-grid(s)]
- SEVERITY: [Critical/High/Medium] with brief justification
- MECHANISM: [2-3 sentences on how the attack works]
- MICRO-GRID IMPACT: [can the affected MG(s) safely island, or do they depend
  on the main grid right now?]
- IMMEDIATE ACTIONS (next 30 min): [numbered list]
- PREVENTIVE MEASURES (24h): [numbered list]
- ESTIMATED RECOVERY TIME: [if outage occurs]"""


import json
import re


def parse_structured_report(raw_text: str) -> Optional[Dict]:
    """
    Pull the PART-1 JSON object out of the LLM's raw response. Returns None
    if no valid JSON object with the expected keys is found (e.g. the model
    ignored the instruction) so callers can fall back gracefully.
    """
    expected = {"attack_location_buses", "affected_microgrids", "severity",
                "mechanism_summary", "confidence_0to1"}
    # Try every {...} block in the text, first-to-last, keep the first that parses
    # and has the expected keys.
    for m in re.finditer(r"\{[^{}]*\}", raw_text, flags=re.DOTALL):
        try:
            obj = json.loads(m.group(0))
        except json.JSONDecodeError:
            continue
        if expected.issubset(obj.keys()):
            obj["attack_location_buses"] = [int(b) for b in obj["attack_location_buses"]]
            return obj
    return None


def grounding_score(structured: Optional[Dict], saliency_buses: List[int],
                     true_attacked_buses: Optional[List[int]] = None) -> Dict[str, float]:
    """
    Does the LLM's STATED attack location actually match what the detector's
    saliency map (and, offline, the ground truth) say? This is the piece that
    closes the loop between "the CNN attended to the right buses" (measured by
    localization_score/deletion_insertion_score on the saliency map) and
    "the LLM correctly reported that" (measured here on the LLM's own claims).
    A free-text report can sound confident and specific while silently
    hallucinating bus numbers the detector never flagged -- this catches that.

    - vs_saliency: overlap between LLM-claimed buses and the detector's own
      top saliency buses. Ground-truth-FREE, usable at deployment.
    - vs_ground_truth: overlap between LLM-claimed buses and the truly
      falsified buses. Only computable in simulation/evaluation.
    """
    def _jaccard(a: List[int], b: List[int]) -> float:
        A, B = set(a), set(b)
        if not A and not B:
            return float("nan")
        if not A or not B:
            return 0.0
        return len(A & B) / len(A | B)

    if structured is None:
        return {"parsed": False, "vs_saliency": float("nan"),
                "vs_ground_truth": float("nan")}

    claimed = structured.get("attack_location_buses", [])
    out = {"parsed": True, "vs_saliency": _jaccard(claimed, saliency_buses)}
    out["vs_ground_truth"] = (_jaccard(claimed, true_attacked_buses)
                              if true_attacked_buses is not None else float("nan"))
    return out


def _der_type_for_bus(bus_id: int) -> str:
    if bus_id in WT_BUSES: return "WT"
    if bus_id in PV_BUSES: return "PV"
    if bus_id in BM_BUSES: return "BM"
    return "load-only"


def _build_prompt(ctx: AttackContext) -> str:
    v_min_bus = int(np.argmin(ctx.bus_voltages)) + 1
    v_min_val = float(np.min(ctx.bus_voltages))

    bus_unit = "saliency" if ctx.buses_from_saliency else "MW"
    anomaly_str = ", ".join([
        f"Bus {b} [{_der_type_for_bus(b)}, {BUS_TO_MICROGRID.get(b,'?')}] ({m:.3f} {bus_unit})"
        for b, m in zip(ctx.top_anomaly_buses[:5], ctx.anomaly_magnitudes[:5])
    ]) or "Localizing..."
    bus_header = ("BUSES DRIVING THE DETECTOR'S DECISION (CNN saliency, normalized)"
                  if ctx.buses_from_saliency else "ANOMALOUS BUSES (ground-truth)")
    prob_line = (f"Detector attack probability: {ctx.attack_probability*100:.1f}%\n"
                 if ctx.attack_probability is not None else "")

    scenario_desc = {"S1": "generation dispatch falsification",
                     "S2": "load curtailment falsification"}.get(ctx.scenario, "unknown")
    urgency = (f"{ctx.hours_to_outage}h to predicted outage"
               if ctx.hours_to_outage else "outage timing uncertain")

    mg_str = ", ".join(ctx.affected_microgrids) if ctx.affected_microgrids else "Unknown"

    return f"""FDI ATTACK ALERT — Hour {ctx.current_hour:02d}:00
You are interpreting the decision of a trained CNN detector. The margin and the
flagged buses below are the MODEL'S OWN OUTPUTS (prediction + saliency), not
ground truth. Explain WHY the model raised this alarm and what it implies.

Status: {'[ALARM]' if ctx.alarm_triggered else '[WARNING]'} | {urgency}
Confidence: {ctx.confidence*100:.1f}% | Type: {scenario_desc}
AFFECTED MICRO-GRID(S) (from model-attributed buses): {mg_str}

DETECTOR OUTPUT:
  {prob_line}  Predicted system margin (2h ahead) = {ctx.predicted_margin:.4f} MW
  Security threshold = {SECURITY_THRESHOLD_MW} MW

SYSTEM STATE:
  Load={ctx.total_load_MW:.4f} MW | Gen={ctx.total_gen_MW:.4f} MW | DER(WT+PV+BM)={ctx.total_pv_MW:.4f} MW
  Reserve={ctx.reserve_MW:.4f} MW
  Min voltage={v_min_val:.4f} pu at Bus {v_min_bus}

MARGIN HISTORY (last {len(ctx.margin_history)}h): {[f"{m:.3f}" for m in ctx.margin_history]}
{bus_header}: {anomaly_str}

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

        json_header = json.dumps({
            "attack_location_buses": ctx.top_anomaly_buses,
            "affected_microgrids":   mg_list,
            "severity":              severity.capitalize(),
            "mechanism_summary":     mechanism.split(".")[0].strip() + ".",
            "confidence_0to1":       round(ctx.confidence, 3),
        })

        return f"""{json_header}
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
                          model_pred_margin: float = None,   # NEW: cnn.predict_margin
                          attack_prob: float = None,         # NEW: cnn.predict_proba
                          saliency_buses: List[int] = None,  # NEW: 1-idx, from bus_saliency
                          saliency_mags:  List[float] = None,# NEW: normalized saliency
                          confidence: float = 0.90) -> AttackContext:
    """
    Build the LLM context. When `model_pred_margin` / `attack_prob` /
    `saliency_buses` are supplied, the report is grounded in the DETECTOR'S OWN
    decision (this is the interpretability path). If they are omitted, it falls
    back to the legacy ground-truth behavior for backward compatibility.
    """
    T = len(attack_result.system_margin_true)
    ws = max(0, current_hour - 6)

    # ── Buses: prefer model saliency; else legacy ground-truth falsification ──
    if saliency_buses is not None:
        top_buses = [int(b) for b in saliency_buses[:5]]
        top_mags  = [float(m) for m in (saliency_mags or [0.0]*len(top_buses))[:5]]
        buses_from_saliency = True
    elif attack_result.falsification_signal is not None:
        delta_t  = np.abs(attack_result.falsification_signal[current_hour])
        top_idx  = np.argsort(delta_t)[::-1][:10]
        top_buses = [int(i)+1 for i in top_idx if delta_t[i] > 1e-4][:5]
        top_mags  = [float(delta_t[i]) for i in top_idx if delta_t[i] > 1e-4][:5]
        buses_from_saliency = False
    else:
        top_buses, top_mags, buses_from_saliency = [], [], False

    # ── Micro-grids: derive from the (model-attributed) top buses ────────────
    affected_mgs = sorted(set(
        BUS_TO_MICROGRID.get(b, "UNASSIGNED") for b in top_buses
    )) if top_buses else (getattr(attack_result, "affected_microgrids", None) or [])

    # ── Margin: prefer the model's prediction; else ground truth ─────────────
    if model_pred_margin is not None:
        predicted_margin = float(model_pred_margin)
    else:
        predicted_margin = float(attack_result.system_margin_true[min(current_hour+2, T-1)])

    future = attack_result.system_margin_true[current_hour:]
    below  = np.where(future < SECURITY_THRESHOLD_MW)[0]
    hours_to_outage = int(below[0]) if len(below) > 0 else None

    if attack_prob is not None:
        alarm = attack_prob > 0.5
        conf  = float(attack_prob)
    else:
        alarm = predicted_margin < SECURITY_THRESHOLD_MW
        conf  = confidence

    return AttackContext(
        scenario=attack_result.scenario, current_hour=current_hour,
        predicted_margin=predicted_margin,
        actual_margin=float(attack_result.system_margin_true[current_hour]),
        margin_history=attack_result.system_margin_true[ws:current_hour+1].tolist(),
        alarm_triggered=alarm,
        hours_to_outage=hours_to_outage,
        top_anomaly_buses=top_buses, anomaly_magnitudes=top_mags,
        bus_voltages=pf_result.V_pu,
        total_load_MW=float(load_MW_t.sum()),
        total_gen_MW=float(attack_result.original_dispatch[current_hour].sum()),
        total_pv_MW=float(der_gen_MW_t.sum()),
        reserve_MW=float(load_MW_t.sum() * 0.05),
        falsification_signal=attack_result.falsification_signal,
        confidence=conf,
        affected_microgrids=affected_mgs,
        attack_probability=attack_prob,
        buses_from_saliency=buses_from_saliency,
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
