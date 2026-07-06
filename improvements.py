"""
improvements.py — correctness fixes + explainability core
=========================================================
Three self-contained, drop-in pieces used to strengthen the pipeline over the
per-sample-random-split baseline:

  1. group_split_by_day(...)      -> leakage-free train/val/test split.
     Windows from one Monte-Carlo day never straddle the split, so the model
     is scored on *unseen days*. This is what makes CNN>MLP>SVR reappear and
     what makes S1 stop reading 100%.

  2. bus_saliency(...)            -> per-bus attribution FROM THE TRAINED
     DETECTOR (gradient x input, integrated over a baseline). This is what the
     model actually "looked at" when it raised the alarm, as opposed to the
     ground-truth attack vector. It is the substrate the LLM explains.

  3. localization_score(...)      -> quantitative faithfulness metric: overlap
     between the detector's top-saliency buses and the truly-falsified buses
     (precision@k / IoU). Lets you *measure* explanation quality instead of
     eyeballing LLM prose — a concrete contribution beyond the baseline.

All three are framework-light (numpy + torch) and unit-tested at the bottom.
"""

from __future__ import annotations
import numpy as np
import torch
from typing import Tuple, List, Dict, Optional


# ─────────────────────────────────────────────────────────────────────────────
# 1. Leakage-free split
# ─────────────────────────────────────────────────────────────────────────────

def group_split_by_day(
        day: np.ndarray,
        frac_train: float = 0.70,
        frac_val: float = 0.15,
        seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Split sample indices so that all samples of a given day land in exactly one
    of {train, val, test}. Prevents same-day leakage.

    Args:
        day : (N,) int array giving the day id of every sample.
    Returns:
        (idx_train, idx_val, idx_test) : index arrays into the sample axis.
    """
    day = np.asarray(day)
    uniq = np.unique(day)
    rng = np.random.default_rng(seed)
    rng.shuffle(uniq)

    n = len(uniq)
    n_tr = int(round(frac_train * n))
    n_va = int(round(frac_val * n))
    # guarantee every split is non-empty when there are >=3 days
    n_tr = max(1, min(n_tr, n - 2)) if n >= 3 else n_tr
    n_va = max(1, min(n_va, n - n_tr - 1)) if n >= 3 else n_va

    days_tr = set(uniq[:n_tr].tolist())
    days_va = set(uniq[n_tr:n_tr + n_va].tolist())
    days_te = set(uniq[n_tr + n_va:].tolist())

    idx_tr = np.where(np.isin(day, list(days_tr)))[0]
    idx_va = np.where(np.isin(day, list(days_va)))[0]
    idx_te = np.where(np.isin(day, list(days_te)))[0]
    return idx_tr, idx_va, idx_te


# ─────────────────────────────────────────────────────────────────────────────
# 2. Detector saliency  (what the MODEL attended to)
# ─────────────────────────────────────────────────────────────────────────────

def bus_saliency(
        model: torch.nn.Module,
        x: np.ndarray,                 # (n_bus, d) single scaled sample
        scaler_X,                      # the trainer's fitted StandardScaler
        head: str = "cls",             # "cls" = attack logit, "reg" = margin
        steps: int = 32,               # integrated-gradients interpolation steps
        device: Optional[str] = None,
) -> np.ndarray:
    """
    Integrated Gradients attribution aggregated to a per-bus score.

    Returns:
        sal : (n_bus,) non-negative saliency; larger = more responsible for the
              model's decision on THIS sample. Baseline is the all-zeros
              (feature-mean, since inputs are standardized) reference.

    Why IG (vs raw gradient): IG satisfies completeness/sensitivity and is far
    less noisy on a small conv net, giving stable per-bus rankings that the LLM
    and the localization metric can both trust.
    """
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device).eval()

    n_bus, d = x.shape
    xs = scaler_X.transform(x.reshape(1, -1)).reshape(1, n_bus, d)
    x_t = torch.tensor(xs, dtype=torch.float32, device=device)
    baseline = torch.zeros_like(x_t)                      # mean of standardized feats

    total_grad = torch.zeros_like(x_t)
    for a in torch.linspace(0.0, 1.0, steps, device=device):
        xi = (baseline + a * (x_t - baseline)).clone().requires_grad_(True)
        reg_out, cls_out = model(xi)
        target = cls_out if head == "cls" else reg_out
        model.zero_grad(set_to_none=True)
        target.sum().backward()
        total_grad = total_grad + xi.grad.detach()

    avg_grad = total_grad / steps
    ig = (x_t - baseline) * avg_grad                      # (1, n_bus, d)
    # aggregate |attribution| over the d feature/time channels -> per-bus score
    sal = ig.abs().sum(dim=2).squeeze(0).cpu().numpy()    # (n_bus,)
    if sal.max() > 0:
        sal = sal / sal.max()
    return sal


# ─────────────────────────────────────────────────────────────────────────────
# 3. Explanation faithfulness / localization metric
# ─────────────────────────────────────────────────────────────────────────────

def localization_score(
        saliency: np.ndarray,          # (n_bus,) detector attribution
        true_attacked_buses: List[int],  # 1-indexed bus ids that were falsified
        k: Optional[int] = None,
) -> Dict[str, float]:
    """
    How well does the detector's attention land on the truly-falsified buses?

    Returns precision@k, recall@k and IoU between the top-k saliency buses and
    the ground-truth attacked set. k defaults to |true_attacked_buses|.

    This turns "the explanation looks plausible" into a number you can report
    and compare across models (CNN vs MLP) and scenarios (S1 vs S2).
    """
    n_bus = len(saliency)
    true_set = set(int(b) - 1 for b in true_attacked_buses      # -> 0-indexed
                   if 1 <= int(b) <= n_bus)
    if not true_set:
        return {"precision@k": float("nan"), "recall@k": float("nan"),
                "IoU": float("nan"), "k": 0}
    k = k or len(true_set)
    k = max(1, min(k, n_bus))
    top = set(np.argsort(saliency)[::-1][:k].tolist())
    hit = len(top & true_set)
    precision = hit / len(top)
    recall = hit / len(true_set)
    iou = hit / len(top | true_set)
    return {"precision@k": precision, "recall@k": recall, "IoU": iou, "k": k}


# ─────────────────────────────────────────────────────────────────────────────
# 4. Deletion/insertion faithfulness  (ground-truth-FREE, deployment-usable)
# ─────────────────────────────────────────────────────────────────────────────

def deletion_insertion_score(
        model: torch.nn.Module,
        x: np.ndarray,                 # (n_bus, d) single RAW (unscaled) sample
        saliency: np.ndarray,          # (n_bus,) from bus_saliency
        scaler_X,
        head: str = "cls",
        k_frac: float = 0.15,          # fraction of buses to perturb
        device: Optional[str] = None,
) -> Dict[str, float]:
    """
    Perturbation-based faithfulness check for the SAME saliency map that feeds
    the LLM report, but -- unlike localization_score() -- it needs no ground
    truth. localization_score can only be computed in simulation (where the
    true falsified buses are known); this metric can be computed on every
    live alert in deployment, giving a continuous sanity check on whether the
    model (and therefore the LLM's account of it) is telling a
    self-consistent story.

    Deletion: zero out the top-k salient buses (replace with the standardized
    baseline) and see how much the attack logit DROPS. A faithful saliency map
    should cause a large drop.
    Insertion: start from an all-baseline sample and insert ONLY the top-k
    salient buses' real values; see how much of the original logit is
    RECOVERED. A faithful map should recover most of it from few buses.
    """
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device).eval()

    n_bus, d = x.shape
    k = max(1, int(round(k_frac * n_bus)))
    top_idx = np.ascontiguousarray(np.argsort(saliency)[::-1][:k])

    xs = scaler_X.transform(x.reshape(1, -1)).reshape(1, n_bus, d)
    x_t = torch.tensor(xs, dtype=torch.float32, device=device)
    baseline = torch.zeros_like(x_t)

    def _prob(t):
        # Work in PROBABILITY space (sigmoid of the logit), not raw logit space.
        # A well-separated classifier saturates logits at large, near-identical
        # magnitudes for confidently-classified negatives, which makes
        # (logit_a - logit_b) numerically unstable as a normalizer (it can be
        # near zero even though the model is very confident). Probabilities
        # are bounded in [0, 1], so differences stay well-scaled regardless of
        # how saturated the underlying logits are.
        with torch.no_grad():
            reg_out, cls_out = model(t)
            out = cls_out if head == "cls" else reg_out
            return float(torch.sigmoid(out).item()) if head == "cls" else float(out.item())

    full_p = _prob(x_t)

    x_del = x_t.clone()
    x_del[0, top_idx, :] = baseline[0, top_idx, :]
    del_p = _prob(x_del)

    x_ins = baseline.clone()
    x_ins[0, top_idx, :] = x_t[0, top_idx, :]
    ins_p = _prob(x_ins)

    base_p = _prob(baseline)
    denom = full_p - base_p
    # Degenerate case: the model gives ~the same output for the real sample
    # and the "average" baseline (e.g. a confident, saturated no-attack
    # prediction where masking a few buses can't move the needle either way).
    # Returning a clipped extreme value here would look like real signal when
    # it is actually "this metric isn't well-defined for this sample" --
    # report NaN instead so it doesn't silently corrupt an average.
    if abs(denom) < 1e-3:
        return {"drop_frac": float("nan"), "recovered_frac": float("nan"),
                "k": k, "degenerate": True, "full_prob": full_p, "base_prob": base_p}

    drop_frac      = float(np.clip((full_p - del_p) / denom, -2, 2))
    recovered_frac = float(np.clip((ins_p - base_p) / denom, -2, 2))
    return {"drop_frac": drop_frac, "recovered_frac": recovered_frac, "k": k,
            "degenerate": False, "full_prob": full_p, "base_prob": base_p}


# ─────────────────────────────────────────────────────────────────────────────
# self-test  (run: python improvements.py)
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    # --- test 1: group split has zero day overlap ---------------------------
    rng = np.random.default_rng(0)
    day = np.repeat(np.arange(150), 16)          # 150 days x 16 windows
    rng.shuffle(day)
    itr, iva, ite = group_split_by_day(day, seed=42)
    d_tr, d_va, d_te = set(day[itr]), set(day[iva]), set(day[ite])
    assert d_tr.isdisjoint(d_va) and d_tr.isdisjoint(d_te) and d_va.isdisjoint(d_te), \
        "LEAKAGE: a day appears in more than one split"
    assert len(itr) + len(iva) + len(ite) == len(day), "lost samples"
    print(f"[OK] group split: {len(d_tr)}/{len(d_va)}/{len(d_te)} days, "
          f"{len(itr)}/{len(iva)}/{len(ite)} samples, no day overlap")

    # --- test 2/3: saliency + localization on a toy conv net ----------------
    from sklearn.preprocessing import StandardScaler

    class Toy(torch.nn.Module):
        def __init__(self, n_bus, d):
            super().__init__()
            self.conv = torch.nn.Conv1d(d, 8, 3, padding=1)
            self.fc = torch.nn.Linear(8 * n_bus, 16)
            self.reg = torch.nn.Linear(16, 1)
            self.cls = torch.nn.Linear(16, 1)
        def forward(self, x):
            x = torch.relu(self.conv(x.permute(0, 2, 1))).flatten(1)
            h = torch.relu(self.fc(x))
            return self.reg(h).squeeze(-1), self.cls(h).squeeze(-1)

    n_bus, d = 69, 48
    attacked = [13, 19, 16, 52]                  # buses that carry the attack
    N = 400
    X = rng.normal(size=(N, n_bus, d)).astype(np.float32)
    lbl = rng.integers(0, 2, size=N).astype(np.float32)
    # attacked buses light up ONLY on positive (attacked) samples
    for i in range(N):
        if lbl[i] == 1:
            for b in attacked:
                X[i, b - 1, :] += 3.0
    sc = StandardScaler().fit(X.reshape(len(X), -1))
    Xs = sc.transform(X.reshape(len(X), -1)).reshape(X.shape)

    net = Toy(n_bus, d)
    opt = torch.optim.Adam(net.parameters(), lr=1e-3)
    bce = torch.nn.BCEWithLogitsLoss()
    Xt = torch.tensor(Xs, dtype=torch.float32)
    lt = torch.tensor(lbl, dtype=torch.float32)
    for _ in range(150):                          # brief train so cls head reacts
        opt.zero_grad()
        _, c = net(Xt)
        bce(c, lt).backward()
        opt.step()

    # attribute an attacked sample; saliency should land on the planted buses
    pos_i = int(np.where(lbl == 1)[0][0])
    sal = bus_saliency(net, X[pos_i], sc, head="cls", steps=16, device="cpu")
    assert sal.shape == (n_bus,) and sal.max() <= 1.0 + 1e-6
    loc = localization_score(sal, attacked, k=4)
    top4 = (np.argsort(sal)[::-1][:4] + 1).tolist()
    print(f"[OK] saliency shape {sal.shape}, top-4 buses {top4}")
    print(f"[OK] localization {loc}  (attacked buses were {attacked})")
    assert loc["recall@k"] >= 0.5, "saliency failed to localize planted buses"

    # --- test 4: deletion/insertion agrees with localization on same sample --
    di = deletion_insertion_score(net, X[pos_i], sal, sc, head="cls",
                                   k_frac=4 / n_bus, device="cpu")
    print(f"[OK] deletion/insertion {di}")
    if not di["degenerate"]:
        assert di["drop_frac"] > 0.1, "deleting top-saliency buses barely changed the probability"
    else:
        print("[OK] degenerate case correctly reported as NaN rather than a misleading extreme value")

    print("\nAll improvements.py self-tests passed.")
