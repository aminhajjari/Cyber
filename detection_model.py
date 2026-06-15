"""
CNN-Based FDI Attack Detection Model
Architecture matches Table I of Wu et al., IEEE Trans. Smart Grid, 2025

Input:  multi-interval dispatch signals + network status (voltage mag + angle)
        shaped as (n_bus, d) where d = features × monitoring_window
Output: system margin regression → security classification

Also includes MLP and SVR baselines for comparison (Table II).
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
from typing import Tuple, Dict, List, Optional
import os

from config import CNN_CONFIG, T_MONITORING, T_PRED_AHEAD, SECURITY_THRESHOLD_MW


# ─────────────────────────────────────────────────────────────────────────────
# Feature Tensor Construction  (eq. 3 in paper)
# ─────────────────────────────────────────────────────────────────────────────

def build_input_tensor(
        gen_dispatch_hat:  np.ndarray,   # (T, n_bus) predicted dispatch
        gen_dispatch_meas: np.ndarray,   # (T, n_bus) actual measurement
        curtail_hat:       np.ndarray,   # (T, n_bus)
        curtail_meas:      np.ndarray,
        stor_hat:          np.ndarray,
        stor_meas:         np.ndarray,
        V_mag:             np.ndarray,   # (T, n_bus) voltage magnitudes
        theta:             np.ndarray,   # (T, n_bus) phase angles
        t_pred:            int,          # current time step (T_pred)
        T_m:               int = T_MONITORING,
        feature_set:       str = "full"  # "PV" | "PVtheta" | "full"
) -> np.ndarray:
    """
    Build input tensor x^i for one observation (eq. 3a-3b).
    Monitoring window: [t_pred - T_m - T_pred_ahead, t_pred - T_pred_ahead].

    Returns: (n_bus, d) array
      where d = n_features × T_m
    """
    n_bus = gen_dispatch_hat.shape[1]
    T_m_start = max(0, t_pred - T_PRED_AHEAD - T_m)
    T_m_end   = max(0, t_pred - T_PRED_AHEAD)

    if T_m_start >= T_m_end:
        # Not enough history yet
        return None

    features_per_bus = []
    for n in range(n_bus):
        feat = []
        # Gen dispatch (predicted + actual)
        feat.append(gen_dispatch_hat [T_m_start:T_m_end, n])
        feat.append(gen_dispatch_meas[T_m_start:T_m_end, n])
        # Load curtailment
        feat.append(curtail_hat [T_m_start:T_m_end, n])
        feat.append(curtail_meas[T_m_start:T_m_end, n])
        # Storage
        feat.append(stor_hat [T_m_start:T_m_end, n])
        feat.append(stor_meas[T_m_start:T_m_end, n])
        # Voltage magnitude
        if feature_set in ("PVtheta", "full", "PV"):
            feat.append(V_mag[T_m_start:T_m_end, n])
        # Phase angle
        if feature_set in ("PVtheta", "full"):
            feat.append(theta[T_m_start:T_m_end, n])

        features_per_bus.append(np.concatenate(feat))  # (d,)

    return np.stack(features_per_bus)  # (n_bus, d)


def build_dataset(attack_results:  list,
                  normal_results:  list,
                  pf_results_atk:  list,
                  pf_results_norm: list,
                  T_m: int = T_MONITORING,
                  feature_set: str = "full"
                  ) -> Tuple[np.ndarray, np.ndarray]:
    """
    Build (X, y) dataset from attack and normal simulation results.

    X : (N_samples, n_bus, d)   — input tensor
    y : (N_samples,)            — system margin at T_pred (regression target)
    """
    X_list, y_list = [], []

    def _add_samples(results, pf_results, is_attack: bool):
        for idx, (res, pf_day) in enumerate(zip(results, pf_results)):
            T = len(res.system_margin_true)
            for t_pred in range(T_m + T_PRED_AHEAD, T):
                x = build_input_tensor(
                    gen_dispatch_hat  = res.original_dispatch,
                    gen_dispatch_meas = (res.falsified_dispatch
                                         if is_attack else res.original_dispatch),
                    curtail_hat  = np.zeros_like(res.original_dispatch),
                    curtail_meas = np.zeros_like(res.original_dispatch),
                    stor_hat     = np.zeros_like(res.original_dispatch),
                    stor_meas    = np.zeros_like(res.original_dispatch),
                    V_mag        = pf_day["V_mag"],
                    theta        = pf_day["theta"],
                    t_pred       = t_pred,
                    T_m          = T_m,
                    feature_set  = feature_set,
                )
                if x is None:
                    continue
                X_list.append(x)
                y_list.append(res.system_margin_true[t_pred])

    _add_samples(attack_results, pf_results_atk,  is_attack=True)
    _add_samples(normal_results, pf_results_norm,  is_attack=False)

    X = np.stack(X_list).astype(np.float32)
    y = np.array(y_list, dtype=np.float32)
    print(f"[Dataset] Built {X.shape[0]} samples, X shape: {X.shape}")
    return X, y


# ─────────────────────────────────────────────────────────────────────────────
# CNN Model  (Table I — matching paper architecture)
# ─────────────────────────────────────────────────────────────────────────────

class FDI_CNN(nn.Module):
    """
    1D CNN for FDI attack detection and system margin regression.
    Architecture (Table I):
      Conv1D(128 filters, kernel=3) → MaxPool1D → Flatten → FC(128) → FC(64) → FC(1)
    """

    def __init__(self, n_bus: int, d_features: int,
                 cfg: dict = CNN_CONFIG):
        super().__init__()
        self.n_bus      = n_bus
        self.d_features = d_features

        # Conv1D along the bus dimension (spatial dependencies)
        self.conv_block = nn.Sequential(
            nn.Conv1d(in_channels=d_features,
                      out_channels=cfg["conv1d_filters"],
                      kernel_size=cfg["conv1d_kernel"],
                      padding=cfg["conv1d_kernel"]//2),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=max(1, cfg["pool_size"])),
            nn.Dropout(cfg["dropout"]),
        )

        # Compute flattened size after conv/pool
        with torch.no_grad():
            dummy = torch.zeros(1, d_features, n_bus)
            conv_out = self.conv_block(dummy)
            flat_size = conv_out.numel()

        # Fully connected regression head
        fc_layers = []
        in_dim = flat_size
        for out_dim in cfg["fc_layers"]:
            fc_layers += [nn.Linear(in_dim, out_dim), nn.ReLU(),
                          nn.Dropout(cfg["dropout"])]
            in_dim = out_dim
        fc_layers.append(nn.Linear(in_dim, 1))   # regression output
        self.fc = nn.Sequential(*fc_layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (batch, n_bus, d_features)
        Returns: (batch,) system margin prediction
        """
        # Conv1D expects (batch, channels, length) → channels = d_features, length = n_bus
        x = x.permute(0, 2, 1)         # (batch, d_features, n_bus)
        x = self.conv_block(x)
        x = x.flatten(1)
        x = self.fc(x)
        return x.squeeze(-1)


# ─────────────────────────────────────────────────────────────────────────────
# MLP Baseline  (Table I)
# ─────────────────────────────────────────────────────────────────────────────

class FDI_MLP(nn.Module):
    """Two fully connected layers (128 + 64 neurons) — Table I baseline."""

    def __init__(self, input_dim: int, cfg: dict = CNN_CONFIG):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128), nn.ReLU(), nn.Dropout(cfg["dropout"]),
            nn.Linear(128, 64),        nn.ReLU(), nn.Dropout(cfg["dropout"]),
            nn.Linear(64, 1)
        )

    def forward(self, x):
        return self.net(x.flatten(1)).squeeze(-1)


# ─────────────────────────────────────────────────────────────────────────────
# Training & Evaluation
# ─────────────────────────────────────────────────────────────────────────────

class DetectionModelTrainer:

    def __init__(self, model_type: str = "CNN",
                 n_bus: int = 69,
                 d_features: int = None,
                 cfg: dict = CNN_CONFIG,
                 device: str = None):

        self.model_type = model_type
        self.cfg        = cfg
        self.n_bus      = n_bus
        self.device     = device or ("cuda" if torch.cuda.is_available() else "cpu")
        print(f"[CNN] Using device: {self.device}")

        if d_features is not None:
            self._init_model(d_features)

    def _init_model(self, d_features: int):
        self.d_features = d_features
        if self.model_type == "CNN":
            self.model = FDI_CNN(self.n_bus, d_features, self.cfg).to(self.device)
        elif self.model_type == "MLP":
            self.model = FDI_MLP(self.n_bus * d_features, self.cfg).to(self.device)
        else:
            raise ValueError(f"Unknown model_type: {self.model_type}")

        self.optimizer = optim.Adam(self.model.parameters(),
                                     lr=self.cfg["learning_rate"])
        self.criterion = nn.MSELoss()
        print(f"[{self.model_type}] Parameters: "
              f"{sum(p.numel() for p in self.model.parameters()):,}")

    def fit(self, X_train: np.ndarray, y_train: np.ndarray,
            X_val: np.ndarray,   y_val: np.ndarray) -> Dict:
        """
        Train the detection model.
        X: (N, n_bus, d_features)  y: (N,)
        Returns training history.
        """
        if not hasattr(self, 'model'):
            d_features = X_train.shape[2]
            self._init_model(d_features)

        # Normalize features per-feature (eq. 3 preprocessing)
        self.scaler_X = StandardScaler()
        X_flat = X_train.reshape(len(X_train), -1)
        self.scaler_X.fit(X_flat)
        X_train_n = self.scaler_X.transform(X_flat).reshape(X_train.shape)
        X_val_n   = self.scaler_X.transform(
            X_val.reshape(len(X_val), -1)).reshape(X_val.shape)

        self.scaler_y = StandardScaler()
        y_train_n = self.scaler_y.fit_transform(y_train.reshape(-1,1)).ravel()
        y_val_n   = self.scaler_y.transform(y_val.reshape(-1,1)).ravel()

        # Build DataLoaders
        ds_train = TensorDataset(torch.tensor(X_train_n, dtype=torch.float32),
                                  torch.tensor(y_train_n, dtype=torch.float32))
        ds_val   = TensorDataset(torch.tensor(X_val_n, dtype=torch.float32),
                                  torch.tensor(y_val_n, dtype=torch.float32))
        dl_train = DataLoader(ds_train, batch_size=self.cfg["batch_size"], shuffle=True)
        dl_val   = DataLoader(ds_val,   batch_size=self.cfg["batch_size"])

        history = {"train_mse": [], "val_mse": []}
        best_val_mse  = float("inf")
        best_state    = None

        for epoch in range(self.cfg["epochs"]):
            self.model.train()
            train_loss = 0.0
            for X_b, y_b in dl_train:
                X_b, y_b = X_b.to(self.device), y_b.to(self.device)
                self.optimizer.zero_grad()
                pred = self.model(X_b)
                loss = self.criterion(pred, y_b)
                loss.backward()
                self.optimizer.step()
                train_loss += loss.item() * len(y_b)

            train_mse = train_loss / len(ds_train)

            self.model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for X_b, y_b in dl_val:
                    X_b, y_b = X_b.to(self.device), y_b.to(self.device)
                    pred = self.model(X_b)
                    val_loss += self.criterion(pred, y_b).item() * len(y_b)
            val_mse = val_loss / len(ds_val)

            history["train_mse"].append(train_mse)
            history["val_mse"].append(val_mse)

            if val_mse < best_val_mse:
                best_val_mse = val_mse
                best_state   = {k: v.cpu().clone() for k, v in
                                 self.model.state_dict().items()}

            if (epoch + 1) % 100 == 0:
                print(f"[{self.model_type}] Epoch {epoch+1:4d} | "
                      f"Train MSE: {train_mse:.4f} | Val MSE: {val_mse:.4f}")

        # Restore best model
        self.model.load_state_dict(best_state)
        print(f"[{self.model_type}] Training done. Best val MSE: {best_val_mse:.4f}")
        return history

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict system margin (original scale, MW)."""
        X_flat = X.reshape(len(X), -1)
        X_n    = self.scaler_X.transform(X_flat).reshape(X.shape)
        X_t    = torch.tensor(X_n, dtype=torch.float32).to(self.device)
        self.model.eval()
        with torch.no_grad():
            pred_n = self.model(X_t).cpu().numpy()
        return self.scaler_y.inverse_transform(pred_n.reshape(-1,1)).ravel()

    def evaluate(self, X: np.ndarray, y_true: np.ndarray,
                 label: str = "test") -> Dict:
        """
        Full evaluation: regression MSE + classification metrics (Table II).
        """
        y_pred = self.predict(X)

        # --- Regression ---
        mse_overall = float(mean_squared_error(y_true, y_pred))

        # Split attack vs normal (assume balanced dataset: first half attack)
        n_half = len(y_true) // 2
        mse_attack = float(mean_squared_error(y_true[:n_half], y_pred[:n_half]))
        mse_normal = float(mean_squared_error(y_true[n_half:], y_pred[n_half:]))

        # --- Classification (security alarm) ---
        # True label: margin < threshold → 1 (under attack / unsafe)
        y_true_cls = (y_true < SECURITY_THRESHOLD_MW).astype(int)
        y_pred_cls = (y_pred < SECURITY_THRESHOLD_MW).astype(int)

        TP = int(((y_true_cls == 1) & (y_pred_cls == 1)).sum())
        TN = int(((y_true_cls == 0) & (y_pred_cls == 0)).sum())
        FP = int(((y_true_cls == 0) & (y_pred_cls == 1)).sum())
        FN = int(((y_true_cls == 1) & (y_pred_cls == 0)).sum())

        accuracy  = (TP + TN) / max(TP + TN + FP + FN, 1) * 100
        precision = TP / max(TP + FP, 1) * 100
        tpr       = TP / max(TP + FN, 1) * 100
        fpr       = FP / max(TN + FP, 1) * 100

        metrics = {
            "label":      label,
            "MSE_overall": mse_overall,
            "MSE_attack":  mse_attack,
            "MSE_normal":  mse_normal,
            "Accuracy":    accuracy,
            "Precision":   precision,
            "TPR":         tpr,
            "FPR":         fpr,
            "TP": TP, "TN": TN, "FP": FP, "FN": FN,
        }

        print(f"\n[Eval/{label}] MSE_overall={mse_overall:.4f} | "
              f"Acc={accuracy:.2f}% | Prec={precision:.2f}% | "
              f"TPR={tpr:.2f}% | FPR={fpr:.2f}%")
        return metrics

    def save(self, path: str):
        import pickle
        state = {
            "model_state": self.model.state_dict(),
            "model_type":  self.model_type,
            "n_bus":       self.n_bus,
            "d_features":  self.d_features,
            "cfg":         self.cfg,
            "scaler_X":    self.scaler_X,
            "scaler_y":    self.scaler_y,
        }
        with open(path, "wb") as f:
            pickle.dump(state, f)
        print(f"[{self.model_type}] Model saved to {path}")

    @classmethod
    def load(cls, path: str, device: str = None) -> "DetectionModelTrainer":
        import pickle
        with open(path, "rb") as f:
            state = pickle.load(f)
        trainer = cls(model_type=state["model_type"],
                      n_bus=state["n_bus"],
                      d_features=state["d_features"],
                      cfg=state["cfg"],
                      device=device)
        trainer.model.load_state_dict(state["model_state"])
        trainer.scaler_X = state["scaler_X"]
        trainer.scaler_y = state["scaler_y"]
        return trainer


class SVRDetector:
    """Kernel SVR baseline (RBF kernel, Table I)."""

    def __init__(self):
        self.svr      = SVR(kernel="rbf", C=10.0, epsilon=0.1)
        self.scaler_X = StandardScaler()
        self.scaler_y = StandardScaler()

    def fit(self, X_train, y_train):
        X_flat = X_train.reshape(len(X_train), -1)
        X_n    = self.scaler_X.fit_transform(X_flat)
        y_n    = self.scaler_y.fit_transform(y_train.reshape(-1,1)).ravel()
        print("[SVR] Fitting kernel SVR...")
        self.svr.fit(X_n, y_n)

    def predict(self, X):
        X_n = self.scaler_X.transform(X.reshape(len(X),-1))
        return self.scaler_y.inverse_transform(
            self.svr.predict(X_n).reshape(-1,1)).ravel()

    def evaluate(self, X, y_true, label="test") -> Dict:
        y_pred = self.predict(X)
        mse    = float(mean_squared_error(y_true, y_pred))
        y_tc   = (y_true < SECURITY_THRESHOLD_MW).astype(int)
        y_pc   = (y_pred < SECURITY_THRESHOLD_MW).astype(int)
        TP = int(((y_tc==1)&(y_pc==1)).sum())
        TN = int(((y_tc==0)&(y_pc==0)).sum())
        FP = int(((y_tc==0)&(y_pc==1)).sum())
        FN = int(((y_tc==1)&(y_pc==0)).sum())
        acc  = (TP+TN)/max(TP+TN+FP+FN,1)*100
        prec = TP/max(TP+FP,1)*100
        tpr  = TP/max(TP+FN,1)*100
        fpr  = FP/max(TN+FP,1)*100
        metrics = dict(label=label, MSE_overall=mse,
                       Accuracy=acc, Precision=prec, TPR=tpr, FPR=fpr,
                       TP=TP, TN=TN, FP=FP, FN=FN)
        print(f"[SVR/{label}] MSE={mse:.4f} | Acc={acc:.2f}% | TPR={tpr:.2f}%")
        return metrics
