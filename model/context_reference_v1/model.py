"""Historical network with an optional nine-feature context-reference branch.

The original read encoder, seven-base embedding, pooling and 63 distribution
summaries are unchanged. References are fitted by the trainer on permitted
training sites; this module never fits references or reads labels.
"""
from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import torch
from torch import nn


def source_sha256(path):
    """Canonical code fingerprint: preserve content, normalize CRLF to LF."""
    content = Path(path).read_bytes().replace(b"\r\n", b"\n")
    return hashlib.sha256(content).hexdigest()


def read_transform(x):
    x = np.array(x, dtype=np.float32, copy=True)
    for j in (0, 3, 6):
        x[..., j] = np.sign(x[..., j]) * np.log1p(np.abs(x[..., j]) * 1000)
    for j in (1, 4, 7):
        x[..., j] = np.sign(x[..., j]) * np.log1p(np.abs(x[..., j]))
    return x


def summary_transform(x):
    scale = np.tile([1000., 1., 1.], 21).astype(np.float32)
    return (np.sign(x) * np.log1p(np.abs(x) * scale)).astype(np.float32)


class SiteModel(nn.Module):
    def __init__(self, config, norm):
        super().__init__()
        self.config = dict(config)
        self.context_mode = config.get("context_mode", "none")
        if self.context_mode not in ("none", "relative"):
            raise ValueError("context_mode must be none or relative")
        self.extra_n = 9 if self.context_mode == "relative" else 0
        for name in ("read_mean", "read_sd", "summary_mean", "summary_sd"):
            self.register_buffer(name, torch.tensor(norm[name], dtype=torch.float32))
        if self.read_mean.shape != (9,) or self.read_sd.shape != (9,):
            raise ValueError("Read normalization must contain nine features")
        if self.summary_mean.shape != (63,) or self.summary_sd.shape != (63,):
            raise ValueError("Summary normalization must contain 63 features")
        for name in ("read_mean", "read_sd", "summary_mean", "summary_sd"):
            if not torch.isfinite(getattr(self, name)).all():
                raise ValueError("Normalization values must be finite")
        if (self.read_sd <= 0).any() or (self.summary_sd <= 0).any():
            raise ValueError("Normalization scales must be positive")
        w, drop = config["width"], config["dropout"]
        self.embedding = nn.Embedding(5, 4)
        self.encoder = nn.Sequential(nn.Linear(37, w), nn.SiLU(),
                                     nn.Linear(w, w), nn.SiLU())
        self.att_v = nn.Linear(w, w)
        self.att_u = nn.Linear(w, w)
        self.att_w = nn.Linear(w, 1)
        self.read_logit = nn.Linear(w, 1)
        summary_n = config.get("summary_n", 63)
        if not 1 <= summary_n <= 63:
            raise ValueError("summary_n must be between one and 63")
        mode = config["pool"]
        if mode not in ("summary", "attention", "deepsets", "noisyor"):
            raise ValueError("Unknown read pooling mode")
        pooled = 0 if mode == "summary" else (3*w if mode in ("deepsets", "attention") else 1)
        self.head = nn.Sequential(nn.Linear(28+summary_n+pooled+self.extra_n, w),
            nn.SiLU(), nn.Dropout(drop), nn.Linear(w, w//2), nn.SiLU(),
            nn.Dropout(drop), nn.Linear(w//2, 1))

    def forward(self, reads, mask, sequence, summaries, extra_features=None):
        context = self.embedding(sequence).flatten(1)
        summary = ((summaries-self.summary_mean)/self.summary_sd)[:, :self.config.get("summary_n", 63)]
        pieces = [context, summary]
        mode = self.config["pool"]
        if mode != "summary":
            if mask.dtype != torch.bool or not mask.any(dim=1).all():
                raise ValueError("Every site requires at least one unmasked read")
            x = (reads-self.read_mean)/self.read_sd
            h = self.encoder(torch.cat([x, context[:, None, :].expand(-1, x.shape[1], -1)], dim=-1))
            m = mask[..., None]
            n = m.sum(1).clamp_min(1)
            mean = (h*m).sum(1)/n
            var = (((h-mean[:, None, :])**2)*m).sum(1)/n
            if mode == "attention":
                logits = self.att_w(torch.tanh(self.att_v(h))*torch.sigmoid(self.att_u(h))).squeeze(-1)
                weights = torch.softmax(logits.masked_fill(~mask, -torch.inf), dim=1)
                pooled = (weights[..., None]*h).sum(1)
                pieces += [pooled, mean, torch.sqrt(var+1e-6)]
            elif mode == "deepsets":
                maximum = h.masked_fill(~m, -torch.inf).max(1).values
                pieces += [mean, torch.sqrt(var+1e-6), maximum]
            else:
                log_survival = (-nn.functional.softplus(self.read_logit(h))*m).sum(1)/n
                p = -torch.expm1(20*log_survival)
                pieces += [torch.logit(p.clamp(1e-6, 1-1e-6))]
        if self.extra_n:
            if extra_features is None or extra_features.shape != (len(reads), 9):
                raise ValueError("Relative mode requires an N x 9 context-reference tensor")
            if not torch.isfinite(extra_features).all():
                raise ValueError("Context-reference features must be finite")
            pieces.append(extra_features)
        elif extra_features is not None and extra_features.shape != (len(reads), 0):
            raise ValueError("None mode must not receive additional features")
        return self.head(torch.cat(pieces, dim=1)).squeeze(-1)


def load_model(path, allow_development_model=False):
    """Load only this project's own trusted bundle; inference stays on CPU."""
    bundle = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(bundle, dict) or bundle.get("kind") != "context_reference_v1":
        raise ValueError("Expected a context_reference_v1 model bundle")
    if bundle.get("stage") not in ("cv_inner", "cv_refit", "final_fitting", "final_data0", "smoke"):
        raise ValueError("Unrecognised model stage")
    if bundle["stage"] != "final_data0" and not allow_development_model:
        raise ValueError("Prediction requires final_data0; explicitly allow a development model for checks")
    if bundle["stage"] == "final_data0":
        completed = bundle.get("fit_completed")
        if not isinstance(completed, dict):
            raise ValueError("Final model is missing completed-fit promotion metadata")
        for name in ("signature", "best_artifact_sha256"):
            value = completed.get(name)
            if not isinstance(value, str) or len(value) != 64 or any(c not in "0123456789abcdef" for c in value):
                raise ValueError("Final completed-fit signature/hash is invalid")
        epochs = completed.get("epochs_completed")
        selected = completed.get("selected_epochs")
        if (type(epochs) is not int or type(selected) is not int or selected < 1
                or epochs != selected or bundle.get("epoch") != selected):
            raise ValueError("Final model has not completed every prescribed epoch")
        if completed["signature"] != bundle.get("fit_signature"):
            raise ValueError("Final completed-fit signature differs from its model bundle")
    fingerprints = bundle.get("input_fingerprints")
    if fingerprints is not None and not isinstance(fingerprints, dict):
        raise ValueError("Model input/source fingerprint metadata is invalid")
    sources = None if fingerprints is None else fingerprints.get("source_sha256")
    if bundle["stage"] == "final_data0" and not isinstance(sources, dict):
        raise ValueError("Final model is missing its fitted source fingerprint map")
    if sources is not None:
        if not isinstance(sources, dict):
            raise ValueError("Model source fingerprint map is invalid")
        for name in ("model.py", "context_features.py", "data.py"):
            if sources.get(name) != source_sha256(Path(__file__).resolve().parent / name):
                raise ValueError("Model prediction source differs from its fitting source: " + name)
    model = SiteModel(bundle["config"], bundle["norm"])
    reference = bundle.get("context_reference")
    if model.context_mode == "relative" and not isinstance(reference, dict):
        raise ValueError("Relative model bundle is missing its fitted context reference")
    if model.context_mode == "none" and reference is not None:
        raise ValueError("None-mode model must not contain an active context reference")
    model.load_state_dict(bundle["state_dict"], strict=True)
    if not all(torch.isfinite(value).all() for value in model.state_dict().values()):
        raise ValueError("Saved model contains non-finite weights or normalization")
    model.eval()
    return model, bundle
