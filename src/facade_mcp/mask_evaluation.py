"""Pixel metrics for explicitly paired, human-approved instance masks."""
from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
from PIL import Image
from pydantic import Field, model_validator

from .model_gateway import Strict


class MaskPair(Strict):
    instance_id: str = Field(min_length=1)
    prediction: Path
    reference: Path


class MaskEvaluation(Strict):
    dataset_version: str = Field(min_length=1)
    reference_approved_by: str = Field(min_length=1)
    pairs: list[MaskPair] = Field(min_length=1)

    @model_validator(mode="after")
    def unique_instances(self):
        if len({p.instance_id for p in self.pairs}) != len(self.pairs):
            raise ValueError("Duplicate instance IDs")
        if not self.reference_approved_by.strip():
            raise ValueError("Reference approver is required")
        return self


def read_mask(path: Path):
    with Image.open(path) as image:
        values = np.asarray(image)
    if values.ndim != 2 or not np.isin(values, [0, 1, 255]).all():
        raise ValueError(f"Expected a single-channel binary mask: {path.name}")
    return values > 0


def evaluate_masks(spec: MaskEvaluation, base: Path):
    rows = []
    for pair in spec.pairs:
        pred_path = (base / pair.prediction).resolve()
        ref_path = (base / pair.reference).resolve()
        pred, ref = read_mask(pred_path), read_mask(ref_path)
        if pred.shape != ref.shape:
            raise ValueError(f"Mask dimensions differ: {pair.instance_id}")
        # Empty references are not positive defect instances. Reject rather than
        # awarding a perfect score to empty/empty masks.
        if not ref.any():
            raise ValueError(f"Empty positive reference: {pair.instance_id}")
        overlap = int(np.logical_and(pred, ref).sum())
        union = int(np.logical_or(pred, ref).sum())
        rows.append({"instance_id": pair.instance_id,
                     "dice": 2 * overlap / (int(pred.sum()) + int(ref.sum())),
                     "iou": overlap / union,
                     "prediction_sha256": hashlib.sha256(pred_path.read_bytes()).hexdigest(),
                     "reference_sha256": hashlib.sha256(ref_path.read_bytes()).hexdigest(),
                     "width": pred.shape[1], "height": pred.shape[0]})
    return {"dataset_version": spec.dataset_version,
            "reference_approved_by": spec.reference_approved_by,
            "scope": "paired positive-instance masks only; not detection recall or whole-report acceptance",
            "instances": rows, "count": len(rows),
            "macro_dice": sum(r["dice"] for r in rows) / len(rows),
            "macro_iou": sum(r["iou"] for r in rows) / len(rows)}
