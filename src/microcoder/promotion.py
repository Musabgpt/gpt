"""Checkpoint promotion policy for MicroCoder.

A candidate is promoted only when validation loss improves and the Python
anchor suite does not regress beyond the configured tolerance.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping


@dataclass(frozen=True)
class PromotionPolicy:
    min_val_loss_improvement: float = 0.0
    max_anchor_regression: float = 0.0


@dataclass(frozen=True)
class PromotionDecision:
    promote: bool
    reasons: tuple[str, ...]


def decide_promotion(
    baseline: Mapping[str, float],
    candidate: Mapping[str, float],
    policy: PromotionPolicy = PromotionPolicy(),
) -> PromotionDecision:
    required = {"val_loss", "anchor_score"}
    missing = sorted(required - baseline.keys() | required - candidate.keys())
    if missing:
        return PromotionDecision(False, (f"missing metrics: {', '.join(missing)}",))

    b_loss = float(baseline["val_loss"])
    c_loss = float(candidate["val_loss"])
    b_anchor = float(baseline["anchor_score"])
    c_anchor = float(candidate["anchor_score"])

    reasons: list[str] = []
    improvement = b_loss - c_loss
    if improvement <= policy.min_val_loss_improvement:
        reasons.append(
            f"validation loss improvement {improvement:.6g} is not greater than "
            f"required {policy.min_val_loss_improvement:.6g}"
        )

    regression = b_anchor - c_anchor
    if regression > policy.max_anchor_regression:
        reasons.append(
            f"anchor regression {regression:.6g} exceeds allowed "
            f"{policy.max_anchor_regression:.6g}"
        )

    return PromotionDecision(not reasons, tuple(reasons))
