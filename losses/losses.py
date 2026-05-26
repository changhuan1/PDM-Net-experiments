from __future__ import annotations

import torch
import torch.nn.functional as F


def symmetric_kl(logits_a: torch.Tensor, logits_b: torch.Tensor) -> torch.Tensor:
    log_a = F.log_softmax(logits_a, dim=1)
    log_b = F.log_softmax(logits_b, dim=1)
    prob_a = log_a.exp()
    prob_b = log_b.exp()
    return 0.5 * (
        F.kl_div(log_a, prob_b, reduction="batchmean")
        + F.kl_div(log_b, prob_a, reduction="batchmean")
    )


def compute_loss(outputs: dict[str, torch.Tensor], labels: torch.Tensor, cfg: dict) -> tuple[torch.Tensor, dict[str, float]]:
    if "logits_p" not in outputs:
        loss = F.cross_entropy(outputs["logits"], labels)
        return loss, {"loss": float(loss.detach().cpu()), "ce": float(loss.detach().cpu())}

    lambda_mask = float(cfg.get("lambda_mask", 0.0))
    lambda_consistency = float(cfg.get("lambda_consistency", 0.0))
    use_fused_ce = bool(cfg.get("use_fused_ce", False))

    loss_g = F.cross_entropy(outputs["logits_g"], labels)
    loss_p = F.cross_entropy(outputs["logits_p"], labels)
    total = loss_g + loss_p
    logs = {
        "loss_g": float(loss_g.detach().cpu()),
        "loss_p": float(loss_p.detach().cpu()),
    }

    if use_fused_ce:
        loss_fused = F.cross_entropy(outputs["logits"], labels)
        total = total + loss_fused
        logs["loss_fused"] = float(loss_fused.detach().cpu())

    if lambda_consistency > 0 and "logits_g" in outputs and "logits_p" in outputs:
        loss_cons = symmetric_kl(outputs["logits_g"], outputs["logits_p"])
        total = total + lambda_consistency * loss_cons
        logs["loss_consistency"] = float(loss_cons.detach().cpu())

    selected_mask = outputs.get("selected_mask")
    if lambda_mask > 0 and selected_mask is not None:
        loss_mask = selected_mask.mean()
        total = total + lambda_mask * loss_mask
        logs["loss_mask"] = float(loss_mask.detach().cpu())

    logs["loss"] = float(total.detach().cpu())
    return total, logs
