import torch
import torch.nn.functional as F
from typing import Tuple


def compute_nessie_multitask_loss(
    net_outputs: dict,
    target_value_bin: torch.Tensor,
    target_search_policy_p1: torch.Tensor,
    target_search_policy_p2: torch.Tensor,
    target_survival: torch.Tensor,
    target_next_ko: torch.Tensor,
    target_hp_diff: torch.Tensor
) -> Tuple[torch.Tensor, dict]:
    """
    Calcula la pérdida combinada con cabezas auxiliares para lograr convergencia en ~375k partidas.
    """
    # 1. Pérdida de valor distribuido (Cross-Entropy sobre los 21 bins de retorno final)
    loss_value = F.cross_entropy(net_outputs["value_logits"], target_value_bin)

    # 2. Pérdida de destilación de política hacia la estrategia de equilibrio Double-Oracle (x*, y*)
    log_probs_p1 = F.log_softmax(net_outputs["p1_logits"], dim=-1)
    log_probs_p2 = F.log_softmax(net_outputs["p2_logits"], dim=-1)
    loss_policy = -(torch.sum(target_search_policy_p1 * log_probs_p1, dim=-1).mean() +
                    torch.sum(target_search_policy_p2 * log_probs_p2, dim=-1).mean()) * 0.5

    # 3. Pérdidas de Cabezas Auxiliares (Gradientes densos de estado futuro)
    loss_aux_surv = F.binary_cross_entropy_with_logits(
        net_outputs["aux_survival_logits"], target_survival
    )
    loss_aux_ko = F.binary_cross_entropy_with_logits(
        net_outputs["aux_next_turn_ko_head"], target_next_ko
    ) if "aux_next_turn_ko_head" in net_outputs else F.binary_cross_entropy_with_logits(
        net_outputs["aux_ko_logits"], target_next_ko
    )
    loss_aux_hp = F.mse_loss(net_outputs["aux_hp_diff"], target_hp_diff)

    total_loss = (
        1.0 * loss_value +
        0.5 * loss_policy +
        0.3 * loss_aux_surv +
        0.3 * loss_aux_ko +
        0.2 * loss_aux_hp
    )

    metrics = {
        "loss_total": total_loss.item(),
        "loss_value": loss_value.item(),
        "loss_policy": loss_policy.item(),
        "loss_aux": (loss_aux_surv + loss_aux_ko + loss_aux_hp).item()
    }
    return total_loss, metrics
