"""Logits-level distillation losses for Stage B."""

import torch
import torch.nn.functional as F


def response_ce_loss(student_logits, labels):
    """Hard-label CE on response tokens only."""
    shift_logits = student_logits[:, :-1, :].contiguous()
    shift_labels = labels[:, 1:].contiguous()
    return F.cross_entropy(
        shift_logits.view(-1, shift_logits.size(-1)),
        shift_labels.view(-1),
        ignore_index=-100,
    )


def response_kl_loss(student_logits, teacher_logits, labels, temperature):
    """KL(teacher || student) on response-token prediction positions only."""
    shift_student_logits = student_logits[:, :-1, :].contiguous()
    shift_teacher_logits = teacher_logits[:, :-1, :].contiguous()
    shift_labels = labels[:, 1:].contiguous()
    response_mask = shift_labels.ne(-100)

    if response_mask.sum().item() == 0:
        return shift_student_logits.new_tensor(0.0)

    student_log_probs = F.log_softmax(
        shift_student_logits.float() / temperature, dim=-1
    )
    teacher_probs = F.softmax(shift_teacher_logits.float() / temperature, dim=-1)
    token_kl = F.kl_div(student_log_probs, teacher_probs, reduction="none").sum(dim=-1)

    token_kl = token_kl * response_mask.to(token_kl.dtype)
    return token_kl.sum() / response_mask.sum() * (temperature**2)


def distillation_losses(student_logits, teacher_logits, labels, temperature):
    ce_loss = response_ce_loss(student_logits, labels)
    kd_loss = response_kl_loss(student_logits, teacher_logits, labels, temperature)
    return ce_loss, kd_loss
