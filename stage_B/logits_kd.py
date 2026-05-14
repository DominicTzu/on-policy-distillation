"""Logits-level distillation losses for Stage B."""

import torch
import torch.nn.functional as F


def response_ce_loss(student_logits, labels):
    """Hard-label CE on response tokens only."""
    shift_logits = student_logits[:, :-1, :]
    shift_labels = labels[:, 1:]
    response_mask = shift_labels.ne(-100)

    if response_mask.sum().item() == 0:
        return shift_logits.new_tensor(0.0)

    response_logits = shift_logits[response_mask.to(shift_logits.device)]
    response_labels = shift_labels[response_mask].to(response_logits.device)
    return F.cross_entropy(response_logits, response_labels)


def response_kl_loss(
    student_logits,
    teacher_logits,
    labels,
    temperature,
    eos_token_id=None,
    exclude_eos=False,
):
    """KL(teacher || student) on response-token prediction positions only."""
    shift_student_logits = student_logits[:, :-1, :]
    shift_teacher_logits = teacher_logits[:, :-1, :]
    shift_labels = labels[:, 1:]
    response_mask = shift_labels.ne(-100)
    if exclude_eos and eos_token_id is not None:
        response_mask = response_mask & shift_labels.ne(eos_token_id)

    if response_mask.sum().item() == 0:
        return shift_student_logits.new_tensor(0.0)

    shared_vocab_size = min(
        shift_student_logits.size(-1), shift_teacher_logits.size(-1)
    )
    student_response_logits = shift_student_logits[..., :shared_vocab_size][
        response_mask.to(shift_student_logits.device)
    ]
    teacher_response_logits = shift_teacher_logits[..., :shared_vocab_size][
        response_mask.to(shift_teacher_logits.device)
    ].to(student_response_logits.device)

    student_log_probs = F.log_softmax(
        student_response_logits.float() / temperature, dim=-1
    )
    teacher_probs = F.softmax(teacher_response_logits.float() / temperature, dim=-1)
    token_kl = F.kl_div(student_log_probs, teacher_probs, reduction="none").sum(dim=-1)
    return token_kl.mean() * (temperature**2)


def distillation_losses(
    student_logits,
    teacher_logits,
    labels,
    temperature,
    eos_token_id=None,
    exclude_eos_from_kd=False,
):
    ce_loss = response_ce_loss(student_logits, labels)
    kd_loss = response_kl_loss(
        student_logits,
        teacher_logits,
        labels,
        temperature,
        eos_token_id=eos_token_id,
        exclude_eos=exclude_eos_from_kd,
    )
    return ce_loss, kd_loss
