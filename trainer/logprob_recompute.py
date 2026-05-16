"""
BF16 policy log-probability recomputation.

Given a batch of tokenized (prompt + response) sequences and a causal LM,
returns per-token log-probabilities for the response tokens only.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


def recompute_train_logprobs_with_grad(
    model: torch.nn.Module,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    response_mask: torch.Tensor,
) -> torch.Tensor:
    """
    Compute per-token log-probabilities of the training model with gradient flow.

    Called once per training step after rollout. The model does a single
    forward pass over all (prompt + response) sequences in the batch — no
    autoregressive generation. Gradients flow through the returned logprobs
    into loss.backward() and then into the model weights.

    Args:
        model:          causal LM (BF16 training model)
        input_ids:      [batch, seq_len]  prompt + response tokens, left-padded
        attention_mask: [batch, seq_len]  1 = real token, 0 = padding
        response_mask:  [batch, seq_len]  1 = response token, 0 = prompt/pad

    Returns:
        logprobs: [batch, response_len]  log π_train(token_t | context)
                  only response tokens, prompt positions zeroed and trimmed.
    """
    outputs = model(input_ids=input_ids, attention_mask=attention_mask)
    logits = outputs.logits  # [batch, seq_len, vocab]

    shift_logits = logits[:, :-1, :].float()
    shift_labels = input_ids[:, 1:]
    shift_resp   = response_mask[:, 1:]

    log_probs_all = F.log_softmax(shift_logits, dim=-1)
    token_logprobs = log_probs_all.gather(
        dim=-1,
        index=shift_labels.unsqueeze(-1),
    ).squeeze(-1)

    token_logprobs = token_logprobs * shift_resp
    response_len = int(shift_resp.sum(dim=-1).max().item())

    if response_len == 0:
        return torch.zeros(input_ids.shape[0], 1, device=input_ids.device)

    return token_logprobs[:, -response_len:]
