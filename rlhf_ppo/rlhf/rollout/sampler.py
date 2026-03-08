import torch

from rlhf_ppo.rlhf.models.policy import PolicyWithValue, logprobs_from_logits
from rlhf_ppo.rlhf.rollout.trajectory import TrajectoryBatch

@torch.no_grad()
def collect_rollouts(policy: PolicyWithValue, ref: PolicyWithValue, prompts, max_new_tokens=128):
    device = policy.model.device
    seqs, prompt_len = policy.generate(prompts, max_new_tokens=max_new_tokens)

    # Build attention mask (pad-free because generate returns full length; but sequences may vary with eos)
    # We'll create attention_mask = 1 for all tokens; then mask out padding if present (rare here).
    input_ids = seqs.to(device)
    attention_mask = torch.ones_like(input_ids, device=device)

    # Action mask: generated tokens are positions [prompt_len ... T-1]
    B, T = input_ids.shape
    action_mask = torch.zeros((B, T), dtype=torch.bool, device=device)
    action_mask[:, prompt_len:] = True

    # Compute logprobs under current policy and reference for taken tokens
    logits, values = policy.forward_logits_and_values(input_ids, attention_mask)
    ref_logits, _ = ref.forward_logits_and_values(input_ids, attention_mask)

    # Next-token prediction: logprob of token t is from logits at t-1
    # Shift
    labels = input_ids[:, 1:]
    logits = logits[:, :-1, :]
    ref_logits = ref_logits[:, :-1, :]
    values = values[:, :-1]

    # Shift masks accordingly
    action_mask_shift = action_mask[:, 1:]  # aligns with labels/logits

    logp = logprobs_from_logits(logits, labels)
    logp_ref = logprobs_from_logits(ref_logits, labels)

    # Store old logp as behavior policy logp (on-policy collection)
    logp_old = logp.detach()

    # For now, placeholder rewards (Week 1 toy reward)
    rewards = torch.zeros((B,), device=device)
    dones = torch.ones((B,), dtype=torch.bool, device=device)

    batch = TrajectoryBatch(
        input_ids=input_ids[:, 1:],               # align to labels space [B, T-1]
        attention_mask=attention_mask[:, 1:],
        action_mask=action_mask_shift,
        logp_old=logp_old,
        logp_ref=logp_ref.detach(),
        values=values,
        rewards=rewards,
        dones=dones,
    )
    return batch
