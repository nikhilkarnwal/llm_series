import torch
from rlhf_ppo.rlhf.algo.ppo_loss import kl_penalty
from rlhf_ppo.rlhf.algo.ppo_loss import entropy_bonus
from rlhf_ppo.rlhf.algo.ppo_loss import value_loss
from rlhf_ppo.rlhf.algo.ppo_loss import ppo_loss
from rlhf_ppo.rlhf.algo.advantages import compute_gae
from rlhf_ppo.rlhf.models.policy import logprobs_from_logits
from rlhf_ppo.rlhf.rollout.trajectory import TrajectoryBatch
from rlhf_ppo.rlhf.models.policy import PolicyWithValue
import torch.optim as optim

def ppo_update_step(policy: PolicyWithValue, batch: TrajectoryBatch, optimizer, clip_eps=0.2,
                    kl_coef=0.02, vf_coef=0.5, ent_coef=0.01, grad_clip=1.0):

    # Forward current policy on stored sequences
    logits, values = policy.forward_logits_and_values(batch.input_ids, batch.attention_mask)
    # Align logits to next-token labels: batch.input_ids already shifted (labels)
    # Need labels = input_ids (next tokens), logits correspond to predicting each token from previous
    labels = batch.input_ids
    logits = logits[:, :-1, :]
    labels = labels[:, 1:]
    values = values[:, :-1]

    action_mask = batch.action_mask[:, 1:]  # align to labels space
    # Compute logp under current policy for taken tokens
    logp = logprobs_from_logits(logits, labels)

    # Advantages/returns
    adv, rets = compute_gae(values.detach(), batch.rewards, action_mask)

    # Loss terms
    pol_loss, clip_frac, approx_kl = ppo_loss(logp, batch.logp_old[:, 1:], adv, action_mask, clip_eps)
    v_loss = value_loss(values, rets, action_mask, vf_coef)
    ent_loss = entropy_bonus(logits, labels, action_mask, ent_coef)
    kl_loss, kl_val = kl_penalty(logp, batch.logp_ref[:, 1:], action_mask, kl_coef)

    loss = pol_loss + v_loss + ent_loss + kl_loss

    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(policy.parameters(), grad_clip)
    optimizer.step()

    metrics = {
        "loss_total": loss.item(),
        "loss_policy": pol_loss.item(),
        "loss_value": v_loss.item(),
        "loss_entropy": ent_loss.item(),
        "loss_kl": kl_loss.item(),
        "clip_frac": clip_frac,
        "approx_kl": approx_kl,
        "kl_ref": kl_val,
        "reward_mean": batch.rewards.mean().item(),
    }
    return metrics
