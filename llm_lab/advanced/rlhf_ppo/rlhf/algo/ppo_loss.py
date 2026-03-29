import torch


def ppo_loss(policy_logp, logp_old, advantages, action_mask, clip_eps=0.2):
    # ratio = exp(logp - logp_old)
    ratio = torch.exp(policy_logp - logp_old)
    # only action tokens
    ratio_a = ratio[action_mask]
    adv_a = advantages[action_mask]

    unclipped = ratio_a * adv_a
    clipped = torch.clamp(ratio_a, 1.0 - clip_eps, 1.0 + clip_eps) * adv_a
    loss = -torch.mean(torch.min(unclipped, clipped))
    # for metrics
    clip_frac = torch.mean((torch.abs(ratio_a - 1.0) > clip_eps).float()).item()
    approx_kl = 0.5 * torch.mean((policy_logp[action_mask] - logp_old[action_mask])**2).item()
    return loss, clip_frac, approx_kl

def value_loss(values, returns, action_mask, vf_coef=0.5):
    v = values[action_mask]
    r = returns[action_mask]
    return vf_coef * torch.mean((v - r) ** 2)

def entropy_bonus(logits, labels, action_mask, ent_coef=0.01):
    # entropy of categorical at each token
    logp = torch.log_softmax(logits, dim=-1)
    p = torch.softmax(logits, dim=-1)
    ent = -(p * logp).sum(dim=-1)  # [B,T]
    ent_a = ent[action_mask]
    return -ent_coef * ent_a.mean()  # negative because we add to loss

def kl_penalty(logp, logp_ref, action_mask, kl_coef=0.02):
    # approx KL on actions: E[logp - logp_ref]
    kl = (logp - logp_ref)[action_mask].mean()
    return kl_coef * kl, kl.item()
