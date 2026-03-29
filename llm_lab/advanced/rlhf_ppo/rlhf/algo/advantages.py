import torch


def compute_gae(values, rewards_scalar, action_mask, gamma=0.99, lam=0.95):
    """
    values: [B,T]
    rewards_scalar: [B]
    action_mask: [B,T] boolean (True on generated tokens)
    We'll create a per-token reward that is 0 except last action token gets rewards_scalar.
    """
    B, T = values.shape
    device = values.device

    rewards = torch.zeros((B, T), device=device)
    # last action index per sequence
    last_idx = action_mask.long().sum(dim=1) - 1  # count of action tokens -1
    # convert to positions: find indices where action_mask True
    for b in range(B):
        if last_idx[b] >= 0:
            act_positions = torch.nonzero(action_mask[b], as_tuple=False).squeeze(-1)
            rewards[b, act_positions[last_idx[b]]] = rewards_scalar[b]

    advantages = torch.zeros((B, T), device=device)
    returns = torch.zeros((B, T), device=device)

    next_value = torch.zeros((B,), device=device)
    gae = torch.zeros((B,), device=device)

    # iterate backwards
    for t in reversed(range(T)):
        mask_t = action_mask[:, t].float()
        delta = rewards[:, t] + gamma * next_value - values[:, t]
        gae = (delta + gamma * lam * gae) * mask_t + gae * (1 - mask_t)  # only update on action tokens
        advantages[:, t] = gae
        next_value = values[:, t]
        returns[:, t] = advantages[:, t] + values[:, t]

    # Normalize advantages over action tokens
    adv = advantages[action_mask]
    advantages = advantages.clone()
    advantages[action_mask] = (adv - adv.mean()) / (adv.std() + 1e-8)
    return advantages, returns
