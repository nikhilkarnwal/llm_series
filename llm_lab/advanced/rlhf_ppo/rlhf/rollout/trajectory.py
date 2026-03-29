from dataclasses import dataclass
import torch

@dataclass
class TrajectoryBatch:
    # Full sequences (prompt + generated)
    input_ids: torch.Tensor            # [B, T]
    attention_mask: torch.Tensor       # [B, T]

    # Mask of which positions correspond to generated tokens (actions)
    action_mask: torch.Tensor          # [B, T] boolean

    # Old logprobs of actions under behavior policy (pi_old)
    logp_old: torch.Tensor             # [B, T] (only valid where action_mask True)

    # Reference logprobs (pi_ref) for KL penalty
    logp_ref: torch.Tensor             # [B, T]

    # Value predictions at each token (or at action tokens)
    values: torch.Tensor               # [B, T]

    # Rewards: we’ll use scalar per sequence then spread/attach to final token for now
    rewards: torch.Tensor              # [B] (scalar reward per sequence)

    # Optional: episode ends (for future multi-step env); here always True per sequence
    dones: torch.Tensor                # [B] boolean
