import torch
import copy

from rlhf.models.policy import PolicyWithValue

def make_reference_model(policy: PolicyWithValue) -> PolicyWithValue:
    ref = copy.deepcopy(policy)
    for p in ref.parameters():
        p.requires_grad_(False)
    ref.eval()
    return ref

