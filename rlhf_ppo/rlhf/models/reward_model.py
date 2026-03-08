import torch


def toy_reward(tokenizer, input_ids_batch, prompt_lens, target_phrase="final answer:"):
    # input_ids_batch: [B, T_total] (pre-shift), prompt_lens: int
    rewards = []
    for ids in input_ids_batch:
        text = tokenizer.decode(ids.tolist(), skip_special_tokens=True).lower()
        r = 1.0 if target_phrase in text else 0.0
        # mild length penalty
        r -= 0.001 * max(0, len(ids) - 1024)
        rewards.append(r)
    return torch.tensor(rewards, device=input_ids_batch.device, dtype=torch.float32)