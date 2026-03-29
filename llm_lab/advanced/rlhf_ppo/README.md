# 1) The System You’ll Build (Anthropic-style architecture)

### Components

1. **Policy**: LLM with LoRA adapters (trainable)
2. **Reference policy**: frozen snapshot of SFT policy (for KL)
3. **Reward model (RM)**: scores (prompt, response) → scalar reward
4. **Rollout workers**: generate trajectories + logprobs + values
5. **PPO trainer**: computes advantages + PPO loss + KL penalty + update
6. **Eval harness**: deterministic prompt suites + win-rate + safety stats
7. **Experiment manager**: configs, seeds, checkpoints, metrics, artifacts

This directly matches:

* RL infra & training abstractions 
* distributed experiment management (we’ll simulate via async workers) 
* profiling/optimization + caching 
* clean APIs + automated testing frameworks 

---

# 2) Repo Structure (copy this as-is)

```
rlhf-ppo-anthropic/
  README.md
  pyproject.toml

  configs/
    sft.yaml
    rm.yaml
    ppo.yaml
    eval.yaml

  rlhf/
    __init__.py

    models/
      policy.py          # load LLM + LoRA, forward for logprobs, generate
      reference.py       # frozen reference wrapper
      reward_model.py    # RM training + inference
      value_head.py      # value function head for PPO

    data/
      preference_dataset.py   # (prompt, chosen, rejected)
      prompt_dataset.py       # prompts for rollouts
      collate.py              # packing, masking

    rollout/
      sampler.py         # batched generation
      trajectory.py      # stores tokens, logprobs, values, rewards, masks
      async_workers.py   # asyncio-based rollout workers (Colab-safe)
      cache.py           # prompt->response caching for eval / rollouts

    algo/
      advantages.py      # GAE, whitening
      ppo_loss.py        # clipped objective + value loss + entropy
      kl.py              # exact/approx KL computations
      trainer.py         # main PPO loop (opt step, sched, grad clip)

    eval/
      suites.py          # reasoning/safety/code mini-suites
      metrics.py         # win-rate, refusal rate, length drift
      judge.py           # RM-as-judge or LLM-as-judge (optional)

    infra/
      logger.py          # wandb/jsonl
      checkpoint.py      # save/restore (LoRA adapters + optimizer)
      profiling.py       # torch.profiler, token/sec, memory
      config.py          # hydra/omegaconf or plain yaml
      determinism.py     # seed control

  scripts/
    run_sft.py
    train_rm.py
    run_ppo.py
    run_eval.py
```

This looks like how a Research Engineer actually builds RL infra.

---

# 3) Month 1 Plan (4 weeks) — exact build order

## Week 1 — Minimal PPO on a toy reward (correctness first)

**Goal:** PPO loop works end-to-end with stable metrics.

### Deliverables

* `Trajectory` object storing: tokens, attention_mask, logprobs_old, values, rewards, dones
* `Policy.generate()` returns tokens + logprobs
* `ValueHead` attached to policy hidden states

### Implementation steps (21 hrs)

1. **Policy wrapper**

* Load 7B model 4-bit
* Apply LoRA to attention + MLP layers
* Implement:

  * `logprobs(input_ids, attention_mask, action_mask)` for generated tokens
  * `generate(prompts, max_new_tokens, temperature, top_p)`

2. **Value head**

* Linear head on last hidden states
* Only computes values on generated token positions

3. **PPO loss**

* Clipped policy objective
* Value loss (clipped optional)
* Entropy bonus
* Gradient clipping + LR schedule

4. **Toy reward**

* Don’t start with RM yet. Use something simple to validate mechanics:

  * reward = +1 if response contains a target string / correct format
  * penalty for length > L
* Add KL penalty vs reference (even in toy)

**Success criteria**

* No NaNs for 1–2k steps
* KL stays in a sane range
* Entropy doesn’t instantly collapse
* Token/sec and VRAM logged

---

## Week 2 — Train Reward Model (real RLHF begins)

**Goal:** reward becomes meaningful, and PPO reacts to it.

### Build

* Preference dataset ingestion: `(prompt, chosen, rejected)`
* Train RM as a pairwise ranker:

  * score(prompt, chosen) > score(prompt, rejected)
* Use a smaller backbone for RM if needed (e.g., DeBERTa or small transformer) to keep Colab fast.

**Success criteria**

* RM validation accuracy above baseline
* Reward distribution stable (no absurd extremes)
* RM inference < 50ms per sample (batched)

---

## Week 3 — PPO-RLHF with KL control + stability defenses

**Goal:** PPO fine-tunes policy against RM without collapse.

### Add

* **Adaptive KL controller** (target KL)
* Reward normalization/clipping
* Response length normalization
* Refusal/degenerate behavior detectors

**Metrics you MUST log**

* mean reward, std reward
* KL mean / p95
* entropy mean
* clip fraction
* value loss
* advantage stats
* response length drift
* refusal rate

**Success criteria**

* PPO improves reward without runaway KL
* No “always refuse” collapse
* No repetitive token loops

---

## Week 4 — Evaluation Harness + “Research Engineer” polish

**Goal:** you can demonstrate improvements credibly.

### Build

* Fixed eval prompt suites:

  * reasoning (math word problems)
  * harmlessness (benign safety prompts)
  * code (small Python tasks)
* Win-rate:

  * PPO vs SFT
  * PPO vs DPO (if you add DPO quickly)
* Regression tests:

  * “these 20 prompts must not get worse”

**Success criteria**

* You can produce a clean table:

  * win-rate ↑
  * refusal rate controlled
  * length drift bounded
  * qualitative examples

---

# 4) Colab-specific engineering choices (to avoid pain)

### Model + memory

* Use 4-bit + LoRA
* gradient checkpointing ON
* batch with grad accumulation

### Rollouts

* Batch generation heavily (avoid per-sample generation)
* Maintain a prompt cache for eval to save money/time

### Async workers (role-aligned!)

Implement `asyncio` rollout workers even on a single GPU:

* one coroutine prepares prompts / postprocesses
* one coroutine batches generation
* one coroutine computes RM scores

This gives you a credible story for:

> async/concurrent programming and distributed experiment management 

---

# 5) What to Put on Resume After Month 1 (new bullets)

Add under Uber or “Selected Projects”:

* Built PPO-based RLHF system for 7B LLM with reward modeling, KL control, and evaluation harness; implemented clean training abstractions, stability monitoring, and performance profiling (token/sec, memory).
* Designed async rollout + reward scoring pipeline with caching and batched inference to accelerate training/evaluation workflows.

These directly match the job bullets .

---

# 6) Next: I’ll give you the exact PPO math & code skeleton

To start immediately, I’ll provide:

* `Trajectory` dataclass
* GAE computation
* PPO loss function (token-masked)
* KL penalty computation
* Training loop skeleton designed for LoRA + 4-bit

Before I do that: which base model do you prefer to use on Colab?

Pick one (no wrong choice):

1. **Mistral 7B**
2. **Llama 3 8B**
3. **Qwen2.5 7B**

If you don’t care, I’ll default to **Qwen2.5 7B** because it’s strong and fine-tunes well.
