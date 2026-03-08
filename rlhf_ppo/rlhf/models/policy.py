import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

class PolicyWithValue(nn.Module):
    def __init__(self, base_model_name: str, lora_r: int = 16, lora_alpha: int = 32, lora_dropout: float = 0.05):
        super().__init__()
        bnb_cfg = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )

        self.tokenizer = AutoTokenizer.from_pretrained(base_model_name, use_fast=True)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        base = AutoModelForCausalLM.from_pretrained(
            base_model_name,
            quantization_config=bnb_cfg,
            device_map="auto",
            torch_dtype=torch.bfloat16,
        )
        base.gradient_checkpointing_enable()
        base = prepare_model_for_kbit_training(base)

        # LoRA targets: for Qwen2.5, typical: q_proj, k_proj, v_proj, o_proj + maybe gate/up/down proj
        lora_cfg = LoraConfig(
            r=lora_r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            bias="none",
            task_type="CAUSAL_LM",
            target_modules=["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"],
        )
        self.model = get_peft_model(base, lora_cfg)

        hidden = self.model.base_model.model.config.hidden_size
        self.value_head = nn.Linear(hidden, 1)

    @torch.no_grad()
    def generate(self, prompts, max_new_tokens=128, temperature=0.7, top_p=0.9):
        tok = self.tokenizer(
            prompts, return_tensors="pt", padding=True, truncation=True, max_length=1024
        ).to(self.model.device)
        out = self.model.generate(
            **tok,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=temperature,
            top_p=top_p,
            pad_token_id=self.tokenizer.pad_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
            return_dict_in_generate=True,
            output_scores=False,
        )
        return out.sequences, tok["input_ids"].shape[1]  # sequences [B, T_total], prompt_len

    def forward_logits_and_values(self, input_ids, attention_mask):
        # Need hidden states for value head
        out = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
            use_cache=False,
        )
        logits = out.logits  # [B, T, V]
        last_hidden = out.hidden_states[-1]  # [B, T, H]
        values = self.value_head(last_hidden).squeeze(-1)  # [B, T]
        return logits, values

def logprobs_from_logits(logits, labels):
    # logits: [B,T,V], labels: [B,T]
    logp = torch.log_softmax(logits, dim=-1)
    return torch.gather(logp, dim=-1, index=labels.unsqueeze(-1)).squeeze(-1)  # [B,T]
