# services/llm_service.py
from typing import List, Literal, Optional, Protocol
import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
try:
    from transformers import BitsAndBytesConfig  # optional
except Exception:
    BitsAndBytesConfig = None

Mode = Literal["merge", "fusion"]

class LlmService(Protocol):
    def summarize(self, texts: List[str], target_tokens: int = 220, mode: Mode = "merge") -> str: ...

def build_merge_prompt(texts: List[str], target_tokens: int) -> str:
    parts = [f"--- TEXT {i+1} ---\n{t}" for i, t in enumerate(texts)]
    joined = "\n\n".join(parts)
    return (
        "You merge near-duplicate or overlapping notes into ONE concise, non-redundant summary.\n"
        "Keep shared core ideas and add unique details from each input. Avoid repetition.\n"
        f"Target about {target_tokens} tokens. Output ONLY the merged summary.\n\n"
        f"{joined}\n\nMerged summary:\n"
    )

def build_fusion_prompt(texts: List[str]) -> str:
    lines = [f"TEXT {i+1}: {t}" for i, t in enumerate(texts, 1)]
    joined = "\n".join(lines)
    return (
        "Combine the inputs into ONE concise sentence.\n"
        "Rules:\n"
        "• Preserve all distinct events/facts.\n"
        "• Normalize repeated time expressions (e.g., say 'today' once).\n"
        "• Avoid redundancy.\n"
        "• Output exactly one sentence, no preamble.\n\n"
        f"{joined}\n\nOne-sentence fusion:"
    )

class HfLlmService:
    def __init__(
        self,
        model_name: str = "lmsys/vicuna-7b-v1.5",
        four_bit: bool = True,          # enable 4-bit for 12GB VRAM
        max_ctx: int = 2048,            # keep modest for KV-cache headroom
        temperature: float = 0.7,
        top_p: float = 0.95,
        repetition_penalty: float = 1.05,
        seed: Optional[int] = None,
        use_safetensors: bool = True,
    ):
        self.temperature = temperature
        self.top_p = top_p
        self.max_ctx = max_ctx
        self.repetition_penalty = repetition_penalty

        if seed is not None:
            torch.manual_seed(seed)

        # Prefer 4-bit when CUDA + bitsandbytes are available
        use_4bit = bool(four_bit and torch.cuda.is_available() and BitsAndBytesConfig is not None)
        bnb = None
        if use_4bit:
            bnb = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.float16,
            )

        self.tok = AutoTokenizer.from_pretrained(model_name, use_fast=False)

        # Fit 12GB GPU: allow limited CPU spill, avoid disk offload
        if torch.cuda.is_available():
            device_map = {"": "cuda:0"}
            max_memory = {"cuda:0": "11GiB", "cpu": "28GiB"}  # keep headroom on both
            torch_dtype = torch.float16
        else:
            device_map = "cpu"
            max_memory = None
            torch_dtype = torch.float32

        print(f"[LlmService] loading model={model_name} 4bit={use_4bit} "
              f"device_map={device_map} dtype={torch_dtype}")

        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            quantization_config=bnb,             # None if not using 4-bit
            device_map=device_map,
            max_memory=max_memory,               # enables CPU spill, not disk
            low_cpu_mem_usage=True,
            torch_dtype=torch_dtype,
            use_safetensors=use_safetensors,
        ).eval()

        if self.model.generation_config.pad_token_id is None:
            self.model.generation_config.pad_token_id = self.tok.eos_token_id

        if torch.cuda.is_available():
            torch.backends.cuda.matmul.allow_tf32 = True
            print("[LlmService] device map:", getattr(self.model, "hf_device_map", "n/a"))
        # Set default generation behavior on the model itself
        gc = self.model.generation_config
        gc.do_sample = True
        gc.temperature = self.temperature
        gc.top_p = self.top_p
        gc.repetition_penalty = self.repetition_penalty
        gc.eos_token_id = self.tok.eos_token_id
        if gc.pad_token_id is None:
            gc.pad_token_id = self.tok.eos_token_id

    @torch.inference_mode()
    def summarize(self, texts: List[str], target_tokens: int = 220, mode: Mode = "merge") -> str:
        print(f"[LlmService] summarize() mode={mode}, target={target_tokens}, n_texts={len(texts)}")

        # Build a single user message
        if mode == "fusion":
            user_msg = build_fusion_prompt(texts)
            max_new = min(64, max(24, target_tokens))
        else:
            user_msg = build_merge_prompt(texts, target_tokens)
            max_new = min(256, target_tokens)

        # Helpful previews
        for i, t in enumerate(texts, 1):
            preview = (t[:80] + "...") if len(t) > 80 else t
            print(f"  [LlmService] Text {i} preview: {preview}")

        # Use chat template if available (Vicuna/LLaMA instruction formats)
        # Use chat template if available; otherwise fall back to a manual chat prefix
        use_chat = hasattr(self.tok, "apply_chat_template") and getattr(self.tok, "chat_template", None)

        if use_chat:
            try:
                messages = [{"role": "user", "content": user_msg}]
                prompt_str = self.tok.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True  # adds assistant header
                )
            except Exception as e:
                # Fallback if template is missing or apply_chat_template errors
                # Vicuna-style single-turn prompt:
                prompt_str = f"USER: {user_msg}\nASSISTANT:"
        else:
            # No chat templating on this tokenizer — use a simple Vicuna/LLaMA-style header
            prompt_str = f"USER: {user_msg}\nASSISTANT:"

        # Tokenize
        inputs = self.tok(
            prompt_str,
            return_tensors="pt",
            truncation=True,
            max_length=self.max_ctx,
        )
        dev = self.model.get_input_embeddings().weight.device
        inputs = {k: v.to(dev) for k, v in inputs.items()}

        # Generate (sampling params already on generation_config, but keep max_new here)
        out_ids = self.model.generate(
            **inputs,
            max_new_tokens=max_new,
        )

        # Decode ONLY newly generated tokens (don’t echo prompt/system instructions)
        prompt_len = inputs["input_ids"].shape[1]
        gen_only = out_ids[0][prompt_len:]
        text = self.tok.decode(gen_only, skip_special_tokens=True).strip()

        # Optional post-trim for fusion: keep it to a single sentence if the model rambles
        if mode == "fusion":
            # naive first-sentence cut (handles ., !, ?). You can replace with a stricter splitter if you like.
            for sep in [".", "!", "?"]:
                idx = text.find(sep)
                if idx != -1:
                    text = text[: idx + 1].strip()
                    break

        print(f"[LlmService] -> generated ~{len(text)//4} est tokens")
        return text