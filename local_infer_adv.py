import os
import argparse
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_dir", required=True, help="Path to local HF-format model directory")
    ap.add_argument("--prompt", default="Hello! In one short sentence, say where you are running.")
    ap.add_argument("--max_new_tokens", type=int, default=30)
    args = ap.parse_args()

    # Hard force offline behavior (to catch missing files early)
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("torch:", torch.__version__)
    print("cuda available:", torch.cuda.is_available())
    if torch.cuda.is_available():
        print("gpu:", torch.cuda.get_device_name(0))

    # load tokenizer
    tok = AutoTokenizer.from_pretrained(args.model_dir, local_files_only=True, use_fast=True)
    # load model
    max_memory = {0: "60GiB", 1: "60GiB", 2: "60GiB", 3: "60GiB"}  # headroom to avoid OOM spikes

    model = AutoModelForCausalLM.from_pretrained(
        args.model_dir,
        local_files_only=True,
        torch_dtype=torch.bfloat16,      # H100 loves BF16 (often as good as FP16, sometimes more stable)
        device_map="auto",
        max_memory=max_memory,
        low_cpu_mem_usage=True,
        attn_implementation="flash_attention_2",  # if installed; biggest help for long-context prefill
    )
    model.eval()
    # tokenize
    inputs = tok(args.prompt, return_tensors="pt")
    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    # infer
    with torch.inference_mode():
        out = model.generate(
            **inputs,
            max_new_tokens=args.max_new_tokens,
            do_sample=False,
            use_cache=True,   # keep KV cache (default, but better to be explicit)
        )

    print(tok.decode(out[0], skip_special_tokens=True))

if __name__ == "__main__":
    main()
