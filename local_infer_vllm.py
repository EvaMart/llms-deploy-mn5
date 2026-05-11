import os
import argparse
from vllm import LLM, SamplingParams

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_dir", required=True)
    ap.add_argument("--prompt", default="Hello from MN5 GPU job.")
    ap.add_argument("--max_new_tokens", type=int, default=30)
    ap.add_argument("--tensor_parallel_size", type=int, default=1)
    args = ap.parse_args()

    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"

    llm = LLM(
        model=args.model_dir,
        tensor_parallel_size=args.tensor_parallel_size,
        dtype="bfloat16",
        trust_remote_code=True,
    )

    sampling_params = SamplingParams(
        temperature=0.0,
        max_tokens=args.max_new_tokens,
    )

    outputs = llm.generate([args.prompt], sampling_params)

    for output in outputs:
        print(output.outputs[0].text)

if __name__ == "__main__":
    main()