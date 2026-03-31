# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import argparse

from vllm import LLM, SamplingParams
import nt_ops
from nt_ops.worker import NTVLLMWorker

PROMPTS = [
    "Hello, my name is",
    "The president of the United States is",
    "The capital of France is",
    "The future of AI is",
]


def main():
    parser = argparse.ArgumentParser(description="Basic nt_ops + vLLM generation example")
    parser.add_argument("--model", required=True, help="Path or HF repo of the model")
    parser.add_argument("--max-tokens", type=int, default=100)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--enforce-eager", action="store_true", default=True)
    args = parser.parse_args()

    sampling_params = SamplingParams(
        temperature=args.temperature,
        top_p=args.top_p,
        max_tokens=args.max_tokens,
    )

    llm = LLM(model=args.model, enforce_eager=args.enforce_eager,
              worker_cls=NTVLLMWorker)
    outputs = llm.generate(PROMPTS, sampling_params)

    print("\nGenerated Outputs:\n" + "-" * 60)
    for output in outputs:
        prompt = output.prompt
        generated_text = output.outputs[0].text
        print(f"Prompt:    {prompt!r}")
        print(f"Output:    {generated_text!r}")
        print("-" * 60)


if __name__ == "__main__":
    main()
