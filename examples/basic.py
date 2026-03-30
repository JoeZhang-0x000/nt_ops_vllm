# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import argparse
import os

from vllm import LLM, SamplingParams

import nt_ops

PROMPTS = [
    "Hello, my name is",
    "The president of the United States is",
    "The capital of France is",
    "The future of AI is",
]

SAMPLING_PARAMS = SamplingParams(temperature=0.8, top_p=0.95, max_tokens=100)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        default=os.environ.get("NT_OPS_VLLM_MODEL_PATH"),
        help="Path or HF id for the Qwen3 model.",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    if not args.model:
        raise SystemExit(
            "Set NT_OPS_VLLM_MODEL_PATH or pass --model to run the basic example."
        )

    nt_ops.install(process_scope="exclusive_qwen3")
    print("NT capability report:", nt_ops.get_capability_report())

    llm = LLM(model=args.model, enforce_eager=True)
    outputs = llm.generate(PROMPTS, SAMPLING_PARAMS)

    print("\nGenerated Outputs:\n" + "-" * 60)
    for output in outputs:
        prompt = output.prompt
        generated_text = output.outputs[0].text
        print(f"Prompt:    {prompt!r}")
        print(f"Output:    {generated_text!r}")
        print("-" * 60)

    print("NT capability report:", nt_ops.get_capability_report())


if __name__ == "__main__":
    main()
