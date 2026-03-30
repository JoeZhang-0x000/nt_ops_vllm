# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import argparse
import os

from vllm import LLM, SamplingParams

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
    parser.add_argument(
        "--worker-cls",
        default=os.environ.get("NT_OPS_VLLM_WORKER_CLS", "nt_ops.worker.NTVLLMWorker"),
        help="Fully qualified worker class used inside vLLM worker processes.",
    )
    parser.add_argument(
        "--base-worker-cls",
        default=os.environ.get("NT_OPS_VLLM_BASE_WORKER_CLS"),
        help=(
            "Optional fully qualified vLLM base worker class. "
            "Use this when the platform worker cannot be auto-resolved."
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    if not args.model:
        raise SystemExit(
            "Set NT_OPS_VLLM_MODEL_PATH or pass --model to run the basic example."
        )

    if args.base_worker_cls:
        os.environ["NT_OPS_VLLM_BASE_WORKER_CLS"] = args.base_worker_cls

    llm = LLM(
        model=args.model,
        enforce_eager=True,
        worker_cls=args.worker_cls,
    )
    print("NT worker reports before:", llm.collective_rpc("get_nt_ops_report"))
    outputs = llm.generate(PROMPTS, SAMPLING_PARAMS)

    print("\nGenerated Outputs:\n" + "-" * 60)
    for output in outputs:
        prompt = output.prompt
        generated_text = output.outputs[0].text
        print(f"Prompt:    {prompt!r}")
        print(f"Output:    {generated_text!r}")
        print("-" * 60)

    print("NT worker reports after:", llm.collective_rpc("get_nt_ops_report"))


if __name__ == "__main__":
    main()
