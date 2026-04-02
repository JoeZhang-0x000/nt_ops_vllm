# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
vLLM throughput benchmark with nt_ops.

Measures tokens/s by running a fixed batch of prompts through the model
and timing end-to-end generation.

Usage:
    python examples/benchmark_throughput.py --model /path/to/model
    python examples/benchmark_throughput.py --model /path/to/model \\
        --input-len 128 --output-len 256 --batch-size 32 --num-iters 3
"""

import argparse
import os
import time
from typing import TYPE_CHECKING, List, Tuple

import nt_ops

if TYPE_CHECKING:
    from vllm import LLM, SamplingParams


def build_prompts(input_len: int, batch_size: int) -> List[str]:
    """Generate synthetic prompts of roughly `input_len` tokens."""
    # A single token is ~4 chars on average for English text.
    token_word = "benchmark "
    words_needed = max(1, input_len // 2)  # conservative: 2 chars/token average
    prompt = (token_word * words_needed).strip()
    return [prompt] * batch_size


def run_warmup(
    llm: "LLM", prompts: List[str], sampling_params: "SamplingParams"
) -> None:
    print("Warming up...")
    llm.generate(prompts[:1], sampling_params)


def benchmark(
    llm: "LLM",
    prompts: List[str],
    sampling_params: "SamplingParams",
    num_iters: int,
) -> Tuple[float, float]:
    """
    Returns (mean_throughput_tokens_per_sec, mean_latency_sec).
    """
    throughputs = []
    latencies = []

    for i in range(num_iters):
        t0 = time.perf_counter()
        outputs = llm.generate(prompts, sampling_params)
        elapsed = time.perf_counter() - t0

        total_output_tokens = sum(len(o.outputs[0].token_ids) for o in outputs)
        total_input_tokens = sum(len(o.prompt_token_ids or []) for o in outputs)
        total_tokens = total_input_tokens + total_output_tokens

        throughputs.append(total_tokens / elapsed)
        latencies.append(elapsed)
        print(
            f"  iter {i + 1}/{num_iters}: "
            f"{elapsed:.2f}s  "
            f"input={total_input_tokens} output={total_output_tokens} tokens  "
            f"throughput={total_tokens / elapsed:.1f} tok/s"
        )

    mean_tp = sum(throughputs) / len(throughputs)
    mean_lat = sum(latencies) / len(latencies)
    return mean_tp, mean_lat


def main() -> None:
    parser = argparse.ArgumentParser(
        description="vLLM throughput benchmark with nt_ops"
    )
    parser.add_argument("--model", required=True, help="Path or HF repo of the model")
    parser.add_argument(
        "--input-len",
        type=int,
        default=64,
        help="Approximate number of input tokens per prompt (default: 64)",
    )
    parser.add_argument(
        "--output-len",
        type=int,
        default=128,
        help="Number of output tokens to generate (default: 128)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=16,
        help="Number of prompts per batch (default: 16)",
    )
    parser.add_argument(
        "--num-iters",
        type=int,
        default=3,
        help="Number of timed iterations (default: 3)",
    )
    parser.add_argument(
        "--enforce-eager",
        dest="enforce_eager",
        action="store_true",
        help="Disable CUDA graph capture (default: True)",
    )
    parser.add_argument(
        "--no-enforce-eager",
        dest="enforce_eager",
        action="store_false",
        help="Allow graph capture when the backend supports it",
    )
    parser.set_defaults(enforce_eager=True)
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="Sampling temperature; 0 = greedy (default: 0)",
    )
    args = parser.parse_args()

    print("=" * 60)
    print("nt_ops vLLM Throughput Benchmark")
    print("=" * 60)
    print(f"  model       : {args.model}")
    print(f"  input_len   : ~{args.input_len} tokens")
    print(f"  output_len  : {args.output_len} tokens")
    print(f"  batch_size  : {args.batch_size}")
    print(f"  num_iters   : {args.num_iters}")
    print("  nt_ops      : enabled via MLU plugin stack")
    print(
        f"  multiproc   : {os.environ.get('VLLM_WORKER_MULTIPROC_METHOD', nt_ops.PHASE1_MLU_MULTIPROC_METHOD)}"
    )
    print("=" * 60)

    os.environ.setdefault(
        "VLLM_WORKER_MULTIPROC_METHOD", nt_ops.PHASE1_MLU_MULTIPROC_METHOD
    )

    from vllm import LLM, SamplingParams

    sampling_params = SamplingParams.from_optional(
        temperature=args.temperature,
        max_tokens=args.output_len,
    )

    llm = LLM(model=args.model, enforce_eager=args.enforce_eager)

    prompts = build_prompts(args.input_len, args.batch_size)

    run_warmup(llm, prompts, sampling_params)

    print(f"\nRunning {args.num_iters} timed iterations...")
    mean_tp, mean_lat = benchmark(llm, prompts, sampling_params, args.num_iters)

    print("\n" + "=" * 60)
    print("Results")
    print("=" * 60)
    print(f"  Mean throughput : {mean_tp:.1f} tokens/s")
    print(f"  Mean latency    : {mean_lat:.3f} s/batch")
    print(f"  Throughput/req  : {mean_tp / args.batch_size:.1f} tokens/s/req")

    report = nt_ops.get_vllm_capability_report(llm)
    print(f"\nnt_ops capability report:")
    print(f"  profile   : {report.get('profile')}")
    print(f"  status    : {report.get('status')}")
    print(f"  exercised : {report.get('exercised')}")
    print(f"  hits      : {report.get('hits')}")

    print("=" * 60)


if __name__ == "__main__":
    main()
