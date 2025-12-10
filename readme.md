# NT Ops for vLLM

## TODO
[x] [Correctness](nt_ops/doc/correctness.md)

## Project Introduction

This project aims to replace the default operators in [vLLM](https://github.com/vllm-project/vllm) with high-performance operators from [Ninetoothed](https://github.com/InfiniTensor/ninetoothed). By integrating Ninetoothed, we strive to enhance the inference efficiency and flexibility of vLLM.

## Quick Start

Follow the steps below to set up the environment and run the example.

### 1. Install Ninetoothed

First, clone and install the Ninetoothed library:

```bash
git clone https://github.com/InfiniTensor/ninetoothed.git
cd ninetoothed
pip install -e .
```


### 2. Install vLLM

Next, clone and install the vLLM library:

```bash
git clone https://github.com/vllm-project/vllm.git
cd vllm
pip install -e .
```


### 3. Install NT Ops for vLLM

Now, clone and install this library:

```bash
git clone git@github.com:JoeZhang-0x000/nt_ops_vllm.git
cd nt_ops_vllm
pip install -e .
```


### 4. Run Example

Finally, run the example to verify the installation:

```bash
VLLM_ATTENTION_BACKEND=TRITON_ATTN python examples/basic.py
```

## Debugging

To facilitate debugging and verification, we provide highlighted INFO logs. After running the example above, check your console output for the following messages:
```
(EngineCore_DP0 pid=3127755) [2025-12-10 15:54:29] INFO rms.py:325: NT RMS is enabled.
(EngineCore_DP0 pid=3127755) [2025-12-10 15:54:29] INFO linear.py:156: NT GEMM is enabled.
(EngineCore_DP0 pid=3127755) [2025-12-10 15:54:29] INFO activation.py:67: NT SILU AND MUL is enabled.
```
If you see these logs, it indicates that the Ninetoothed operators have been successfully enabled and are replacing the default vLLM operators.