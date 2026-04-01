# NT Ops for vLLM

## TODO

* [x] [Correctness](doc/correctness.md)
* [ ] [Flexibility](doc/cross_platform.md)
* [ ] [Performance](doc/performance.md)

## Project Introduction

This project aims to replace the default operators in [vLLM](https://github.com/vllm-project/vllm) with high-performance operators from [Ninetoothed](https://github.com/InfiniTensor/ninetoothed). By integrating Ninetoothed, we strive to enhance the inference efficiency and flexibility of vLLM.

For Cambricon MLU, `nt_ops` now registers a vLLM platform plugin and an nt-ops-aware MLU worker path. The primary integration path is normal vLLM MLU selection, not manual `worker_cls="nt_ops.worker.NTVLLMWorker"` injection.

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

### 3. Install vLLM-MLU

Install the MLU backend that provides the underlying MLU platform/runtime:

```bash
git clone https://github.com/Cambricon/vllm-mlu
cd vllm-mlu
pip install -e .
```


### 4. Install NT Ops for vLLM

Now, clone and install this library:

```bash
git clone git@github.com:JoeZhang-0x000/nt_ops_vllm.git
cd nt_ops_vllm
pip install -e .
```

> **Note:** This repo vendors the `ntops` operator package directly. If you previously installed the upstream `ntops` package separately, uninstall it first to avoid namespace conflicts:
> ```bash
> pip uninstall ntops
> pip install -e .
> ```


### 5. Run Example

Finally, run the example to verify the installation:

```bash
VLLM_PLUGINS=nt_ops_mlu,nt_ops_mlu_hijack VLLM_ATTENTION_BACKEND=TRITON_ATTN python examples/basic.py --model /path/to/model
```

The example no longer passes a custom `worker_cls`. When selecting the nt-ops MLU platform plugin explicitly, include both `nt_ops_mlu` and `nt_ops_mlu_hijack` in `VLLM_PLUGINS`: the first activates the nt-ops-aware MLU worker path, and the second preserves the upstream `vllm-mlu` hijack/spawn behavior required for MLU startup.

## Debugging

To facilitate debugging and verification, we provide highlighted INFO logs. After running the example above, check your console output for the following messages:
```
(EngineCore_DP0 pid=3127755) [2025-12-10 15:54:29] INFO rms.py:325: NT RMS is enabled.
(EngineCore_DP0 pid=3127755) [2025-12-10 15:54:29] INFO linear.py:156: NT GEMM is enabled.
(EngineCore_DP0 pid=3127755) [2025-12-10 15:54:29] INFO activation.py:67: NT SILU AND MUL is enabled.
```
If you see these logs, it indicates that the Ninetoothed operators have been successfully enabled and are replacing the default vLLM operators.

When debugging fused MLU backends, treat `status=installed` and `exercised=[...]` as different signals: a patch can install successfully but still remain unexercised on a backend that fuses the operator internally.
