# NT Ops for vLLM

## TODO

* [x] [Correctness](doc/correctness.md)
* [ ] [Flexibility](doc/cross_platform.md)
* [ ] [Performance](doc/performance.md)

## Project Introduction

This project aims to replace the default operators in [vLLM](https://github.com/vllm-project/vllm) with high-performance operators from [Ninetoothed](https://github.com/InfiniTensor/ninetoothed). By integrating Ninetoothed, we strive to enhance the inference efficiency and flexibility of vLLM.

For Cambricon MLU, `nt_ops` now registers an nt-ops-aware MLU general plugin and worker path. The phase-1 backend story is intentionally thin: `vllm-mlu` still owns the only MLU platform plugin and the underlying platform/runtime bring-up, while `nt_ops_vllm` owns worker-time nt-ops attachment, capability reporting, and a small operator matrix. The primary integration path is normal vLLM MLU selection, not manual `worker_cls="nt_ops.worker.NTVLLMWorker"` injection.

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
VLLM_TARGET_DEVICE=empty pip install -e .
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
VLLM_PLUGINS=mlu,nt_ops_mlu VLLM_WORKER_MULTIPROC_METHOD=spawn VLLM_ATTENTION_BACKEND=TRITON_ATTN python examples/basic.py --model /path/to/model
```

The example no longer passes a custom `worker_cls`. When selecting plugins explicitly, include the upstream `mlu` platform plugin and the `nt_ops_mlu` general plugin in `VLLM_PLUGINS`. `vllm-mlu` must remain the sole platform plugin owner for MLU; `nt_ops_mlu` layers its worker rewrite and hijack setup on top. Phase 1 currently treats `VLLM_WORKER_MULTIPROC_METHOD=spawn` as part of the supported MLU runtime contract.

Phase 1 does not claim standalone-backend packaging. `vllm-mlu` remains a required part of the runtime stack in this phase.

## Debugging

To facilitate debugging and verification, we provide highlighted INFO logs. After running the example above, check your console output for the following messages:
```
(EngineCore_DP0 pid=3127755) [2025-12-10 15:54:29] INFO rms.py:325: NT RMS is enabled.
(EngineCore_DP0 pid=3127755) [2025-12-10 15:54:29] INFO activation.py:67: NT SILU AND MUL is enabled.
```
If you see these logs, it indicates that the phase-1 nt-ops substitutions have been successfully enabled on the supported MLU path.

When debugging fused MLU backends, treat `status=installed` and `exercised=[...]` as different signals: a patch can install successfully but still remain unexercised on a backend that fuses the operator internally.
