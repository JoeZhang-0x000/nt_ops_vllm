# NT Ops for vLLM

## Current Scope

This repository currently targets a narrow first phase:

- single-process, single-GPU CUDA execution
- Qwen3 minimal dense path
- NT-backed `RMSNorm`, `RoPE`, and `SiluAndMul`
- vLLM fallback for `attention`, general `GEMM`, `lm_head`, `embedding`, `logits_processor`, and `sampler`

The runtime profile is process-wide. It is only intended for an exclusive Qwen3 process.

## Quick Start

### 1. Install dependencies

Install local editable copies of:

```bash
cd ninetoothed && pip install -e .
cd ../ntops && pip install -e .
cd ../vllm && pip install -e .
cd ../nt_ops_vllm && pip install -e .
```

### 2. Run the basic example

Set the model path explicitly and run the example:

```bash
export NT_OPS_VLLM_MODEL_PATH=/path/to/Qwen3-0.6B
python examples/basic.py
```

The example calls:

```python
nt_ops.install(process_scope="exclusive_qwen3")
```

before creating the `LLM` instance.

## Runtime Behavior

`nt_ops.install()` applies a process-wide patch profile. The default profile is `qwen3_minimal_dense`.

You can inspect the active runtime state and capability report:

```python
import nt_ops

state = nt_ops.get_runtime_state()
report = nt_ops.get_capability_report()
```

The capability report distinguishes:

- `enabled`: components currently patched to NT-backed implementations
- `fallback`: components intentionally left on vLLM
- `disabled`: components explicitly out of scope for the active profile
- `hits`: runtime hit counters for patched components

## Debugging

After a successful run, the capability report should show hit counts for:

- `rms_norm`
- `fused_add_rms_norm`
- `rope`
- `silu_and_mul`

If those hit counters stay at zero, the process did not execute the intended NT-backed Qwen3 path.
