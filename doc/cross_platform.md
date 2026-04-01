## Cross-Platform Support
This project explores how **NT Ops** can be integrated into vLLM across hardware backends. Today the concrete, supported focus is the thin nt-ops-enabled MLU path; broader cross-device coverage remains exploratory rather than seamless.
Supported devices include:
- [ ] **CPU** (x86/ARM)
- [x] **NVIDIA GPU** (CUDA)
- [ ] **Moore Threads GPU** (MUSA) | 摩尔线程
- [ ] **Iluvatar CoreX GPU** (BI) | 天数智芯
- [ ] **MetaX GPU** (MACA) | 沐曦
- [ ] **Hygon DCU** (ROCm) ｜ 海光
- [ ] **Huawei Ascend NPU** (CANN) | 华为昇腾
- [x] **Cambricon MLU** (CNES) | 寒武纪
- [ ] **Kunlun XPU** (XTDK) | 昆仑芯

Current architectural focus is Cambricon MLU: the repo now models MLU as a thin nt-ops-enabled backend path on top of `vllm-mlu`, while broader cross-device generalization remains intentionally deferred.
