from __future__ import annotations

import importlib


_WORKER_REWRITES = {
    "vllm_mlu.v1.worker.gpu_worker.MLUWorker": "nt_ops.workers.mlu.NTOpsMLUV1Worker",
    "vllm_mlu.worker.worker.MLUWorker": "nt_ops.workers.mlu.NTOpsMLUV0Worker",
}


def _rewrite_worker_qualname(worker_cls: str) -> str:
    return _WORKER_REWRITES.get(worker_cls, worker_cls)


def _resolve_mlu_platform_cls() -> type:
    module = importlib.import_module("vllm_mlu.platforms.mlu")
    return module.MLUPlatform


class NTOpsMLUPlatform(_resolve_mlu_platform_cls()):
    @classmethod
    def check_and_update_config(cls, vllm_config) -> None:
        super().check_and_update_config(vllm_config)

        parallel_config = vllm_config.parallel_config
        parallel_config.worker_cls = _rewrite_worker_qualname(
            parallel_config.worker_cls,
        )

        sd_worker_cls = getattr(parallel_config, "sd_worker_cls", None)
        if sd_worker_cls is not None:
            parallel_config.sd_worker_cls = _rewrite_worker_qualname(sd_worker_cls)
