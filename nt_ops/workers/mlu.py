from __future__ import annotations

import importlib

from nt_ops.runtime import REQUIRED_PROCESS_SCOPE, get_capability_report, install


def _resolve_worker_cls(qualname: str) -> type:
    module_name, _, attr_path = qualname.rpartition(".")
    if not module_name or not attr_path:
        raise ValueError(f"Invalid qualified name: {qualname!r}")

    obj = importlib.import_module(module_name)
    for attr in attr_path.split("."):
        obj = getattr(obj, attr)

    if not isinstance(obj, type):
        raise TypeError(f"Resolved object is not a class: {qualname!r}")
    return obj


class _NTOpsWorkerMixin:
    def __init__(self, *args, **kwargs):
        install(process_scope=REQUIRED_PROCESS_SCOPE)
        super().__init__(*args, **kwargs)

    def get_nt_ops_report(self) -> dict[str, object]:
        return get_capability_report()


class _LazyWorkerMeta(type):
    def __call__(cls, *args, **kwargs):
        impl_cls = cls._get_impl_cls()
        return impl_cls(*args, **kwargs)


class _LazyWorkerBase(metaclass=_LazyWorkerMeta):
    _worker_qualname: str
    _impl_cls: type | None = None

    @classmethod
    def _get_impl_cls(cls) -> type:
        if cls._impl_cls is None:
            cls._impl_cls = type(
                f"_{cls.__name__}Impl",
                (_NTOpsWorkerMixin, _resolve_worker_cls(cls._worker_qualname)),
                {},
            )
        return cls._impl_cls


class NTOpsMLUV1Worker(_LazyWorkerBase):
    _worker_qualname = "vllm_mlu.v1.worker.gpu_worker.MLUWorker"


class NTOpsMLUV0Worker(_LazyWorkerBase):
    _worker_qualname = "vllm_mlu.worker.worker.MLUWorker"
