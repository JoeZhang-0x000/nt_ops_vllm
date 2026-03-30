from __future__ import annotations

from collections import Counter
from dataclasses import dataclass


@dataclass(frozen=True)
class CapabilityProfile:
    name: str
    enabled: tuple[str, ...]
    fallback: tuple[str, ...]
    disabled: tuple[str, ...]


_HITS: Counter[str] = Counter()


def record_hit(component: str) -> None:
    _HITS[component] += 1


def reset_hits() -> None:
    _HITS.clear()


def snapshot_hits() -> dict[str, int]:
    return dict(_HITS)


def build_capability_report(
    profile: CapabilityProfile,
    *,
    status: str,
    errors: tuple[str, ...] = (),
) -> dict[str, object]:
    return {
        "profile": profile.name,
        "status": status,
        "enabled": list(profile.enabled),
        "fallback": list(profile.fallback),
        "disabled": list(profile.disabled),
        "hits": snapshot_hits(),
        "errors": list(errors),
    }


def get_capability_report(profile: CapabilityProfile | None = None) -> dict[str, object]:
    if profile is None:
        return {"hits": snapshot_hits()}
    return build_capability_report(profile, status="unknown")
