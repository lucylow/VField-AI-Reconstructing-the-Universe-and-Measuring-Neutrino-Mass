"""Safe hyperparameter search scaffold.

This module deliberately does not execute code returned by an LLM. A model may suggest JSON/YAML
configuration candidates, which are schema-checked and evaluated by a fixed harness. In a research
setting, immutable test sets and quality gates must be kept outside the proposal loop.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from itertools import product

from nufield.discovery.objectives import CandidateScore, scalar_score


@dataclass(frozen=True, slots=True)
class SearchCandidate:
    bandwidth: float
    top_k: int
    feature_variant: str = "radial_pk"

    def validate(self) -> None:
        if not 0.05 <= self.bandwidth <= 10:
            raise ValueError("bandwidth outside safe declared search domain")
        if not 4 <= self.top_k <= 4096:
            raise ValueError("top_k outside safe declared search domain")
        if self.feature_variant not in {"radial_pk", "radial_pk_plus_counts"}:
            raise ValueError("unknown feature_variant")


def candidate_grid(
    bandwidths: Iterable[float], top_ks: Iterable[int], feature_variants=("radial_pk",)
) -> list[SearchCandidate]:
    candidates = []
    for bw, topk, variant in product(bandwidths, top_ks, feature_variants):
        item = SearchCandidate(float(bw), int(topk), str(variant))
        item.validate()
        candidates.append(item)
    return candidates


def select_candidate(
    results: list[tuple[SearchCandidate, CandidateScore]],
) -> tuple[SearchCandidate, CandidateScore]:
    admissible = [(c, s) for c, s in results if s.admissible()]
    if not admissible:
        raise ValueError("no candidate passed scientific quality gates")
    return min(admissible, key=lambda item: scalar_score(item[1]))
