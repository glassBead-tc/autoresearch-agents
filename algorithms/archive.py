"""
Archive data structures for open-endedness algorithms.

Stores agent variants with their evaluation scores, behavioral descriptors,
and source code for retrieval and selection by search algorithms.
"""

from __future__ import annotations

import json
import math
import random
from dataclasses import dataclass, field, asdict
from pathlib import Path


@dataclass
class AgentVariant:
    """A snapshot of an agent.py implementation with its evaluation results."""

    code: str
    scores: dict = field(default_factory=dict)  # overall, correctness, helpfulness, tool_usage
    descriptors: dict = field(default_factory=dict)  # behavioral descriptor values
    iteration: int = 0
    parent_iteration: int | None = None
    description: str = ""
    commit_hash: str = ""

    @property
    def fitness(self) -> float:
        return self.scores.get("overall_score", 0.0)


# ---------------------------------------------------------------------------
# Grid archive for MAP-Elites
# ---------------------------------------------------------------------------


class GridArchive:
    """
    2D grid archive indexed by behavioral descriptors.

    Each cell stores the highest-fitness variant observed for that
    combination of descriptor values.
    """

    def __init__(
        self,
        dims: list[str],
        ranges: list[tuple[float, float]],
        resolutions: list[int],
    ):
        self.dims = dims
        self.ranges = ranges
        self.resolutions = resolutions
        self.grid: dict[tuple[int, ...], AgentVariant] = {}

    def _to_index(self, descriptors: dict) -> tuple[int, ...]:
        idx = []
        for dim, (lo, hi), res in zip(self.dims, self.ranges, self.resolutions):
            val = descriptors.get(dim, lo)
            val = max(lo, min(hi, val))
            # Map value to bin index
            bin_idx = int((val - lo) / (hi - lo + 1e-9) * res)
            bin_idx = max(0, min(res - 1, bin_idx))
            idx.append(bin_idx)
        return tuple(idx)

    def add(self, variant: AgentVariant) -> bool:
        """Add variant to archive. Returns True if it was placed (new cell or better fitness)."""
        idx = self._to_index(variant.descriptors)
        existing = self.grid.get(idx)
        if existing is None or variant.fitness > existing.fitness:
            self.grid[idx] = variant
            return True
        return False

    def sample(self) -> AgentVariant | None:
        """Sample a random variant from occupied cells."""
        if not self.grid:
            return None
        return random.choice(list(self.grid.values()))

    @property
    def size(self) -> int:
        return len(self.grid)

    @property
    def coverage(self) -> float:
        total = 1
        for r in self.resolutions:
            total *= r
        return self.size / total

    @property
    def best(self) -> AgentVariant | None:
        if not self.grid:
            return None
        return max(self.grid.values(), key=lambda v: v.fitness)

    def summary(self) -> dict:
        if not self.grid:
            return {"size": 0, "coverage": 0.0, "best_fitness": 0.0}
        fitnesses = [v.fitness for v in self.grid.values()]
        return {
            "size": self.size,
            "coverage": self.coverage,
            "best_fitness": max(fitnesses),
            "mean_fitness": sum(fitnesses) / len(fitnesses),
        }

    def to_dict(self) -> dict:
        return {
            "dims": self.dims,
            "ranges": self.ranges,
            "resolutions": self.resolutions,
            "grid": {
                str(k): asdict(v) for k, v in self.grid.items()
            },
        }

    @classmethod
    def from_dict(cls, data: dict) -> GridArchive:
        archive = cls(data["dims"], [tuple(r) for r in data["ranges"]], data["resolutions"])
        for k_str, v_data in data["grid"].items():
            idx = tuple(json.loads(k_str.replace("(", "[").replace(")", "]")))
            archive.grid[idx] = AgentVariant(**v_data)
        return archive


# ---------------------------------------------------------------------------
# Unstructured archive for Novelty Search / ADAS / Go-Explore
# ---------------------------------------------------------------------------


class UnstructuredArchive:
    """
    A growing list of agent variants with optional novelty scoring.
    Used by Novelty Search, ADAS, and Go-Explore.
    """

    def __init__(self, max_size: int = 500):
        self.variants: list[AgentVariant] = []
        self.max_size = max_size

    def add(self, variant: AgentVariant) -> None:
        self.variants.append(variant)
        if len(self.variants) > self.max_size:
            # Remove lowest-fitness variant
            self.variants.sort(key=lambda v: v.fitness, reverse=True)
            self.variants = self.variants[: self.max_size]

    def sample(self, n: int = 1) -> list[AgentVariant]:
        if not self.variants:
            return []
        return random.sample(self.variants, min(n, len(self.variants)))

    def sample_weighted_by_novelty(self, novelty_scores: list[float]) -> AgentVariant | None:
        """Sample with probability proportional to novelty scores."""
        if not self.variants:
            return None
        total = sum(novelty_scores)
        if total == 0:
            return random.choice(self.variants)
        probs = [s / total for s in novelty_scores]
        return random.choices(self.variants, weights=probs, k=1)[0]

    @property
    def size(self) -> int:
        return len(self.variants)

    @property
    def best(self) -> AgentVariant | None:
        if not self.variants:
            return None
        return max(self.variants, key=lambda v: v.fitness)

    def summary(self) -> dict:
        if not self.variants:
            return {"size": 0, "best_fitness": 0.0}
        fitnesses = [v.fitness for v in self.variants]
        return {
            "size": self.size,
            "best_fitness": max(fitnesses),
            "mean_fitness": sum(fitnesses) / len(fitnesses),
        }

    def to_dict(self) -> dict:
        return {
            "max_size": self.max_size,
            "variants": [asdict(v) for v in self.variants],
        }

    @classmethod
    def from_dict(cls, data: dict) -> UnstructuredArchive:
        archive = cls(max_size=data.get("max_size", 500))
        archive.variants = [AgentVariant(**v) for v in data["variants"]]
        return archive


def behavioral_distance(a: dict, b: dict, keys: list[str] | None = None) -> float:
    """Euclidean distance between two behavioral descriptor dicts."""
    keys = keys or list(set(a.keys()) & set(b.keys()))
    if not keys:
        return 0.0
    return math.sqrt(sum((a.get(k, 0) - b.get(k, 0)) ** 2 for k in keys))
