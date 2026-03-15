"""
Novelty Search for agent optimization.

Instead of (or in addition to) optimizing fitness, selects for agents that
behave DIFFERENTLY from all previously seen agents. This drives exploration
of the behavioral space and can discover stepping stones to high fitness
that a purely fitness-driven search would miss.

Reference:
    Lehman & Stanley (2011). "Abandoning Objectives: Evolution through the
    Search for Novelty Alone." Evolutionary Computation 19(2).

Usage:
    from algorithms.novelty_search import NoveltySearch

    search = NoveltySearch(
        novelty_weight=0.5,  # balance novelty vs fitness
        k_nearest=5,
        max_iterations=100,
    )
    search.run()
"""

from __future__ import annotations

import json
import random
from pathlib import Path

from algorithms.archive import AgentVariant, UnstructuredArchive, behavioral_distance
from algorithms.base import OpenEndedSearch


class NoveltySearch(OpenEndedSearch):
    """
    Novelty Search with optional fitness weighting.

    Maintains two collections:
    - population: current set of agents being evolved
    - novelty_archive: agents added based on their novelty score

    Selection pressure is based on:
        combined_score = (1 - novelty_weight) * fitness + novelty_weight * novelty

    When novelty_weight=1.0, this is pure novelty search.
    When novelty_weight=0.0, this degrades to fitness-only selection.
    """

    def __init__(
        self,
        *,
        novelty_weight: float = 0.5,
        k_nearest: int = 5,
        novelty_threshold: float | None = None,
        population_size: int = 20,
        archive_size: int = 200,
        descriptor_keys: list[str] | None = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.novelty_weight = novelty_weight
        self.k_nearest = k_nearest
        self.novelty_threshold = novelty_threshold
        self.population = UnstructuredArchive(max_size=population_size)
        self.novelty_archive = UnstructuredArchive(max_size=archive_size)
        self.descriptor_keys = descriptor_keys or [
            "tool_usage_score", "correctness", "helpfulness",
            "code_lines", "num_tools", "model_tier",
        ]

    def compute_novelty(self, variant: AgentVariant) -> float:
        """
        Compute novelty as mean distance to k-nearest neighbors in
        the combined population + novelty archive.
        """
        all_variants = self.population.variants + self.novelty_archive.variants
        if not all_variants:
            return float("inf")

        distances = [
            behavioral_distance(variant.descriptors, v.descriptors, self.descriptor_keys)
            for v in all_variants
        ]
        distances.sort()
        k = min(self.k_nearest, len(distances))
        return sum(distances[:k]) / k if k > 0 else 0.0

    def select_parent(self) -> AgentVariant:
        """Select parent using novelty-weighted tournament selection."""
        if self.population.size == 0:
            raise RuntimeError("Population is empty — run baseline first")

        # Tournament selection with novelty scoring
        tournament_size = min(3, self.population.size)
        candidates = random.sample(self.population.variants, tournament_size)

        best = None
        best_combined = -float("inf")
        for c in candidates:
            novelty = self.compute_novelty(c)
            combined = (1 - self.novelty_weight) * c.fitness + self.novelty_weight * novelty
            if combined > best_combined:
                best_combined = combined
                best = c

        return best

    def update_archive(self, variant: AgentVariant) -> bool:
        """Add to population; also add to novelty archive if novel enough."""
        self.population.add(variant)

        novelty = self.compute_novelty(variant)
        threshold = self.novelty_threshold
        if threshold is None:
            # Adaptive threshold: use median novelty of archive
            if self.novelty_archive.size > 0:
                archive_novelties = [
                    self.compute_novelty(v) for v in self.novelty_archive.variants
                ]
                threshold = sorted(archive_novelties)[len(archive_novelties) // 2]
            else:
                threshold = 0.0

        if novelty >= threshold:
            self.novelty_archive.add(variant)
            return True

        return variant.fitness > 0  # Accept any non-crashed variant

    def archive_summary(self) -> dict:
        pop_summary = self.population.summary()
        nov_summary = self.novelty_archive.summary()
        return {
            "population_size": pop_summary["size"],
            "novelty_archive_size": nov_summary["size"],
            "best_fitness": pop_summary.get("best_fitness", 0),
            "novelty_weight": self.novelty_weight,
        }

    def _has_variants(self) -> bool:
        return self.population.size > 0

    def _get_best(self) -> AgentVariant | None:
        # Check both population and novelty archive for the best
        candidates = []
        if self.population.best:
            candidates.append(self.population.best)
        if self.novelty_archive.best:
            candidates.append(self.novelty_archive.best)
        if not candidates:
            return None
        return max(candidates, key=lambda v: v.fitness)

    def _get_archive_context_for_mutation(self) -> str:
        if self.population.size == 0:
            return ""

        # Show diverse samples from the population
        samples = self.population.sample(min(5, self.population.size))
        sample_lines = []
        for v in samples:
            novelty = self.compute_novelty(v)
            sample_lines.append(
                f"  - fitness={v.fitness:.3f}, novelty={novelty:.3f}, "
                f"descriptors={v.descriptors}, desc=\"{v.description}\""
            )

        return (
            f"## Novelty Search Status:\n"
            f"Population size: {self.population.size}\n"
            f"Novelty archive size: {self.novelty_archive.size}\n"
            f"Novelty weight: {self.novelty_weight}\n\n"
            f"Sample population members:\n" + "\n".join(sample_lines) + "\n\n"
            f"## Goal: Create an agent that behaves DIFFERENTLY from existing ones.\n"
            f"Novelty is measured by behavioral distance in descriptor space:\n"
            f"  {self.descriptor_keys}\n"
            f"Try approaches that would score differently on these dimensions —\n"
            f"different tool counts, different code styles, different model tiers, etc."
        )

    def save_state(self) -> None:
        state = {
            "iteration": self.iteration,
            "novelty_weight": self.novelty_weight,
            "k_nearest": self.k_nearest,
            "novelty_threshold": self.novelty_threshold,
            "descriptor_keys": self.descriptor_keys,
            "population": self.population.to_dict(),
            "novelty_archive": self.novelty_archive.to_dict(),
        }
        (self.state_dir / "novelty_search_state.json").write_text(json.dumps(state, indent=2))

    def load_state(self) -> bool:
        path = self.state_dir / "novelty_search_state.json"
        if not path.exists():
            return False
        try:
            state = json.loads(path.read_text())
            self.iteration = state["iteration"]
            self.novelty_weight = state.get("novelty_weight", self.novelty_weight)
            self.k_nearest = state.get("k_nearest", self.k_nearest)
            self.novelty_threshold = state.get("novelty_threshold", self.novelty_threshold)
            self.descriptor_keys = state.get("descriptor_keys", self.descriptor_keys)
            self.population = UnstructuredArchive.from_dict(state["population"])
            self.novelty_archive = UnstructuredArchive.from_dict(state["novelty_archive"])
            print(f"Resumed Novelty Search from iteration {self.iteration}")
            return True
        except Exception as e:
            print(f"Failed to load state: {e}")
            return False
