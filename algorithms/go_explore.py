"""
Go-Explore for agent optimization.

Maintains an archive of agent "cells" (behavioral niches). At each step,
selects a promising but under-explored cell, returns to it (restores that
agent code), and explores from there. This avoids the "detachment" problem
where search drifts away from promising areas.

Reference:
    Ecoffet, Huizinga, Lehman, Stanley, Clune (2021).
    "First return, then explore." Nature 590.

Usage:
    from algorithms.go_explore import GoExplore

    search = GoExplore(max_iterations=100)
    search.run()
"""

from __future__ import annotations

import json
import math
import random
from pathlib import Path

from algorithms.archive import AgentVariant, UnstructuredArchive
from algorithms.base import OpenEndedSearch


class Cell:
    """A behavioral cell in the Go-Explore archive."""

    def __init__(self, key: tuple, variant: AgentVariant):
        self.key = key
        self.variant = variant
        self.visit_count = 1
        self.best_fitness = variant.fitness

    def update(self, variant: AgentVariant) -> bool:
        """Update cell if new variant is better. Returns True if updated."""
        self.visit_count += 1
        if variant.fitness > self.best_fitness:
            self.best_fitness = variant.fitness
            self.variant = variant
            return True
        return False


class GoExplore(OpenEndedSearch):
    """
    Go-Explore: First return, then explore.

    Phase 1 (Explore): Select a cell, return to its agent code, and explore
    by mutating from there. The cell selection balances:
    - Curiosity: prefer less-visited cells
    - Quality: prefer cells with higher fitness
    - Recency: slight bonus for recently discovered cells

    Phase 2 (Robustify): Not applicable to deterministic agent code — the
    code either works or it doesn't, so we skip the robustification phase.
    """

    def __init__(
        self,
        *,
        cell_dims: list[str] | None = None,
        cell_resolution: int = 5,
        curiosity_weight: float = 1.0,
        quality_weight: float = 1.0,
        archive_size: int = 500,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.cell_dims = cell_dims or [
            "tool_usage_score", "correctness", "num_tools",
        ]
        self.cell_resolution = cell_resolution
        self.curiosity_weight = curiosity_weight
        self.quality_weight = quality_weight
        self.cells: dict[tuple, Cell] = {}
        # Also keep a flat archive of all variants for context
        self.all_variants = UnstructuredArchive(max_size=archive_size)

    def _to_cell_key(self, descriptors: dict) -> tuple:
        """Discretize descriptors into a cell key."""
        parts = []
        for dim in self.cell_dims:
            val = descriptors.get(dim, 0.0)
            # Discretize to resolution bins
            if isinstance(val, (int, float)):
                bin_idx = int(val * self.cell_resolution)
                bin_idx = max(0, min(self.cell_resolution - 1, bin_idx))
                parts.append(bin_idx)
            else:
                parts.append(hash(str(val)) % self.cell_resolution)
        return tuple(parts)

    def _cell_score(self, cell: Cell) -> float:
        """Compute selection score for a cell (higher = more likely to select)."""
        # Curiosity: prefer less-visited cells (1/sqrt(visits))
        curiosity = 1.0 / math.sqrt(cell.visit_count)
        # Quality: prefer cells with higher fitness
        quality = cell.best_fitness
        return (
            self.curiosity_weight * curiosity
            + self.quality_weight * quality
        )

    def select_parent(self) -> AgentVariant:
        """Select a cell weighted by curiosity + quality, return its variant."""
        if not self.cells:
            raise RuntimeError("No cells in archive — run baseline first")

        cells = list(self.cells.values())
        scores = [self._cell_score(c) for c in cells]

        # Softmax-like selection (shift to avoid overflow)
        max_score = max(scores)
        exp_scores = [math.exp(s - max_score) for s in scores]
        total = sum(exp_scores)
        probs = [e / total for e in exp_scores]

        selected = random.choices(cells, weights=probs, k=1)[0]
        return selected.variant

    def update_archive(self, variant: AgentVariant) -> bool:
        """Place variant in its cell. Update if better than existing."""
        self.all_variants.add(variant)

        key = self._to_cell_key(variant.descriptors)
        if key in self.cells:
            return self.cells[key].update(variant)
        else:
            self.cells[key] = Cell(key, variant)
            return True  # New cell discovered

    def archive_summary(self) -> dict:
        if not self.cells:
            return {"num_cells": 0, "best_fitness": 0.0}
        fitnesses = [c.best_fitness for c in self.cells.values()]
        visits = [c.visit_count for c in self.cells.values()]
        return {
            "num_cells": len(self.cells),
            "best_fitness": max(fitnesses),
            "mean_fitness": sum(fitnesses) / len(fitnesses),
            "total_visits": sum(visits),
            "least_visited": min(visits),
        }

    def _has_variants(self) -> bool:
        return len(self.cells) > 0

    def _get_best(self) -> AgentVariant | None:
        if not self.cells:
            return None
        best_cell = max(self.cells.values(), key=lambda c: c.best_fitness)
        return best_cell.variant

    def _get_archive_context_for_mutation(self) -> str:
        if not self.cells:
            return ""

        cells_by_score = sorted(self.cells.values(), key=lambda c: self._cell_score(c), reverse=True)

        cell_lines = []
        for c in cells_by_score[:10]:
            desc = {d: c.variant.descriptors.get(d, "?") for d in self.cell_dims}
            cell_lines.append(
                f"  Cell {c.key}: fitness={c.best_fitness:.3f}, "
                f"visits={c.visit_count}, descriptors={desc}, "
                f"desc=\"{c.variant.description}\""
            )

        # Identify least-visited cells
        least_visited = sorted(self.cells.values(), key=lambda c: c.visit_count)[:3]
        least_lines = [
            f"  Cell {c.key}: visits={c.visit_count}, fitness={c.best_fitness:.3f}"
            for c in least_visited
        ]

        return (
            f"## Go-Explore Archive Status:\n"
            f"Total cells discovered: {len(self.cells)}\n"
            f"Cell dimensions: {self.cell_dims}\n\n"
            f"Top cells (by selection score):\n" + "\n".join(cell_lines) + "\n\n"
            f"Least-visited cells (exploration opportunity):\n" + "\n".join(least_lines) + "\n\n"
            f"## Goal: Explore from the selected parent cell.\n"
            f"Try modifications that could either:\n"
            f"1. Improve the fitness of the current cell\n"
            f"2. Discover a NEW cell (different behavioral niche)\n"
            f"The system rewards both quality and discovering new cells."
        )

    def save_state(self) -> None:
        cells_data = {}
        for key, cell in self.cells.items():
            from dataclasses import asdict
            cells_data[str(key)] = {
                "key": list(key),
                "variant": asdict(cell.variant),
                "visit_count": cell.visit_count,
                "best_fitness": cell.best_fitness,
            }
        state = {
            "iteration": self.iteration,
            "cell_dims": self.cell_dims,
            "cell_resolution": self.cell_resolution,
            "cells": cells_data,
            "all_variants": self.all_variants.to_dict(),
        }
        (self.state_dir / "go_explore_state.json").write_text(json.dumps(state, indent=2))

    def load_state(self) -> bool:
        path = self.state_dir / "go_explore_state.json"
        if not path.exists():
            return False
        try:
            state = json.loads(path.read_text())
            self.iteration = state["iteration"]
            self.cell_dims = state.get("cell_dims", self.cell_dims)
            self.cell_resolution = state.get("cell_resolution", self.cell_resolution)
            self.all_variants = UnstructuredArchive.from_dict(state["all_variants"])

            self.cells = {}
            for key_str, cell_data in state["cells"].items():
                key = tuple(cell_data["key"])
                variant = AgentVariant(**cell_data["variant"])
                cell = Cell(key, variant)
                cell.visit_count = cell_data["visit_count"]
                cell.best_fitness = cell_data["best_fitness"]
                self.cells[key] = cell

            print(f"Resumed Go-Explore from iteration {self.iteration}")
            return True
        except Exception as e:
            print(f"Failed to load state: {e}")
            return False
