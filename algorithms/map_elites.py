"""
MAP-Elites: Quality-Diversity search for agent optimization.

Maintains a grid archive where each cell corresponds to a unique combination
of behavioral descriptors. Each cell stores the highest-fitness agent observed
for that behavioral niche.

Reference:
    Mouret & Clune (2015). "Illuminating search spaces by mapping elites."
    https://arxiv.org/abs/1504.04909

Usage:
    from algorithms.map_elites import MAPElites

    search = MAPElites(
        dims=["tool_usage_score", "prompt_length"],
        ranges=[(0.0, 1.0), (0, 2000)],
        resolutions=[5, 5],
        max_iterations=100,
    )
    search.run()
"""

from __future__ import annotations

import json
from pathlib import Path

from algorithms.archive import AgentVariant, GridArchive
from algorithms.base import OpenEndedSearch


class MAPElites(OpenEndedSearch):
    """
    MAP-Elites illumination algorithm.

    Discovers a diverse collection of high-performing agent variants
    spanning different behavioral niches (e.g., different tool usage
    patterns, prompt styles, model choices).
    """

    def __init__(
        self,
        *,
        dims: list[str] | None = None,
        ranges: list[tuple[float, float]] | None = None,
        resolutions: list[int] | None = None,
        **kwargs,
    ):
        super().__init__(**kwargs)

        # Default: 2D grid over tool_usage_score × code complexity
        self.archive = GridArchive(
            dims=dims or ["tool_usage_score", "correctness"],
            ranges=ranges or [(0.0, 1.0), (0.0, 1.0)],
            resolutions=resolutions or [5, 5],
        )

    def select_parent(self) -> AgentVariant:
        variant = self.archive.sample()
        if variant is None:
            raise RuntimeError("Archive is empty — run baseline first")
        return variant

    def update_archive(self, variant: AgentVariant) -> bool:
        return self.archive.add(variant)

    def archive_summary(self) -> dict:
        return self.archive.summary()

    def _has_variants(self) -> bool:
        return self.archive.size > 0

    def _get_best(self) -> AgentVariant | None:
        return self.archive.best

    def _get_archive_context_for_mutation(self) -> str:
        """Provide MAP-Elites archive context to guide exploration."""
        if self.archive.size == 0:
            return ""

        # Show the archive grid occupancy and which niches are unexplored
        occupied = []
        for idx, v in self.archive.grid.items():
            desc_vals = {d: v.descriptors.get(d, "?") for d in self.archive.dims}
            occupied.append(f"  Cell {idx}: fitness={v.fitness:.3f}, descriptors={desc_vals}")

        total_cells = 1
        for r in self.archive.resolutions:
            total_cells *= r

        return (
            f"## MAP-Elites Archive Status:\n"
            f"Coverage: {self.archive.size}/{total_cells} cells filled "
            f"({self.archive.coverage:.1%})\n"
            f"Dimensions: {self.archive.dims}\n"
            f"Ranges: {self.archive.ranges}\n\n"
            f"Occupied cells:\n" + "\n".join(occupied) + "\n\n"
            f"## Goal: Try to fill EMPTY cells by exploring different behavioral niches.\n"
            f"For example, vary the number/type of tools, prompt style, model choice, etc.\n"
            f"to land in a different region of the descriptor space."
        )

    def save_state(self) -> None:
        state = {
            "iteration": self.iteration,
            "archive": self.archive.to_dict(),
        }
        (self.state_dir / "map_elites_state.json").write_text(json.dumps(state, indent=2))

    def load_state(self) -> bool:
        path = self.state_dir / "map_elites_state.json"
        if not path.exists():
            return False
        try:
            state = json.loads(path.read_text())
            self.iteration = state["iteration"]
            self.archive = GridArchive.from_dict(state["archive"])
            print(f"Resumed MAP-Elites from iteration {self.iteration}")
            return True
        except Exception as e:
            print(f"Failed to load state: {e}")
            return False
