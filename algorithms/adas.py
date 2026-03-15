"""
ADAS: Automated Design of Agentic Systems.

A meta-agent iteratively designs new agent architectures by reviewing
an archive of previous designs and their performance. Unlike simple
hill-climbing, ADAS maintains a growing archive and uses the full
history to inform new designs.

Reference:
    Hu, Lu, Zhao, et al. (2024). "Automated Design of Agentic Systems."
    https://arxiv.org/abs/2408.08435

Usage:
    from algorithms.adas import ADAS

    search = ADAS(max_iterations=100)
    search.run()
"""

from __future__ import annotations

import json
import textwrap
from pathlib import Path

from algorithms.archive import AgentVariant, UnstructuredArchive
from algorithms.base import OpenEndedSearch


class ADAS(OpenEndedSearch):
    """
    Automated Design of Agentic Systems.

    Key differences from basic hill-climbing:
    1. Maintains a growing archive of ALL tried designs (not just the best)
    2. The meta-agent sees the full archive when proposing new designs
    3. Explicitly asks the meta-agent to design novel architectures, not just
       tweak prompts — encourage framework changes, new tool patterns, etc.
    4. Uses structured reflection on what worked and what didn't
    """

    def __init__(
        self,
        *,
        archive_size: int = 100,
        top_k_context: int = 10,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.archive = UnstructuredArchive(max_size=archive_size)
        self.top_k_context = top_k_context
        # Track ALL results (not just archived) for the meta-agent's reflection
        self.history: list[dict] = []

    def select_parent(self) -> AgentVariant:
        """Select the best-performing variant as the starting point."""
        best = self.archive.best
        if best is None:
            raise RuntimeError("Archive is empty — run baseline first")
        return best

    def update_archive(self, variant: AgentVariant) -> bool:
        """ADAS always adds to archive (up to max_size). Returns True."""
        self.archive.add(variant)
        self.history.append({
            "iteration": variant.iteration,
            "fitness": variant.fitness,
            "descriptors": variant.descriptors,
            "description": variant.description,
        })
        return True  # ADAS never rejects — it learns from failures too

    def archive_summary(self) -> dict:
        return self.archive.summary()

    def _has_variants(self) -> bool:
        return self.archive.size > 0

    def _get_best(self) -> AgentVariant | None:
        return self.archive.best

    def _get_archive_context_for_mutation(self) -> str:
        """Provide rich archive context — the key differentiator of ADAS."""
        if not self.history:
            return ""

        # Sort history by fitness to show best designs first
        sorted_history = sorted(self.history, key=lambda h: h["fitness"], reverse=True)
        top_k = sorted_history[: self.top_k_context]
        bottom_k = sorted_history[-min(3, len(sorted_history)) :]

        top_lines = []
        for h in top_k:
            top_lines.append(
                f"  - [{h['fitness']:.3f}] {h['description']} "
                f"(iter {h['iteration']}, descriptors={h['descriptors']})"
            )

        bottom_lines = []
        for h in bottom_k:
            bottom_lines.append(
                f"  - [{h['fitness']:.3f}] {h['description']} "
                f"(iter {h['iteration']})"
            )

        # Show top performing designs with their code
        top_variants = sorted(self.archive.variants, key=lambda v: v.fitness, reverse=True)
        code_examples = []
        for v in top_variants[:3]:
            code_examples.append(
                f"### Design (fitness={v.fitness:.3f}): {v.description}\n"
                f"```python\n{v.code[:2000]}{'...' if len(v.code) > 2000 else ''}\n```"
            )

        return textwrap.dedent(f"""\
            ## ADAS Archive — Design History

            Total designs tried: {len(self.history)}
            Archive size: {self.archive.size}

            ### Top performing designs:
            {chr(10).join(top_lines)}

            ### Lowest performing designs (learn from failures):
            {chr(10).join(bottom_lines)}

            ### Top design implementations:
            {chr(10).join(code_examples)}

            ## ADAS Meta-Agent Instructions:
            You are designing a NEW agentic system. Do NOT just tweak prompts.
            Consider fundamentally different approaches:
            - Different agent architectures (ReAct, plan-and-execute, chain-of-thought)
            - Different tool designs (more specialized tools, tool composition)
            - Different frameworks (LangGraph, plain SDK, custom routing)
            - Novel prompting strategies (few-shot, self-reflection, metacognition)
            - Different model configurations

            Learn from the history above. What patterns lead to high scores?
            What approaches consistently fail? Design something NOVEL that
            hasn't been tried yet.
        """)

    def _build_mutation_prompt(self, parent: AgentVariant) -> str:
        """ADAS uses a more structured design prompt than the base class."""
        archive_context = self._get_archive_context_for_mutation()

        return textwrap.dedent(f"""\
            You are a meta-agent in the ADAS (Automated Design of Agentic Systems) framework.
            Your job is to DESIGN a new agent system, not just tweak an existing one.

            ## Current best agent code:
            ```python
            {parent.code}
            ```

            ## Current best scores:
            {json.dumps(parent.scores, indent=2)}

            {archive_context}

            ## Constraints:
            - The agent MUST preserve the function contract: run_agent_with_tools(question: str) -> dict
              with keys "response" (str) and "tools_used" (list[str])
            - Available packages: langchain_openai, langgraph, openai, anthropic, pydantic, math, json, sys
            - Do NOT install new packages
            - The code must be a complete, runnable agent.py file

            ## Design Task:
            Analyze the archive of previous designs. Identify patterns in what works
            and what doesn't. Then design a NOVEL agent that explores an untried approach.

            Be creative and bold — try fundamentally different architectures, not just
            prompt variations.

            Return your response in this exact format:
            DESCRIPTION: <one-line description of your new design>
            ```python
            <complete agent.py code>
            ```
        """)

    def save_state(self) -> None:
        state = {
            "iteration": self.iteration,
            "archive": self.archive.to_dict(),
            "history": self.history,
        }
        (self.state_dir / "adas_state.json").write_text(json.dumps(state, indent=2))

    def load_state(self) -> bool:
        path = self.state_dir / "adas_state.json"
        if not path.exists():
            return False
        try:
            state = json.loads(path.read_text())
            self.iteration = state["iteration"]
            self.archive = UnstructuredArchive.from_dict(state["archive"])
            self.history = state.get("history", [])
            print(f"Resumed ADAS from iteration {self.iteration}")
            return True
        except Exception as e:
            print(f"Failed to load state: {e}")
            return False
