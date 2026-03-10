"""
Base class for open-endedness search algorithms.

Provides shared infrastructure for:
- Evaluating agent variants via run_eval.py
- Extracting behavioral descriptors from code and eval results
- Using an LLM to propose mutations to agent code
- Git operations for tracking variants
- State persistence and logging
"""

from __future__ import annotations

import ast
import json
import os
import re
import subprocess
import sys
import textwrap
import time
from abc import ABC, abstractmethod
from pathlib import Path

from algorithms.archive import AgentVariant


class OpenEndedSearch(ABC):
    """Base class for open-ended agent search algorithms."""

    def __init__(
        self,
        *,
        agent_path: str = "agent.py",
        eval_cmd: str = "python run_eval.py",
        state_dir: str = "oe_state",
        log_file: str = "oe_results.tsv",
        mutator_model: str = "gpt-4o-mini",
        mutator_provider: str = "openai",
        max_iterations: int | None = None,
        eval_timeout: int = 600,
    ):
        self.agent_path = Path(agent_path).resolve()
        self.eval_cmd = eval_cmd
        self.state_dir = Path(state_dir)
        self.state_dir.mkdir(parents=True, exist_ok=True)
        self.log_file = Path(log_file)
        self.mutator_model = mutator_model
        self.mutator_provider = mutator_provider
        self.max_iterations = max_iterations
        self.eval_timeout = eval_timeout
        self.iteration = 0

        self._init_log()

    # ------------------------------------------------------------------
    # Abstract interface — each algorithm implements these
    # ------------------------------------------------------------------

    @abstractmethod
    def select_parent(self) -> AgentVariant:
        """Choose a variant from the archive to mutate."""

    @abstractmethod
    def update_archive(self, variant: AgentVariant) -> bool:
        """Attempt to add variant to the archive. Returns True if accepted."""

    @abstractmethod
    def archive_summary(self) -> dict:
        """Return a summary dict for logging."""

    @abstractmethod
    def save_state(self) -> None:
        """Persist algorithm state to disk."""

    @abstractmethod
    def load_state(self) -> bool:
        """Load state from disk. Returns True if state was loaded."""

    # ------------------------------------------------------------------
    # Main loop
    # ------------------------------------------------------------------

    def run(self) -> None:
        """Run the open-ended search loop."""
        self.load_state()

        # Evaluate initial agent if archive is empty
        if not self._has_variants():
            print("=== Evaluating baseline agent ===")
            baseline_code = self.agent_path.read_text()
            scores, descriptors = self.evaluate(baseline_code)
            baseline = AgentVariant(
                code=baseline_code,
                scores=scores,
                descriptors=descriptors,
                iteration=0,
                description="baseline",
            )
            self.update_archive(baseline)
            self._log_result(baseline, accepted=True)
            self.save_state()
            print(f"Baseline: {scores}")

        while True:
            self.iteration += 1
            if self.max_iterations and self.iteration > self.max_iterations:
                print(f"Reached max iterations ({self.max_iterations}). Stopping.")
                break

            print(f"\n=== Iteration {self.iteration} ===")
            summary = self.archive_summary()
            print(f"Archive: {summary}")

            # Select parent and mutate
            parent = self.select_parent()
            print(f"Selected parent (iteration {parent.iteration}, fitness {parent.fitness:.4f})")

            try:
                new_code, description = self.mutate(parent)
            except Exception as e:
                print(f"Mutation failed: {e}")
                continue

            # Evaluate
            scores, descriptors = self.evaluate(new_code)
            variant = AgentVariant(
                code=new_code,
                scores=scores,
                descriptors=descriptors,
                iteration=self.iteration,
                parent_iteration=parent.iteration,
                description=description,
            )

            # Update archive
            accepted = self.update_archive(variant)
            status = "accepted" if accepted else "rejected"
            print(f"Result: fitness={variant.fitness:.4f}, {status}")
            print(f"  Descriptors: {descriptors}")

            self._log_result(variant, accepted=accepted)
            self.save_state()

            # Restore best agent to disk
            self._write_best_to_disk()

    @abstractmethod
    def _has_variants(self) -> bool:
        """Check if the archive has any variants."""

    def _write_best_to_disk(self) -> None:
        """Write the best-known variant back to agent.py."""
        best = self._get_best()
        if best:
            self.agent_path.write_text(best.code)

    @abstractmethod
    def _get_best(self) -> AgentVariant | None:
        """Return the best variant in the archive."""

    # ------------------------------------------------------------------
    # Evaluation
    # ------------------------------------------------------------------

    def evaluate(self, code: str) -> tuple[dict, dict]:
        """
        Write code to agent.py, run eval, and return (scores, descriptors).

        scores: {overall_score, avg_correctness, avg_helpfulness, avg_tool_usage, ...}
        descriptors: behavioral descriptors extracted from code + eval results
        """
        # Write code to agent.py
        self.agent_path.write_text(code)

        # Run evaluation
        scores = self._run_eval()

        # Extract behavioral descriptors
        descriptors = self._extract_descriptors(code, scores)

        return scores, descriptors

    def _run_eval(self) -> dict:
        """Run run_eval.py and parse the output."""
        eval_log = self.state_dir / "eval.log"
        try:
            result = subprocess.run(
                self.eval_cmd.split(),
                capture_output=True,
                text=True,
                timeout=self.eval_timeout,
                cwd=self.agent_path.parent,
            )
            output = result.stdout + "\n" + result.stderr
            eval_log.write_text(output)

            if result.returncode != 0:
                print(f"Eval returned non-zero exit code: {result.returncode}")
                print(f"stderr: {result.stderr[-500:]}")
                return {"overall_score": 0.0, "crashed": True}

        except subprocess.TimeoutExpired:
            print(f"Eval timed out after {self.eval_timeout}s")
            return {"overall_score": 0.0, "timed_out": True}
        except Exception as e:
            print(f"Eval error: {e}")
            return {"overall_score": 0.0, "error": str(e)}

        return self._parse_eval_output(output)

    def _parse_eval_output(self, output: str) -> dict:
        """Parse the --- delimited eval output."""
        scores = {}
        in_summary = False
        for line in output.splitlines():
            if line.strip() == "---":
                in_summary = True
                continue
            if in_summary and ":" in line:
                key, val = line.split(":", 1)
                key = key.strip()
                val = val.strip()
                try:
                    scores[key] = float(val)
                except ValueError:
                    scores[key] = val
        return scores

    # ------------------------------------------------------------------
    # Behavioral descriptor extraction
    # ------------------------------------------------------------------

    def _extract_descriptors(self, code: str, scores: dict) -> dict:
        """
        Extract behavioral descriptors from agent code and eval scores.

        Descriptors characterize *how* the agent behaves, not just how well.
        These are used by MAP-Elites for grid placement and by Novelty Search
        for novelty computation.
        """
        descriptors = {}

        # From eval scores (behavioral outcomes)
        descriptors["tool_usage_score"] = scores.get("avg_tool_usage", 0.0)
        descriptors["correctness"] = scores.get("avg_correctness", 0.0)
        descriptors["helpfulness"] = scores.get("avg_helpfulness", 0.0)

        # From code structure (design choices)
        descriptors["code_lines"] = len(code.splitlines())
        descriptors["num_tools"] = self._count_tools(code)
        descriptors["prompt_length"] = self._extract_prompt_length(code)
        descriptors["model_tier"] = self._classify_model_tier(code)

        return descriptors

    def _count_tools(self, code: str) -> int:
        """Count tool functions defined in agent code."""
        try:
            tree = ast.parse(code)
            # Count functions in TOOLS list or heuristically
            tool_count = 0
            for node in ast.walk(tree):
                if isinstance(node, ast.FunctionDef) and node.name not in (
                    "build_agent", "run_agent", "run_agent_with_tools",
                ):
                    tool_count += 1
            return tool_count
        except SyntaxError:
            return 0

    def _extract_prompt_length(self, code: str) -> int:
        """Extract approximate system prompt length."""
        match = re.search(r'SYSTEM_PROMPT\s*=\s*"""(.*?)"""', code, re.DOTALL)
        if not match:
            match = re.search(r"SYSTEM_PROMPT\s*=\s*'''(.*?)'''", code, re.DOTALL)
        if not match:
            match = re.search(r'SYSTEM_PROMPT\s*=\s*"(.*?)"', code, re.DOTALL)
        if match:
            return len(match.group(1))
        return 0

    def _classify_model_tier(self, code: str) -> float:
        """Classify model as a numeric tier for descriptor space. 0=small, 1=large."""
        code_lower = code.lower()
        if any(m in code_lower for m in ["gpt-4o\"", "gpt-4o'", "claude-3-5-sonnet", "claude-3-opus"]):
            return 1.0
        if any(m in code_lower for m in ["gpt-4o-mini", "claude-3-haiku", "claude-3-5-haiku"]):
            return 0.5
        if any(m in code_lower for m in ["gpt-3.5", "gpt-4-turbo"]):
            return 0.3
        return 0.5  # default

    # ------------------------------------------------------------------
    # Mutation via LLM
    # ------------------------------------------------------------------

    def mutate(self, parent: AgentVariant) -> tuple[str, str]:
        """
        Use an LLM to propose a mutation to the parent agent code.

        Returns (new_code, description).
        """
        mutation_prompt = self._build_mutation_prompt(parent)

        if self.mutator_provider == "openai":
            new_code, description = self._mutate_openai(mutation_prompt)
        else:
            raise ValueError(f"Unknown mutator provider: {self.mutator_provider}")

        # Validate the code parses
        try:
            ast.parse(new_code)
        except SyntaxError as e:
            raise ValueError(f"Generated code has syntax error: {e}")

        return new_code, description

    def _build_mutation_prompt(self, parent: AgentVariant) -> str:
        """Build the prompt for the mutator LLM."""
        archive_context = self._get_archive_context_for_mutation()

        return textwrap.dedent(f"""\
            You are an AI agent researcher. Your task is to modify an agent implementation
            to explore a DIFFERENT behavioral niche or improve performance.

            ## Current agent code (parent):
            ```python
            {parent.code}
            ```

            ## Parent evaluation scores:
            {json.dumps(parent.scores, indent=2)}

            ## Parent behavioral descriptors:
            {json.dumps(parent.descriptors, indent=2)}

            {archive_context}

            ## Constraints:
            - The agent MUST preserve the function contract: run_agent_with_tools(question: str) -> dict
              with keys "response" (str) and "tools_used" (list[str])
            - You can change anything: system prompt, tools, model, temperature, architecture, framework
            - The code must be a complete, runnable agent.py file
            - Available packages: langchain_openai, langgraph, openai, anthropic, pydantic, math, json, sys
            - Do NOT install new packages

            ## Instructions:
            Think about what behavioral dimension to explore or what to improve.
            Be creative — try different approaches, not just incremental prompt tweaks.

            Return your response in this exact format:
            DESCRIPTION: <one-line description of what you changed>
            ```python
            <complete agent.py code>
            ```
        """)

    def _get_archive_context_for_mutation(self) -> str:
        """Override in subclasses to provide archive-specific context to the mutator."""
        return ""

    def _mutate_openai(self, prompt: str) -> tuple[str, str]:
        """Call OpenAI to generate a mutation."""
        from openai import OpenAI

        client = OpenAI()
        response = client.chat.completions.create(
            model=self.mutator_model,
            temperature=0.9,
            messages=[
                {"role": "system", "content": "You are an expert AI agent researcher."},
                {"role": "user", "content": prompt},
            ],
            max_tokens=4096,
        )

        text = response.choices[0].message.content or ""
        return self._parse_mutation_response(text)

    def _parse_mutation_response(self, text: str) -> tuple[str, str]:
        """Parse the LLM mutation response into (code, description)."""
        # Extract description
        desc_match = re.search(r"DESCRIPTION:\s*(.+?)(?:\n|$)", text)
        description = desc_match.group(1).strip() if desc_match else "unknown mutation"

        # Extract code block
        code_match = re.search(r"```python\s*\n(.*?)```", text, re.DOTALL)
        if not code_match:
            raise ValueError("No python code block found in mutation response")

        code = code_match.group(1).strip()

        # Basic validation
        if "run_agent_with_tools" not in code:
            raise ValueError("Generated code missing run_agent_with_tools function")

        return code, description

    # ------------------------------------------------------------------
    # Logging
    # ------------------------------------------------------------------

    def _init_log(self) -> None:
        if not self.log_file.exists():
            self.log_file.write_text(
                "iteration\toverall_score\tcorrectness\thelpfulness\ttool_usage\tstatus\tdescription\tdescriptors\n"
            )

    def _log_result(self, variant: AgentVariant, accepted: bool) -> None:
        status = "accepted" if accepted else "rejected"
        scores = variant.scores
        desc_json = json.dumps(variant.descriptors)
        line = (
            f"{variant.iteration}\t"
            f"{scores.get('overall_score', 0):.6f}\t"
            f"{scores.get('avg_correctness', 0):.6f}\t"
            f"{scores.get('avg_helpfulness', 0):.6f}\t"
            f"{scores.get('avg_tool_usage', 0):.6f}\t"
            f"{status}\t"
            f"{variant.description}\t"
            f"{desc_json}\n"
        )
        with open(self.log_file, "a") as f:
            f.write(line)
