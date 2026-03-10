# Autoresearch for Agents

> Inspired by [karpathy/autoresearch](https://github.com/karpathy/autoresearch) — but instead of optimizing ML training code, we optimize **agents** using [LangSmith](https://smith.langchain.com/) observability and evals.

## The Idea

Give an AI coding agent a working agent implementation and an evaluation dataset. Let it experiment autonomously: modify the agent code, run evals, check if scores improved, keep or discard, and repeat. You wake up in the morning to a log of experiments and (hopefully) a better agent.

```
┌─────────────────────────────────────────────────────┐
│                  EXPERIMENT LOOP                     │
│                                                      │
│  1. Read agent.py + results so far                   │
│  2. Propose a change (prompt, tools, architecture)   │
│  3. Edit agent.py                                    │
│  4. git commit                                       │
│  5. Run evaluation: python run_eval.py               │
│  6. Parse scores from eval output                    │
│  7. If improved → keep commit                        │
│     If worse   → git reset  (discard)                │
│  8. Log result to results.tsv                        │
│  9. Repeat forever                                   │
│                                                      │
└─────────────────────────────────────────────────────┘
```

### Comparison with Karpathy's autoresearch

| | karpathy/autoresearch | autoresearch for agents |
|---|---|---|
| **What's optimized** | ML training code (`train.py`) | Agent code (`agent.py`) |
| **Metric** | `val_bpb` (lower is better) | Eval score (higher is better) |
| **Evaluation** | Fixed 5-min training run | LangSmith evaluation pipeline |
| **Observability** | Training logs | LangSmith traces |
| **What the agent edits** | Model architecture, optimizer, hyperparams | Prompts, tools, agent architecture |
| **What's fixed** | `prepare.py` (data, eval) | `run_eval.py` (eval harness), `dataset.json` |

## Project Structure

```
agent.py        — YOUR agent implementation (the coding agent optimizes this)
run_eval.py     — YOUR evaluation harness + metrics (customize before starting)
dataset.json    — YOUR evaluation dataset (customize before starting)
program.md      — instructions for the AI coding agent (customize before starting)
results.tsv     — experiment log (auto-generated)
run_search.py   — CLI for open-endedness algorithms (MAP-Elites, ADAS, etc.)
algorithms/     — open-endedness search algorithm implementations
```

**Before you start the autonomous loop**, you customize everything to fit your use case. Once the loop begins, `run_eval.py` and `dataset.json` are fixed — only `agent.py` changes.

## Quick Start

### Prerequisites

- Python 3.10+
- A [LangSmith API key](https://smith.langchain.com/)
- An LLM API key (OpenAI, Anthropic, etc.)

### Setup

```bash
# 1. Install dependencies (adjust for your agent's needs)
pip install langsmith langchain-openai langgraph

# 2. Set environment variables
export LANGSMITH_API_KEY=<your-key>
export LANGSMITH_TRACING=true
export OPENAI_API_KEY=<your-key>

# 3. Verify the baseline agent works
python agent.py "What is the capital of France?"

# 4. Run a single evaluation
python run_eval.py
```

### Running the Autonomous Agent

Point your coding agent (Claude Code, Cursor, Codex, etc.) at this directory and send this prompt:

<details>
<summary><b>📋 Copy-paste prompt for Claude Code / Cursor / Codex</b></summary>

```
I want you to autonomously optimize an AI agent using an eval-driven experiment loop.

Here's how it works:
- `agent.py` is the agent implementation. This is the ONLY file you modify.
- `run_eval.py` is the evaluation harness. It runs the agent against a fixed dataset
  and scores it using LangSmith. Do NOT modify this file.
- `dataset.json` is the evaluation dataset. Do NOT modify this file.
- `program.md` has detailed instructions for the experiment loop.

Read program.md now and follow the setup instructions. Once setup is confirmed,
start the experiment loop and run autonomously — do not stop to ask me questions.
Keep experimenting until I interrupt you.
```

</details>

The coding agent will then autonomously iterate on `agent.py`, running evals and tracking results. You can walk away and come back to a log of experiments in `results.tsv` and all traces in LangSmith.

#### First time here? Use this prompt instead to set everything up from scratch:

<details>
<summary><b>📋 Copy-paste setup + run prompt</b></summary>

```
I want to set up and run autoresearch for agents — an autonomous experiment loop
that optimizes an AI agent using LangSmith evals.

First, help me get set up:
1. Install dependencies: pip install langsmith langchain-openai langgraph
2. Make sure these env vars are set: LANGSMITH_API_KEY, LANGSMITH_TRACING=true, OPENAI_API_KEY
3. Verify the agent works: python agent.py "What is 2+2?"
4. Run a baseline eval: python run_eval.py

If I want to customize this for my own agent, walk me through:
- Replacing agent.py with my own agent implementation
- Replacing dataset.json with my own test cases
- Updating the evaluators in run_eval.py for my use case

Once everything is set up and the baseline looks good, read program.md and start
the autonomous experiment loop. Do not stop to ask me questions — keep experimenting
until I interrupt you.
```

</details>

## Bring Your Own Everything

This repo ships with a working example (a Q&A agent with calculator tools), but it's designed as a **template**. Customize all three components before starting the autonomous loop.

### Bring Your Own Agent

Replace `agent.py` with any agent implementation. It doesn't need to use LangChain or LangGraph — any Python code works. The only requirement is that it exposes a function that `run_eval.py` can call.

**Examples:**

```python
# Option A: Plain OpenAI SDK
from openai import OpenAI
client = OpenAI()

def run_agent(question: str) -> dict:
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": question}]
    )
    return {"response": response.choices[0].message.content}
```

```python
# Option B: Anthropic SDK
import anthropic
client = anthropic.Anthropic()

def run_agent(question: str) -> dict:
    message = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=1024,
        messages=[{"role": "user", "content": question}]
    )
    return {"response": message.content[0].text}
```

```python
# Option C: LangGraph agent (the default example)
from langgraph.prebuilt import create_react_agent
# ... see agent.py
```

```python
# Option D: Custom agent with no framework
def run_agent(question: str) -> dict:
    # Your custom logic here — RAG, multi-step, tool use, whatever
    return {"response": answer, "tools_used": [...]}
```

The key contract: your agent function takes the inputs from your dataset and returns a dict that your evaluators can score.

### Bring Your Own Dataset

Replace `dataset.json` with your evaluation cases. The format is a JSON array of objects with `inputs` and `outputs`:

```json
[
  {
    "inputs": {"question": "Your input here"},
    "outputs": {"answer": "Expected output", "any_other_field": "..."}
  }
]
```

The field names are up to you — just make sure your evaluators in `run_eval.py` reference the same fields. Some ideas:

- **Q&A agent**: `inputs: {question}`, `outputs: {answer}`
- **RAG agent**: `inputs: {question}`, `outputs: {answer, source_docs}`
- **Code agent**: `inputs: {task}`, `outputs: {code, test_result}`
- **Customer support**: `inputs: {ticket}`, `outputs: {response, category, priority}`

### Bring Your Own Evaluators

Modify the evaluators in `run_eval.py` to match your quality criteria. Each evaluator is a function that takes `(run, example)` and returns `{"score": number, "comment": "..."}`.

```python
# LLM-as-judge evaluator
def my_evaluator(run, example) -> dict:
    run_outputs = run.outputs or {}
    example_outputs = example.outputs or {}
    # Compare run_outputs to example_outputs using an LLM judge
    grade = judge.invoke([...])
    return {"score": grade.score, "comment": grade.reasoning}

# Code-based evaluator
def exact_match(run, example) -> dict:
    actual = run.outputs.get("answer", "")
    expected = example.outputs.get("answer", "")
    return {"score": 1 if actual == expected else 0, "comment": ""}
```

After modifying evaluators, update `program.md` to reflect the new metric names in the output format and TSV columns.

### Update program.md

After customizing the above, update `program.md` to match:
- Update the file list in the Setup section
- Update the "Ideas to try" section with domain-specific suggestions
- Update the output format section if your metrics changed
- Update the TSV columns to match your evaluator names

## Open-Endedness Algorithms

Beyond simple hill-climbing, this project includes **population-based search algorithms** inspired by open-endedness research. These maintain archives of diverse agent variants and explore the design space more broadly.

```bash
# Install (same deps as base project, plus openai for the mutator LLM)
pip install langsmith langchain-openai langgraph openai
```

### Available Algorithms

| Algorithm | Command | What it does |
|-----------|---------|-------------|
| **MAP-Elites** | `python run_search.py map-elites` | Quality-diversity grid search. Fills a grid of behavioral niches, each with the best agent for that niche. |
| **ADAS** | `python run_search.py adas` | Meta-agent designs novel architectures by reviewing an archive of all previous designs. |
| **Novelty Search** | `python run_search.py novelty` | Selects for behavioral novelty, discovering diverse stepping stones to high fitness. |
| **Go-Explore** | `python run_search.py go-explore` | Returns to promising but under-explored cells and explores from there. |

### Quick Start

```bash
# Run MAP-Elites for 50 iterations
python run_search.py map-elites --max-iterations 50

# Run ADAS with a stronger mutator model
python run_search.py adas --mutator-model gpt-4o --max-iterations 30

# Run Novelty Search (pure novelty, no fitness pressure)
python run_search.py novelty --novelty-weight 1.0 --max-iterations 50

# Run Go-Explore with high curiosity
python run_search.py go-explore --curiosity-weight 2.0 --max-iterations 50
```

### How They Work

All algorithms follow the same core loop but differ in **selection** and **archiving**:

1. **Select** a parent agent from the archive (algorithm-specific)
2. **Mutate** it using an LLM (proposes code changes to `agent.py`)
3. **Evaluate** the new variant via `run_eval.py`
4. **Archive** the result (algorithm-specific placement/acceptance)
5. Repeat

State is persisted to `oe_state/` so you can resume interrupted runs. Results are logged to `oe_results.tsv`.

### MAP-Elites

Places agents in a 2D grid indexed by behavioral descriptors (e.g., tool usage score × correctness). Each cell keeps only the best agent for that niche. The mutator LLM is told which cells are empty and encouraged to explore different niches.

```bash
# Custom grid dimensions
python run_search.py map-elites \
  --dims tool_usage_score correctness \
  --resolutions 5 5
```

### ADAS (Automated Design of Agentic Systems)

Inspired by [Hu et al. (2024)](https://arxiv.org/abs/2408.08435). A meta-agent reviews the full archive of previous designs — both successes and failures — and proposes fundamentally new architectures (not just prompt tweaks). Encourages framework changes, novel tool patterns, and different agent types.

### Novelty Search

Inspired by [Lehman & Stanley (2011)](https://direct.mit.edu/evco/article/19/2/189/1001). Balances fitness and behavioral novelty using a configurable weight. Novelty is computed as mean distance to k-nearest neighbors in a behavioral descriptor space. Drives exploration of under-represented regions.

```bash
# Balance novelty and fitness equally
python run_search.py novelty --novelty-weight 0.5

# Pure novelty search (ignore fitness entirely)
python run_search.py novelty --novelty-weight 1.0
```

### Go-Explore

Inspired by [Ecoffet et al. (2021)](https://www.nature.com/articles/s41586-020-03157-9). Discretizes the behavioral space into cells, tracks visit counts, and preferentially returns to promising but under-explored cells. Avoids the "detachment" problem where search drifts away from good regions.

```bash
# More curiosity-driven exploration
python run_search.py go-explore --curiosity-weight 2.0 --quality-weight 0.5
```

### Behavioral Descriptors

All algorithms use behavioral descriptors to characterize agents. These are automatically extracted from code structure and evaluation results:

| Descriptor | Source | Description |
|-----------|--------|-------------|
| `tool_usage_score` | Eval | Average tool usage score from evaluator |
| `correctness` | Eval | Average correctness score |
| `helpfulness` | Eval | Average helpfulness score |
| `code_lines` | Code | Lines of code in agent.py |
| `num_tools` | Code | Number of tool functions defined |
| `prompt_length` | Code | Length of system prompt |
| `model_tier` | Code | Numeric tier of the LLM model (0=small, 1=large) |

## How It Works

### The Evaluation (`run_eval.py`)

Uses LangSmith's `evaluate()` to run the agent against a persistent dataset and score it with your evaluators. The dataset is created once in LangSmith and reused across all experiments, so results accumulate and you can compare experiments side by side in the LangSmith UI.

All agent runs are traced in LangSmith for full observability — you can inspect exactly what happened in each run.

### The Experiment Loop (`program.md`)

The `program.md` file is the "skill" that drives the autonomous coding agent. It tells the agent to:
1. Edit `agent.py` with an experimental idea
2. Commit, run eval, check scores
3. Keep improvements, discard regressions
4. Log everything to `results.tsv`
5. Never stop until interrupted

## License

MIT
