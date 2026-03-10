#!/usr/bin/env python3
"""
CLI entry point for open-endedness search algorithms.

Runs a population-based search to explore the space of agent designs,
going beyond simple hill-climbing to discover diverse, high-quality agents.

Usage:
    # MAP-Elites (quality-diversity grid search)
    python run_search.py map-elites --max-iterations 50

    # ADAS (meta-agent designs new architectures)
    python run_search.py adas --max-iterations 50

    # Novelty Search (behavioral diversity-driven)
    python run_search.py novelty --novelty-weight 0.5 --max-iterations 50

    # Go-Explore (return to promising cells and explore)
    python run_search.py go-explore --max-iterations 50

    # Use a specific mutator model
    python run_search.py map-elites --mutator-model gpt-4o --max-iterations 20
"""

import argparse
import sys


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Open-endedness search algorithms for agent optimization",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    # Common arguments
    parser.add_argument(
        "algorithm",
        choices=["map-elites", "adas", "novelty", "go-explore"],
        help="Search algorithm to run",
    )
    parser.add_argument("--max-iterations", type=int, default=None, help="Max iterations (default: unlimited)")
    parser.add_argument("--agent-path", default="agent.py", help="Path to agent.py")
    parser.add_argument("--eval-cmd", default="python run_eval.py", help="Evaluation command")
    parser.add_argument("--state-dir", default="oe_state", help="Directory for algorithm state")
    parser.add_argument("--log-file", default="oe_results.tsv", help="TSV log file path")
    parser.add_argument("--mutator-model", default="gpt-4o-mini", help="LLM model for generating mutations")
    parser.add_argument("--eval-timeout", type=int, default=600, help="Eval timeout in seconds")

    # MAP-Elites specific
    parser.add_argument(
        "--dims", nargs="+", default=None,
        help="MAP-Elites behavioral dimensions (default: tool_usage_score correctness)",
    )
    parser.add_argument(
        "--resolutions", nargs="+", type=int, default=None,
        help="MAP-Elites grid resolution per dimension (default: 5 5)",
    )

    # Novelty Search specific
    parser.add_argument("--novelty-weight", type=float, default=0.5, help="Novelty vs fitness weight (0-1)")
    parser.add_argument("--k-nearest", type=int, default=5, help="k for k-nearest novelty computation")

    # Go-Explore specific
    parser.add_argument("--curiosity-weight", type=float, default=1.0, help="Go-Explore curiosity weight")
    parser.add_argument("--quality-weight", type=float, default=1.0, help="Go-Explore quality weight")

    # ADAS specific
    parser.add_argument("--top-k-context", type=int, default=10, help="ADAS: top-k designs to show meta-agent")

    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()

    common_kwargs = {
        "agent_path": args.agent_path,
        "eval_cmd": args.eval_cmd,
        "state_dir": args.state_dir,
        "log_file": args.log_file,
        "mutator_model": args.mutator_model,
        "max_iterations": args.max_iterations,
        "eval_timeout": args.eval_timeout,
    }

    if args.algorithm == "map-elites":
        from algorithms.map_elites import MAPElites

        kwargs = {**common_kwargs}
        if args.dims:
            kwargs["dims"] = args.dims
        if args.resolutions:
            kwargs["resolutions"] = args.resolutions
        search = MAPElites(**kwargs)

    elif args.algorithm == "adas":
        from algorithms.adas import ADAS

        search = ADAS(top_k_context=args.top_k_context, **common_kwargs)

    elif args.algorithm == "novelty":
        from algorithms.novelty_search import NoveltySearch

        search = NoveltySearch(
            novelty_weight=args.novelty_weight,
            k_nearest=args.k_nearest,
            **common_kwargs,
        )

    elif args.algorithm == "go-explore":
        from algorithms.go_explore import GoExplore

        search = GoExplore(
            curiosity_weight=args.curiosity_weight,
            quality_weight=args.quality_weight,
            **common_kwargs,
        )

    else:
        parser.print_help()
        sys.exit(1)

    print(f"Starting {args.algorithm} search...")
    print(f"  Agent: {args.agent_path}")
    print(f"  Mutator model: {args.mutator_model}")
    print(f"  Max iterations: {args.max_iterations or 'unlimited'}")
    print(f"  State dir: {args.state_dir}")
    print()

    try:
        search.run()
    except KeyboardInterrupt:
        print("\nInterrupted. State has been saved.")
        search.save_state()


if __name__ == "__main__":
    main()
