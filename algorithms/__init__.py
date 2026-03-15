"""
Open-endedness algorithms for autoresearch-agents.

Provides population-based search algorithms that maintain archives of diverse
agent variants, going beyond simple hill-climbing.

Available algorithms:
    - MAP-Elites: Quality-diversity search over behavioral descriptor grid
    - ADAS: Automated Design of Agentic Systems (meta-agent search)
    - NoveltySearch: Behavioral novelty-driven exploration
    - GoExplore: Archive-driven return-and-explore
"""

from algorithms.map_elites import MAPElites
from algorithms.adas import ADAS
from algorithms.novelty_search import NoveltySearch
from algorithms.go_explore import GoExplore

__all__ = ["MAPElites", "ADAS", "NoveltySearch", "GoExplore"]
