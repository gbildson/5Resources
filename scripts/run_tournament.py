"""Run baseline tournaments for quick sanity checks."""

from __future__ import annotations

import argparse
import json

from catan_rl.bots import HeuristicAgent, RandomLegalAgent
from catan_rl.eval import tournament


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--games", type=int, default=50)
    parser.add_argument("--seed", type=int, default=123)
    args = parser.parse_args()

    agents = [HeuristicAgent(seed=args.seed), RandomLegalAgent(seed=args.seed + 1), RandomLegalAgent(seed=args.seed + 2), RandomLegalAgent(seed=args.seed + 3)]
    stats = tournament(agents, num_games=args.games, base_seed=args.seed)
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
