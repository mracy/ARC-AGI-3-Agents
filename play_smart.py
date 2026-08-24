"""Local adaptive agent for ARC-AGI-3.

This is a next-level baseline. It learns action value from experience:
- rewards actions that advance a level
- punishes actions that cause GAME_OVER
- rewards actions that visibly change the grid
- avoids recently bad actions
"""

import argparse
import logging
import os
import random
import sys
import time
from itertools import zip_longest

from dotenv import load_dotenv

from arc_agi import Arcade
from arcengine import FrameData, GameAction, GameState

from agents.agent import Agent


def _grid_diff(prev: list, curr: list) -> int:
    """Count of cells that changed between two frame grids."""
    diff = 0
    for r1, r2 in zip_longest(prev, curr, fillvalue=()):
        for c1, c2 in zip_longest(r1, r2, fillvalue=-1):
            if c1 != c2:
                diff += 1
    return diff


class LearningAgent(Agent):
    """An agent that adapts its action preference based on observed effects."""

    MAX_ACTIONS = 80
    EXPLORATION_NOISE = 0.1

    def __init__(self, *args: object, **kwargs: object) -> None:
        super().__init__(*args, **kwargs)
        if self.arc_env is None:
            raise ValueError("LearningAgent requires an arc_env")
        seed = int(time.time() * 1_000_000) + hash(self.game_id) % 1_000_000
        random.seed(seed)

        self._last_action_id: int | None = None
        self._last_action_data: dict = {}
        self._last_frame: FrameData | None = None
        self._last_levels = 0

        # action_id -> learned score
        self._action_scores: dict[int, float] = {}
        self._bad_actions: set[int] = set()

    def is_done(self, frames: list[FrameData], latest_frame: FrameData) -> bool:
        return (
            latest_frame.state is GameState.WIN
            or self.action_counter >= self.MAX_ACTIONS
        )

    def choose_action(
        self, frames: list[FrameData], latest_frame: FrameData
    ) -> GameAction:
        if latest_frame.state in (GameState.NOT_PLAYED, GameState.GAME_OVER):
            self._last_action_id = GameAction.RESET.value
            self._last_action_data = {}
            self._last_frame = latest_frame
            return GameAction.RESET

        # ---- Learn from the previous action's effect ----
        if self._last_action_id is not None and self._last_frame is not None:
            if latest_frame.levels_completed > self._last_levels:
                self._action_scores[self._last_action_id] = (
                    self._action_scores.get(self._last_action_id, 0.0) + 100.0
                )
                self._last_levels = latest_frame.levels_completed
            elif latest_frame.state is GameState.GAME_OVER:
                self._action_scores[self._last_action_id] = (
                    self._action_scores.get(self._last_action_id, 0.0) - 50.0
                )
                self._bad_actions.add(self._last_action_id)
            else:
                change = _grid_diff(self._last_frame.frame, latest_frame.frame)
                if change > 0:
                    self._action_scores[self._last_action_id] = (
                        self._action_scores.get(self._last_action_id, 0.0)
                        + change * 0.1
                    )
                else:
                    self._action_scores[self._last_action_id] = (
                        self._action_scores.get(self._last_action_id, 0.0) - 0.2
                    )

        # ---- Choose the next action ----
        available = [
            aid
            for aid in (latest_frame.available_actions or [])
            if aid != 0 and aid not in self._bad_actions
        ]
        if not available:
            # fall back to non-bad actions in the full enum
            available = [
                a.value
                for a in GameAction
                if a is not GameAction.RESET and a.value not in self._bad_actions
            ]
        if not available:
            # all actions are marked bad; try the least bad one
            available = [a for a in (latest_frame.available_actions or []) if a != 0]
            if not available:
                available = [a.value for a in GameAction if a is not GameAction.RESET]

        # Score each available action. Add small random noise for exploration.
        def score(aid: int) -> float:
            return self._action_scores.get(aid, 0.0) + random.uniform(
                0.0, self.EXPLORATION_NOISE
            )

        action_id = max(available, key=score)
        action = GameAction.from_id(action_id)

        if action.is_complex():
            data = {
                "x": random.randint(0, 63),
                "y": random.randint(0, 63),
            }
            action.set_data(data)
            self._last_action_data = data
        else:
            self._last_action_data = {}

        action.reasoning = f"learned score {self._action_scores.get(action_id, 0.0):.2f}"

        self._last_action_id = action_id
        self._last_frame = latest_frame
        return action


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the adaptive learning agent locally")
    parser.add_argument(
        "--game",
        default="ls20",
        help="Public game id to play (e.g. ls20, ar25)",
    )
    parser.add_argument(
        "--max-actions",
        type=int,
        default=80,
        help="Maximum actions before forcing stop",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        stream=sys.stdout,
    )

    load_dotenv()

    # This runner is intentionally local-only.
    os.environ["OPERATION_MODE"] = "local"

    # Make sure we use the local environment directory.
    env_dir = os.environ.get("ENVIRONMENTS_DIR", "").strip()
    if not env_dir or not os.path.isdir(env_dir):
        env_dir = os.path.abspath(
            os.path.join(os.path.dirname(__file__), "..", "environment_files")
        )
        os.environ["ENVIRONMENTS_DIR"] = env_dir

    arcade = Arcade()
    env = arcade.make(args.game, scorecard_id="local-smart")

    LearningAgent.MAX_ACTIONS = args.max_actions
    agent = LearningAgent(
        card_id="local-smart",
        game_id=args.game,
        agent_name="learning",
        ROOT_URL="local",
        record=False,
        arc_env=env,
    )

    agent.main()

    print(
        f"\n--- Result for {args.game} ---\n"
        f"State: {agent.state}\n"
        f"Levels completed: {agent.levels_completed}\n"
        f"Actions used: {agent.action_counter}\n"
        f"Time: {agent.seconds}s"
    )


if __name__ == "__main__":
    main()
