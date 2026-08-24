"""Standalone local baseline runner for ARC-AGI-3.

This script does not require modifying agents/__init__.py or main.py.
It loads a local environment and runs a simple heuristic agent on one game.
"""

import argparse
import logging
import os
import random
import sys
import time

from dotenv import load_dotenv

from arc_agi import Arcade
from arcengine import FrameData, GameAction, GameState

from agents.agent import Agent


class HeuristicAgent(Agent):
    """A minimal baseline that repeats actions that caused a level-up."""

    MAX_ACTIONS = 80

    def __init__(self, *args: object, **kwargs: object) -> None:
        super().__init__(*args, **kwargs)
        if self.arc_env is None:
            raise ValueError("HeuristicAgent requires an arc_env")
        seed = int(time.time() * 1_000_000) + hash(self.game_id) % 1_000_000
        random.seed(seed)
        self._last_action_id: int | None = None
        self._last_action_data: dict = {}
        self._last_levels = 0
        self._repeat_id: int | None = None
        self._repeat_data: dict = {}

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
            return GameAction.RESET

        # Detect if the previous action advanced a level.
        if self._last_action_id is not None:
            if latest_frame.levels_completed > self._last_levels:
                self._last_levels = latest_frame.levels_completed
                self._repeat_id = self._last_action_id
                self._repeat_data = self._last_action_data.copy()
            else:
                self._repeat_id = None
                self._repeat_data = {}

        # If we found a successful action on the last turn, repeat it once.
        if self._repeat_id is not None:
            action = GameAction.from_id(self._repeat_id)
            if self._repeat_data:
                action.set_data(self._repeat_data)
            action.reasoning = "repeat previous action that caused a level-up"
            self._last_action_id = self._repeat_id
            self._last_action_data = self._repeat_data.copy()
            self._repeat_id = None
            return action

        # Otherwise, pick a random non-RESET action from the environment's list.
        available = [
            aid for aid in (latest_frame.available_actions or []) if aid != 0
        ]
        if not available:
            available = [
                a.value
                for a in GameAction
                if a is not GameAction.RESET
            ]

        action_id = random.choice(available)
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
        action.reasoning = "heuristic random exploration"

        self._last_action_id = action_id
        return action


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the heuristic baseline locally")
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
    env = arcade.make(args.game, scorecard_id="local-baseline")

    HeuristicAgent.MAX_ACTIONS = args.max_actions
    agent = HeuristicAgent(
        card_id="local-baseline",
        game_id=args.game,
        agent_name="heuristic",
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
