"""Kaggle submission runner for ARC-AGI-3.

Run this inside a Kaggle notebook (or locally) to play every available game
with the LearningAgent and produce a scorecard / submission.json.

Kaggle usage:
    !python kaggle_submission.py --max-actions 80
"""

import argparse
import base64
import logging
import os
import random
import re
import subprocess
import sys
import time
import uuid
from glob import glob
from itertools import zip_longest
from pathlib import Path
from typing import Any

from dotenv import load_dotenv


def _install_arc_agi_if_needed() -> None:
    """Try to install arc-agi from a local wheel (Kaggle input or repo)."""
    try:
        import arc_agi  # type: ignore
        return
    except ImportError:
        pass

    candidates = []

    # 1. Exact wheel file or directory from env var
    arc_wheel = os.environ.get("ARC_WHEEL", "").strip()
    if arc_wheel:
        if os.path.isfile(arc_wheel) and arc_wheel.endswith(".whl"):
            candidates.append(arc_wheel)
        elif os.path.isdir(arc_wheel):
            candidates.extend(glob(f"{arc_wheel}/*.whl"))

    # 2. Common Kaggle competition input path
    kaggle_default = "/kaggle/input/arc-prize-2026-arc-agi-3"
    if os.path.isdir(kaggle_default):
        candidates.extend(glob(f"{kaggle_default}/arc_agi_3_wheels/*.whl"))
        candidates.extend(glob(f"{kaggle_default}/*.whl"))

    # 3. Any /kaggle/input/* location
    if os.path.isdir("/kaggle/input"):
        for d in glob("/kaggle/input/*"):
            candidates.extend(glob(f"{d}/arc_agi_3_wheels/*.whl"))
            candidates.extend(glob(f"{d}/*.whl"))

    # 4. Local repo layout
    candidates.extend(glob("../arc_agi_3_wheels/*.whl"))
    candidates.extend(glob("arc_agi_3_wheels/*.whl"))

    if not candidates:
        print("--- /kaggle/input contents ---")
        if os.path.isdir("/kaggle/input"):
            print(os.listdir("/kaggle/input"))
            for d in glob("/kaggle/input/*"):
                print(d, os.listdir(d) if os.path.isdir(d) else "file")
        else:
            print("/kaggle/input does not exist")
        raise RuntimeError(
            "arc-agi not installed and no .whl wheel found. "
            "Add the competition wheel as a Kaggle input, or set ARC_WHEEL "
            "to the wheel file/directory."
        )

    wheel = candidates[0]
    print(f"Installing arc-agi from {wheel}")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "--quiet", wheel])


_install_arc_agi_if_needed()

import arc_agi  # noqa: E402
from arc_agi import Arcade, OperationMode
from arc_agi.scorecard import EnvironmentScorecard
from arcengine import FrameData, GameAction, GameState

from agents.agent import Agent


def _find_environments_dir() -> str:
    """Find the environment_files folder (Kaggle input or local)."""
    env_dir = os.environ.get("ENVIRONMENTS_DIR", "").strip()
    if env_dir and os.path.isdir(env_dir):
        return env_dir

    # Explicit Kaggle env dir
    arc_env = os.environ.get("ARC_ENV_DIR", "").strip()
    if arc_env and os.path.isdir(arc_env):
        return arc_env

    # Common Kaggle competition input path
    kaggle_default = "/kaggle/input/arc-prize-2026-arc-agi-3"
    kaggle_env = f"{kaggle_default}/environment_files"
    if os.path.isdir(kaggle_env):
        return kaggle_env

    # Any /kaggle/input/* location
    if os.path.isdir("/kaggle/input"):
        for d in glob("/kaggle/input/*/environment_files"):
            if os.path.isdir(d):
                return d

    # Local layout relative to this script
    local = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "environment_files"))
    if os.path.isdir(local):
        return local

    raise RuntimeError("environment_files directory not found")


def _grid_diff(prev: list, curr: list) -> int:
    diff = 0
    for r1, r2 in zip_longest(prev, curr, fillvalue=()):
        for c1, c2 in zip_longest(r1, r2, fillvalue=-1):
            if c1 != c2:
                diff += 1
    return diff


class LearningAgent(Agent):
    """Adaptive action-value learning agent."""

    MAX_ACTIONS = 80
    EXPLORATION_NOISE = 0.1

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        if self.arc_env is None:
            raise ValueError("LearningAgent requires an arc_env")
        seed = int(time.time() * 1_000_000) + hash(self.game_id) % 1_000_000
        random.seed(seed)

        self._last_action_id: int | None = None
        self._last_action_data: dict = {}
        self._last_frame: FrameData | None = None
        self._last_levels = 0
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

        # Learn from the effect of the previous action
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

        # Choose an action
        available = [
            aid
            for aid in (latest_frame.available_actions or [])
            if aid != 0 and aid not in self._bad_actions
        ]
        if not available:
            available = [
                a.value
                for a in GameAction
                if a is not GameAction.RESET and a.value not in self._bad_actions
            ]
        if not available:
            available = [a for a in (latest_frame.available_actions or []) if a != 0]
            if not available:
                available = [a.value for a in GameAction if a is not GameAction.RESET]

        def score(aid: int) -> float:
            return self._action_scores.get(aid, 0.0) + random.uniform(
                0.0, self.EXPLORATION_NOISE
            )

        action_id = max(available, key=score)
        action = GameAction.from_id(action_id)

        if action.is_complex():
            data = {"x": random.randint(0, 63), "y": random.randint(0, 63)}
            action.set_data(data)
            self._last_action_data = data
        else:
            self._last_action_data = {}

        action.reasoning = f"learned score {self._action_scores.get(action_id, 0.0):.2f}"

        self._last_action_id = action_id
        self._last_frame = latest_frame
        return action


def main() -> None:
    parser = argparse.ArgumentParser(description="Kaggle ARC-AGI-3 submission runner")
    parser.add_argument(
        "--max-actions",
        type=int,
        default=80,
        help="Max actions per game",
    )
    parser.add_argument(
        "--output",
        default="/kaggle/working/submission.json",
        help="Where to write the submission file",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        stream=sys.stdout,
    )

    load_dotenv()

    # Kaggle: work offline with the provided environment files
    os.environ["OPERATION_MODE"] = "offline"
    os.environ["ENVIRONMENTS_DIR"] = _find_environments_dir()
    os.environ["ARC_API_KEY"] = os.environ.get("ARC_API_KEY", "kaggle")
    os.environ.setdefault("RECORDINGS_DIR", "/kaggle/working/recordings")

    arcade = Arcade()
    game_ids = [e.game_id for e in arcade.available_environments]
    if not game_ids:
        raise RuntimeError("No games found in environment_files")

    logging.info(f"Found {len(game_ids)} games: {game_ids[:10]}...")

    scorecard_id = arcade.scorecard_manager.new_scorecard(
        source_url=None,
        tags=["kaggle"],
        api_key="",
        opaque=None,
        competition_mode=True,
    )

    LearningAgent.MAX_ACTIONS = args.max_actions

    for game_id in game_ids:
        logging.info(f"Starting game {game_id}")
        try:
            env = arcade.make(game_id, scorecard_id=scorecard_id)
            if env is None:
                logging.warning(f"Could not create environment for {game_id}")
                continue

            agent = LearningAgent(
                card_id=scorecard_id,
                game_id=game_id,
                agent_name="learning",
                ROOT_URL="local",
                record=False,
                arc_env=env,
            )
            agent.main()
            logging.info(
                f"Finished {game_id}: state={agent.state}, "
                f"levels={agent.levels_completed}, actions={agent.action_counter}"
            )
        except Exception as e:
            logging.error(f"Failed on {game_id}: {e}", exc_info=True)

    # Build the scorecard output
    scorecard, *_ = arcade.scorecard_manager.close_scorecard(scorecard_id, api_key="")
    if scorecard is None:
        raise RuntimeError("close_scorecard returned None")

    env_scorecard = EnvironmentScorecard.from_scorecard(scorecard, arcade.available_environments)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(env_scorecard.model_dump_json(indent=2), encoding="utf-8")
    logging.info(f"Submission written to {output_path}")


if __name__ == "__main__":
    main()
