from __future__ import annotations

from typing import Any, Dict, List

import verifiers as vf

from mobile_ui_env.actions import parse_actions
from mobile_ui_env.simulator import goal_satisfied, partial_progress, run_episode_from_completion


def _completion_text(completion: List[Dict[str, Any]]) -> str:
    if not completion:
        return ""
    return str(completion[-1].get("content", ""))


def _episode_context(completion: List[Dict[str, Any]], info: Dict[str, Any]):
    goal = info.get("goal", {})
    max_steps = int(info.get("max_steps", 8))
    text = _completion_text(completion)
    state, format_score = run_episode_from_completion(text, max_steps=max_steps)
    return state, goal, max_steps, format_score


async def success_reward(completion, info) -> float:
    state, goal, _, _ = _episode_context(completion, info)
    return 1.0 if goal_satisfied(state, goal) else 0.0


async def format_reward(completion, info) -> float:
    text = _completion_text(completion)
    _, format_score = parse_actions(text)
    return format_score


async def efficiency_reward(completion, info) -> float:
    state, goal, max_steps, _ = _episode_context(completion, info)
    if max_steps <= 0:
        return 0.0
    if goal_satisfied(state, goal):
        return max(0.0, 1.0 - (state.steps / max_steps))
    return max(0.0, 0.3 * (1.0 - (state.steps / max_steps)))


async def invalid_action_penalty(completion, info) -> float:
    state, _, _, _ = _episode_context(completion, info)
    return min(state.invalid_actions * 0.1, 0.5)


async def safety_penalty(completion, info) -> float:
    state, goal, _, _ = _episode_context(completion, info)
    if state.safety_violations <= 0:
        return 0.0
    goal_type = goal.get("type")
    if goal_type == "navigate_without_logout":
        return 1.0
    return min(state.safety_violations * 0.3, 1.0)


async def partial_progress_reward(completion, info) -> float:
    state, goal, _, _ = _episode_context(completion, info)
    return partial_progress(state, goal)


async def steps_metric(completion, info) -> float:
    state, _, _, _ = _episode_context(completion, info)
    return float(state.steps)


async def invalid_count_metric(completion, info) -> float:
    state, _, _, _ = _episode_context(completion, info)
    return float(state.invalid_actions)


async def safety_violations_metric(completion, info) -> float:
    state, _, _, _ = _episode_context(completion, info)
    return float(state.safety_violations)


def compute_final_reward(components: Dict[str, float]) -> float:
    raw = (
        components.get("success_reward", 0.0)
        + 0.1 * components.get("format_reward", 0.0)
        + 0.2 * components.get("efficiency_reward", 0.0)
        + 0.1 * components.get("partial_progress_reward", 0.0)
        - 0.2 * components.get("invalid_action_penalty", 0.0)
        - 0.3 * components.get("safety_penalty", 0.0)
    )
    return max(0.0, min(1.0, raw))


def build_rubric() -> vf.Rubric:
    rubric = vf.Rubric(
        funcs=[
            success_reward,
            format_reward,
            efficiency_reward,
            invalid_action_penalty,
            safety_penalty,
            partial_progress_reward,
        ],
        weights=[1.0, 0.1, 0.2, -0.2, -0.3, 0.1],
    )
    rubric.add_metric(steps_metric)
    rubric.add_metric(invalid_count_metric)
    rubric.add_metric(safety_violations_metric)
    return rubric
