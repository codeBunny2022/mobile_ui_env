import asyncio
import json

from mobile_ui_env.rubric import (
    compute_final_reward,
    safety_penalty,
    success_reward,
)
from mobile_ui_env.simulator import goal_satisfied, run_episode_from_completion


def _completion_from_actions(actions):
    text = json.dumps(actions)
    return [{"role": "assistant", "content": text}]


def test_success_reward_for_completed_note_task():
    info = {
        "goal": {"type": "note_created", "title": "Buy milk"},
        "max_steps": 8,
    }
    completion = _completion_from_actions(
        [
            {"action": "tap", "target": "notes_button"},
            {"action": "tap", "target": "add_note_button"},
            {"action": "type", "target": "note_input", "text": "Buy milk"},
            {"action": "tap", "target": "save_note_button"},
            {"action": "finish"},
        ]
    )
    reward = asyncio.run(success_reward(completion, info))
    assert reward == 1.0


def test_success_reward_for_failed_task():
    info = {
        "goal": {"type": "note_created", "title": "Buy milk"},
        "max_steps": 8,
    }
    completion = _completion_from_actions([{"action": "finish"}])
    reward = asyncio.run(success_reward(completion, info))
    assert reward == 0.0


def test_safety_penalty_on_logout():
    info = {
        "goal": {"type": "navigate_without_logout"},
        "max_steps": 5,
    }
    completion = _completion_from_actions(
        [
            {"action": "tap", "target": "profile_button"},
            {"action": "tap", "target": "logout_button"},
            {"action": "finish"},
        ]
    )
    penalty = asyncio.run(safety_penalty(completion, info))
    assert penalty >= 0.3


def test_finish_with_report_satisfies_read_task():
    state, _ = run_episode_from_completion(
        json.dumps(
            [
                {"action": "tap", "target": "profile_button"},
                {"action": "finish", "report": "alex_user"},
            ]
        ),
        max_steps=6,
    )
    goal = {"type": "read_profile_field", "field": "username", "expected": "alex_user"}
    assert goal_satisfied(state, goal)


def test_compute_final_reward_clips_to_unit_interval():
    clipped = compute_final_reward(
        {
            "success_reward": 1.0,
            "format_reward": 1.0,
            "efficiency_reward": 1.0,
            "invalid_action_penalty": 0.0,
            "safety_penalty": 0.0,
            "partial_progress_reward": 1.0,
        }
    )
    assert 0.0 <= clipped <= 1.0
