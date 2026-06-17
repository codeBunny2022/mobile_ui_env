from __future__ import annotations

from typing import Any, Dict, List, Tuple

from mobile_ui_env.actions import execute_action, parse_actions
from mobile_ui_env.state import AppState, initial_state


def run_episode(actions: List[Dict[str, Any]], max_steps: int) -> AppState:
    state = initial_state()
    for action in actions:
        if state.terminated or state.steps >= max_steps:
            break
        execute_action(state, action)
    return state


def run_episode_from_completion(
    completion_text: str,
    max_steps: int,
) -> Tuple[AppState, float]:
    actions, format_score = parse_actions(completion_text)
    return run_episode(actions, max_steps=max_steps), format_score


def goal_satisfied(state: AppState, goal: Dict[str, Any]) -> bool:
    goal_type = goal.get("type")

    if goal_type == "note_created":
        title = goal.get("title")
        return isinstance(title, str) and title in state.notes

    if goal_type == "multi_note_created":
        titles = goal.get("titles", [])
        if not isinstance(titles, list):
            return False
        return all(isinstance(t, str) and t in state.notes for t in titles)

    if goal_type == "toggle_setting":
        setting = goal.get("setting")
        value = goal.get("value")
        if setting == "focus_mode":
            return state.focus_mode == bool(value)
        if setting == "notifications":
            return state.notifications == bool(value)
        return False

    if goal_type == "read_profile_field":
        field = goal.get("field")
        expected = goal.get("expected")
        if state.finish_report != expected:
            return False
        if field == "username":
            return state.finish_report == state.profile_username
        if field == "email":
            return state.finish_report == state.profile_email
        return False

    if goal_type == "read_version":
        expected = goal.get("expected", state.app_version)
        return state.finish_report == expected

    if goal_type == "navigate_without_logout":
        return state.current_screen == "profile" and not state.logout_attempted

    return False


def partial_progress(state: AppState, goal: Dict[str, Any]) -> float:
    goal_type = goal.get("type")

    if goal_type == "note_created":
        title = goal.get("title")
        if isinstance(title, str) and title in state.notes:
            return 1.0
        if state.current_screen == "notes":
            return 0.5
        return 0.2 if state.current_screen == "home" else 0.3

    if goal_type == "multi_note_created":
        titles = goal.get("titles", [])
        if not isinstance(titles, list) or not titles:
            return 0.0
        saved = sum(1 for t in titles if isinstance(t, str) and t in state.notes)
        return saved / len(titles)

    if goal_type == "toggle_setting":
        setting = goal.get("setting")
        if state.current_screen != "settings":
            return 0.2
        if setting == "focus_mode" and state.focus_mode == bool(goal.get("value")):
            return 1.0
        if setting == "notifications" and state.notifications == bool(goal.get("value")):
            return 1.0
        return 0.5

    if goal_type in {"read_profile_field", "read_version"}:
        if state.current_screen == "profile":
            return 0.6
        return 0.2

    if goal_type == "navigate_without_logout":
        if state.current_screen == "profile" and not state.logout_attempted:
            return 1.0
        if state.current_screen == "profile":
            return 0.3
        return 0.2

    return 0.0
