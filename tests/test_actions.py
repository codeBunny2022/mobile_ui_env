import json

from mobile_ui_env.actions import execute_action, parse_actions
from mobile_ui_env.simulator import run_episode
from mobile_ui_env.state import initial_state


def test_valid_tap_changes_screen():
    state = initial_state()
    execute_action(state, {"action": "tap", "target": "notes_button"})
    assert state.current_screen == "notes"
    assert state.invalid_actions == 0


def test_invalid_tap_does_not_crash():
    state = initial_state()
    execute_action(state, {"action": "tap", "target": "add_note_button"})
    assert state.current_screen == "home"
    assert state.invalid_actions == 1


def test_creating_note_updates_state():
    actions = [
        {"action": "tap", "target": "notes_button"},
        {"action": "tap", "target": "add_note_button"},
        {"action": "type", "target": "note_input", "text": "Buy milk"},
        {"action": "tap", "target": "save_note_button"},
        {"action": "finish"},
    ]
    state = run_episode(actions, max_steps=10)
    assert "Buy milk" in state.notes
    assert state.terminated is True


def test_parse_actions_valid_json():
    text = json.dumps(
        [
            {"action": "tap", "target": "notes_button"},
            {"action": "finish"},
        ]
    )
    actions, score = parse_actions(text)
    assert len(actions) == 2
    assert score == 1.0


def test_parse_actions_invalid_json_does_not_crash():
    actions, score = parse_actions("not json at all")
    assert actions == []
    assert score == 0.0


def test_back_returns_to_home():
    state = initial_state()
    execute_action(state, {"action": "tap", "target": "settings_button"})
    execute_action(state, {"action": "back"})
    assert state.current_screen == "home"
