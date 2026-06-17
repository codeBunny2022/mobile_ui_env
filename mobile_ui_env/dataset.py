from __future__ import annotations

from typing import Any, Dict, List, Literal

from datasets import Dataset

from mobile_ui_env.state import initial_state, screen_observation

ACTION_SPEC = """Return a JSON array of actions:
- tap: {"action": "tap", "target": "<id>"}
- type: {"action": "type", "target": "note_input", "text": "..."}
- back: {"action": "back"}
- finish: {"action": "finish"} or {"action": "finish", "report": "..."} for read tasks

Elements by screen:
home: notes_button, settings_button, profile_button
notes: add_note_button, note_input, save_note_button, note_list
settings: focus_mode_toggle, notifications_toggle, version_label
profile: username_label, email_label, logout_button
"""

TRAIN_TASKS: List[Dict[str, Any]] = [
    {"task_id": "task_001", "instruction": "Create a note titled Buy milk", "goal": {"type": "note_created", "title": "Buy milk"}, "max_steps": 8},
    {"task_id": "task_002", "instruction": "Create a note titled Call dentist", "goal": {"type": "note_created", "title": "Call dentist"}, "max_steps": 8},
    {"task_id": "task_003", "instruction": "Create a note titled Team meeting", "goal": {"type": "note_created", "title": "Team meeting"}, "max_steps": 8},
    {"task_id": "task_004", "instruction": "Create a note titled Grocery list", "goal": {"type": "note_created", "title": "Grocery list"}, "max_steps": 8},
    {"task_id": "task_005", "instruction": "Enable focus mode", "goal": {"type": "toggle_setting", "setting": "focus_mode", "value": True}, "max_steps": 6},
    {"task_id": "task_006", "instruction": "Disable notifications", "goal": {"type": "toggle_setting", "setting": "notifications", "value": False}, "max_steps": 6},
    {"task_id": "task_007", "instruction": "Enable notifications", "goal": {"type": "toggle_setting", "setting": "notifications", "value": True}, "max_steps": 6},
    {"task_id": "task_008", "instruction": "Disable focus mode", "goal": {"type": "toggle_setting", "setting": "focus_mode", "value": False}, "max_steps": 6},
    {"task_id": "task_009", "instruction": "Find the username from profile", "goal": {"type": "read_profile_field", "field": "username", "expected": "alex_user"}, "max_steps": 6},
    {"task_id": "task_010", "instruction": "Find the email from profile", "goal": {"type": "read_profile_field", "field": "email", "expected": "alex@example.com"}, "max_steps": 6},
    {"task_id": "task_011", "instruction": "Open settings and report app version", "goal": {"type": "read_version", "expected": "2.1.0"}, "max_steps": 6},
    {"task_id": "task_012", "instruction": "Create two notes: Buy eggs and Walk dog", "goal": {"type": "multi_note_created", "titles": ["Buy eggs", "Walk dog"]}, "max_steps": 14},
    {"task_id": "task_013", "instruction": "Go to profile and do not logout", "goal": {"type": "navigate_without_logout"}, "max_steps": 5},
    {"task_id": "task_014", "instruction": "Create a note titled Pay rent", "goal": {"type": "note_created", "title": "Pay rent"}, "max_steps": 8},
    {"task_id": "task_015", "instruction": "Create a note titled Read book", "goal": {"type": "note_created", "title": "Read book"}, "max_steps": 8},
    {"task_id": "task_016", "instruction": "Turn on focus mode in settings", "goal": {"type": "toggle_setting", "setting": "focus_mode", "value": True}, "max_steps": 6},
    {"task_id": "task_017", "instruction": "Turn off notifications in settings", "goal": {"type": "toggle_setting", "setting": "notifications", "value": False}, "max_steps": 6},
    {"task_id": "task_018", "instruction": "Create a note titled Water plants", "goal": {"type": "note_created", "title": "Water plants"}, "max_steps": 8},
    {"task_id": "task_019", "instruction": "Navigate to profile without logging out", "goal": {"type": "navigate_without_logout"}, "max_steps": 5},
    {"task_id": "task_020", "instruction": "Create two notes: Gym and Yoga", "goal": {"type": "multi_note_created", "titles": ["Gym", "Yoga"]}, "max_steps": 14},
]

EVAL_TASKS: List[Dict[str, Any]] = [
    {"task_id": "eval_001", "instruction": "Create a note titled Pick up laundry", "goal": {"type": "note_created", "title": "Pick up laundry"}, "max_steps": 8},
    {"task_id": "eval_002", "instruction": "Create a note titled Schedule flight", "goal": {"type": "note_created", "title": "Schedule flight"}, "max_steps": 8},
    {"task_id": "eval_003", "instruction": "Enable focus mode in the app settings", "goal": {"type": "toggle_setting", "setting": "focus_mode", "value": True}, "max_steps": 6},
    {"task_id": "eval_004", "instruction": "Disable notifications in the app settings", "goal": {"type": "toggle_setting", "setting": "notifications", "value": False}, "max_steps": 6},
    {"task_id": "eval_005", "instruction": "Find the username shown on the profile screen", "goal": {"type": "read_profile_field", "field": "username", "expected": "alex_user"}, "max_steps": 6},
    {"task_id": "eval_006", "instruction": "Find the email shown on the profile screen", "goal": {"type": "read_profile_field", "field": "email", "expected": "alex@example.com"}, "max_steps": 6},
    {"task_id": "eval_007", "instruction": "Open settings and report the app version", "goal": {"type": "read_version", "expected": "2.1.0"}, "max_steps": 6},
    {"task_id": "eval_008", "instruction": "Create two notes: Buy coffee and Send email", "goal": {"type": "multi_note_created", "titles": ["Buy coffee", "Send email"]}, "max_steps": 14},
    {"task_id": "eval_009", "instruction": "Go to profile and do not tap logout", "goal": {"type": "navigate_without_logout"}, "max_steps": 5},
    {"task_id": "eval_010", "instruction": "Create a note titled Backup files", "goal": {"type": "note_created", "title": "Backup files"}, "max_steps": 8},
]


def _build_prompt(instruction: str) -> List[Dict[str, str]]:
    observation = screen_observation(initial_state())
    content = (
        f"{observation}\n\n"
        f"Task: {instruction}\n\n"
        f"{ACTION_SPEC}"
    )
    return [{"role": "user", "content": content}]


def _to_dataset_row(task: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "task_id": task["task_id"],
        "instruction": task["instruction"],
        "goal": task["goal"],
        "max_steps": task["max_steps"],
        "prompt": _build_prompt(task["instruction"]),
        "info": {
            "task_id": task["task_id"],
            "goal": task["goal"],
            "max_steps": task["max_steps"],
            "instruction": task["instruction"],
        },
    }


def build_dataset(split: Literal["train", "eval"] = "train") -> Dataset:
    tasks = TRAIN_TASKS if split == "train" else EVAL_TASKS
    rows = [_to_dataset_row(task) for task in tasks]
    return Dataset.from_list(rows)

