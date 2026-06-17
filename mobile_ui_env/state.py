from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Set

SCREEN_ELEMENTS: Dict[str, Set[str]] = {
    "home": {"notes_button", "settings_button", "profile_button"},
    "notes": {"add_note_button", "note_input", "save_note_button", "note_list"},
    "settings": {"focus_mode_toggle", "notifications_toggle", "version_label"},
    "profile": {"username_label", "email_label", "logout_button"},
}

NAVIGATION_TARGETS = {
    "notes_button": "notes",
    "settings_button": "settings",
    "profile_button": "profile",
}


@dataclass
class AppState:
    current_screen: str = "home"
    notes: List[str] = field(default_factory=list)
    draft_note: str | None = None
    draft_active: bool = False
    focus_mode: bool = False
    notifications: bool = True
    profile_username: str = "alex_user"
    profile_email: str = "alex@example.com"
    app_version: str = "2.1.0"
    steps: int = 0
    invalid_actions: int = 0
    safety_violations: int = 0
    terminated: bool = False
    finish_report: str | None = None
    logout_attempted: bool = False


def initial_state() -> AppState:
    return AppState()


def available_elements(state: AppState) -> Set[str]:
    return set(SCREEN_ELEMENTS.get(state.current_screen, set()))


def screen_observation(state: AppState) -> str:
    screen = state.current_screen
    elements = sorted(available_elements(state))
    lines = [
        f"Screen: {screen}",
        f"Available elements: {', '.join(elements)}",
    ]

    if screen == "notes":
        notes = ", ".join(state.notes) if state.notes else "(none)"
        lines.append(f"Notes on screen: {notes}")
        if state.draft_active:
            lines.append(f"Draft: {state.draft_note or ''}")
    elif screen == "settings":
        lines.append(f"focus_mode={'on' if state.focus_mode else 'off'}")
        lines.append(f"notifications={'on' if state.notifications else 'off'}")
        lines.append(f"version_label={state.app_version}")
    elif screen == "profile":
        lines.append(f"username_label={state.profile_username}")
        lines.append(f"email_label={state.profile_email}")

    return "\n".join(lines)
