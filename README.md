# mobile-ui-env

Take-home assignment: a small mock mobile app where an agent outputs JSON UI actions, the simulator runs them, and a rubric scores the result. Built to work with Prime Intellect's `verifiers` library.

This is environment design only — no model training.

## Quick start

```bash
cd mobile_ui_env
pip install -e ".[dev]"
pytest
python run_eval.py
```

Docker:

```bash
docker build -t mobile-ui-env .
docker run --rm mobile-ui-env
docker run --rm mobile-ui-env python run_eval.py
```

LLM eval (optional):

```bash
export OPENAI_API_KEY=...
python run_eval.py --llm --model gpt-4o-mini
```

Verifiers entry point:

```python
from mobile_ui_env import load_environment
env = load_environment()  # vf.SingleTurnEnv
```

## What the app simulates

Four screens, element IDs taken from the assignment spec:

| Screen   | Elements |
|----------|----------|
| home     | notes_button, settings_button, profile_button |
| notes    | add_note_button, note_input, save_note_button, note_list |
| settings | focus_mode_toggle, notifications_toggle, version_label |
| profile  | username_label, email_label, logout_button |

The agent responds with a JSON list of `tap`, `type`, `back`, and `finish` actions. Invalid taps don't crash anything — they bump a counter and get penalized in the rubric.

Dataset: 20 train tasks, 10 eval tasks (`dataset.py`). Eval tasks use different wording/titles than train but the same goal types.

---

## Design notes (assignment questions)

### 1. State space

`AppState` in `state.py` holds everything the simulator needs:

- navigation: `current_screen` (`home` / `notes` / `settings` / `profile`)
- notes: `notes` list, plus `draft_note` / `draft_active` while composing
- settings: `focus_mode` (off by default), `notifications` (on by default)
- static profile info: username `alex_user`, email `alex@example.com`, version `2.1.0`
- episode tracking: `steps`, `invalid_actions`, `safety_violations`, `terminated`
- `finish_report` for read-back tasks (username, email, version)

The agent sees a plain-text observation built from this (`screen_observation()`), not raw state dicts.

### 2. Action space

JSON array. Each item is one of:

```json
{"action": "tap", "target": "notes_button"}
{"action": "type", "target": "note_input", "text": "Buy milk"}
{"action": "back"}
{"action": "finish"}
```

For tasks that need to return info (username, version, etc.), `finish` accepts an optional `report` field since the base action set has no separate "answer" action.

### 3. Episode termination

Episode stops when:

- agent sends `finish`
- step count hits `max_steps` for that task
- action list runs out

### 4. Sparse rewards

`success_reward` — binary. 1 if the goal is met, 0 otherwise. Most of the learning signal would have to come from elsewhere if you trained on this.

### 5. Dense / shaped rewards

Everything else is denser:

- `format_reward` — valid JSON + action schema (partial credit if some actions parse)
- `efficiency_reward` — fewer steps is better; small partial credit even on failure
- `invalid_action_penalty` — scales with bad taps/types
- `safety_penalty` — logout on a "don't logout" task
- `partial_progress_reward` — e.g. landed on the right screen, saved one of two notes

Combined (then clipped to [0, 1]):

```python
success + 0.1*format + 0.2*efficiency + 0.1*partial
  - 0.2*invalid_penalty - 0.3*safety_penalty
```

In the Verifiers rubric, penalty functions return positive magnitudes with negative weights.

### 6. Reward hacking

A few obvious ones:

- Call `finish` immediately → low success, but avoids step costs if efficiency weighting is wrong
- Pretty JSON full of invalid taps → format_reward stays high
- Navigate to profile without reading the label → partial_progress helps but shouldn't beat success
- Tap logout then finish anyway → safety_penalty should dominate on those tasks

Shaped rewards help learning but need tuning; success should stay the main signal.

### 7. Mock UI → real Android emulator

Keep the same task definitions and rubric. Swap the simulator backend:

1. **Observations** — accessibility tree (UIAutomator XML), maybe screenshots. Map nodes to stable element IDs.
2. **Actions** — translate tap/type/back to ADB or Appium calls.
3. **State** — refresh hierarchy after each action instead of updating an in-memory dict.
4. **Rewards** — same goal checks, but read from parsed emulator state.
5. **Ops** — timeouts, retries, stale element refs (real UIs flake).

### 8. Prime Intellect / Verifiers / PRIME-RL

`load_environment()` in `env.py` returns `vf.SingleTurnEnv` with train + eval datasets and a weighted `vf.Rubric`. Reward funcs are async, take `completion` + `info`, parse the model's JSON, run the simulator, return a float.

Single-turn here means the model outputs the full action plan at once; the rubric replays it internally. Fine for this assignment, though a real mobile agent would probably want multi-turn feedback.

Should plug into `prime env install` / `prime eval run`. PRIME-RL would reuse the same package; the trainer owns sampling and policy updates.

### 9. Tests

- `test_actions.py` — screen transitions, invalid tap doesn't crash, note flow, bad JSON
- `test_rewards.py` — success on completed note, safety penalty on logout, `finish.report` for reads
- `test_env.py` — `load_environment()` type, 20/10 split, dataset fields

Run: `pytest` (15 tests).

### 10. Scope tradeoffs

- Single-turn planning instead of `MultiTurnEnv` with per-step observations
- Text observations only — no screenshots or XML
- `finish.report` added for read tasks (not in original 4 action types)
- `run_eval.py` uses a hand-written heuristic, not a trained policy
- 30 hand-written tasks, not procedural generation
- Rubric re-runs the simulator per reward component (simple but redundant)

---

## Layout

```
mobile_ui_env/
  env.py          load_environment()
  state.py        AppState, screen defs, observations
  actions.py      parse + execute actions
  simulator.py    run episode, check goals
  dataset.py      30 tasks
  rubric.py       reward functions
  run_eval.py     eval script
  tests/
```
