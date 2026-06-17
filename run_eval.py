#!/usr/bin/env python3
from __future__ import annotations

import argparse
import asyncio
import json
import os
from typing import Any, Dict, List

from mobile_ui_env.dataset import build_dataset
from mobile_ui_env.rubric import (
    compute_final_reward,
    efficiency_reward,
    format_reward,
    invalid_action_penalty,
    partial_progress_reward,
    safety_penalty,
    success_reward,
)
from mobile_ui_env.simulator import goal_satisfied, run_episode_from_completion


def heuristic_baseline(task: Dict[str, Any]) -> str:
    goal = task["goal"]
    goal_type = goal.get("type")

    if goal_type == "note_created":
        title = goal["title"]
        actions = [
            {"action": "tap", "target": "notes_button"},
            {"action": "tap", "target": "add_note_button"},
            {"action": "type", "target": "note_input", "text": title},
            {"action": "tap", "target": "save_note_button"},
            {"action": "finish"},
        ]
    elif goal_type == "multi_note_created":
        actions = [{"action": "tap", "target": "notes_button"}]
        for title in goal["titles"]:
            actions.extend(
                [
                    {"action": "tap", "target": "add_note_button"},
                    {"action": "type", "target": "note_input", "text": title},
                    {"action": "tap", "target": "save_note_button"},
                ]
            )
        actions.append({"action": "finish"})
    elif goal_type == "toggle_setting":
        setting = goal["setting"]
        want_on = bool(goal["value"])
        actions = [{"action": "tap", "target": "settings_button"}]
        # defaults: focus_mode off, notifications on
        if setting == "focus_mode" and want_on:
            actions.append({"action": "tap", "target": "focus_mode_toggle"})
        elif setting == "notifications" and not want_on:
            actions.append({"action": "tap", "target": "notifications_toggle"})
        actions.append({"action": "finish"})
    elif goal_type == "read_profile_field":
        expected = goal["expected"]
        actions = [
            {"action": "tap", "target": "profile_button"},
            {"action": "finish", "report": expected},
        ]
    elif goal_type == "read_version":
        actions = [
            {"action": "tap", "target": "settings_button"},
            {"action": "finish", "report": goal.get("expected", "2.1.0")},
        ]
    elif goal_type == "navigate_without_logout":
        actions = [
            {"action": "tap", "target": "profile_button"},
            {"action": "finish"},
        ]
    else:
        actions = [{"action": "finish"}]

    return json.dumps(actions)


async def llm_baseline(prompt: List[Dict[str, str]], model: str) -> str:
    try:
        from openai import OpenAI
    except ImportError as exc:
        raise RuntimeError("Install openai package for --llm mode: pip install openai") from exc

    client = OpenAI(
        api_key=os.environ.get("OPENAI_API_KEY"),
        base_url=os.environ.get("OPENAI_API_BASE"),
    )
    response = client.chat.completions.create(
        model=model,
        messages=prompt,
        temperature=0.0,
    )
    return response.choices[0].message.content or ""


async def score_task(task: Dict[str, Any], completion_text: str) -> Dict[str, float]:
    info = {
        "task_id": task["task_id"],
        "goal": task["goal"],
        "max_steps": task["max_steps"],
        "instruction": task["instruction"],
    }
    completion = [{"role": "assistant", "content": completion_text}]

    components = {
        "success_reward": await success_reward(completion, info),
        "format_reward": await format_reward(completion, info),
        "efficiency_reward": await efficiency_reward(completion, info),
        "invalid_action_penalty": await invalid_action_penalty(completion, info),
        "safety_penalty": await safety_penalty(completion, info),
        "partial_progress_reward": await partial_progress_reward(completion, info),
    }
    components["final_reward"] = compute_final_reward(components)

    state, _ = run_episode_from_completion(completion_text, max_steps=task["max_steps"])
    components["steps"] = float(state.steps)
    components["invalid_actions"] = float(state.invalid_actions)
    components["safety_violations"] = float(state.safety_violations)
    components["success"] = 1.0 if goal_satisfied(state, task["goal"]) else 0.0
    return components


async def run_eval(use_llm: bool = False, model: str = "gpt-4o-mini") -> None:
    dataset = build_dataset(split="eval")
    results: List[Dict[str, float]] = []

    for row in dataset:
        task = {
            "task_id": row["task_id"],
            "instruction": row["instruction"],
            "goal": row["goal"],
            "max_steps": row["max_steps"],
        }
        if use_llm:
            completion_text = await llm_baseline(row["prompt"], model=model)
        else:
            completion_text = heuristic_baseline(task)

        scored = await score_task(task, completion_text)
        results.append(scored)

    total = len(results)
    success_rate = sum(r["success"] for r in results) / total if total else 0.0
    avg_reward = sum(r["final_reward"] for r in results) / total if total else 0.0
    avg_steps = sum(r["steps"] for r in results) / total if total else 0.0
    total_actions = sum(r["steps"] + r["invalid_actions"] for r in results)
    invalid_rate = (
        sum(r["invalid_actions"] for r in results) / total_actions if total_actions else 0.0
    )
    safety_violations = int(sum(r["safety_violations"] for r in results))

    print(f"Total eval tasks: {total}")
    print(f"Success rate: {success_rate:.0%}")
    print(f"Average reward: {avg_reward:.2f}")
    print(f"Average steps: {avg_steps:.1f}")
    print(f"Invalid action rate: {invalid_rate:.2f}")
    print(f"Safety violations: {safety_violations}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--llm", action="store_true", help="call an OpenAI-compatible API")
    parser.add_argument("--model", default="gpt-4o-mini")
    args = parser.parse_args()
    asyncio.run(run_eval(use_llm=args.llm, model=args.model))


if __name__ == "__main__":
    main()
