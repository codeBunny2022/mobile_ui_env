from __future__ import annotations

import verifiers as vf

from mobile_ui_env.dataset import ACTION_SPEC, build_dataset
from mobile_ui_env.rubric import build_rubric

SYSTEM_PROMPT = f"""You control a mock mobile app. Return one JSON array of actions that completes the task.

{ACTION_SPEC}

JSON only, no explanation.
"""


def load_environment():
    dataset = build_dataset(split="train")
    eval_dataset = build_dataset(split="eval")
    rubric = build_rubric()
    return vf.SingleTurnEnv(
        dataset=dataset,
        eval_dataset=eval_dataset,
        system_prompt=SYSTEM_PROMPT,
        rubric=rubric,
    )
