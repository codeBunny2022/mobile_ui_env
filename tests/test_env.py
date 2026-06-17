import verifiers as vf

from mobile_ui_env.dataset import build_dataset
from mobile_ui_env.env import load_environment


def test_load_environment_returns_single_turn_env():
    env = load_environment()
    assert isinstance(env, vf.SingleTurnEnv)


def test_train_dataset_has_20_tasks():
    dataset = build_dataset(split="train")
    assert len(dataset) == 20


def test_eval_dataset_has_10_tasks():
    dataset = build_dataset(split="eval")
    assert len(dataset) == 10


def test_dataset_rows_have_required_fields():
    dataset = build_dataset(split="train")
    row = dataset[0]
    assert "task_id" in row
    assert "instruction" in row
    assert "goal" in row
    assert "max_steps" in row
    assert "prompt" in row
    assert "info" in row
