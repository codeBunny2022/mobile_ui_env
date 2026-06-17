# AI usage

## What I used AI for

- Reading the assignment PDF and sketching an implementation plan
- Bootstrapping the package layout and boilerplate (pyproject, module files, Dockerfile)
- First pass on reward functions, dataset rows, tests, and README answers
- Checking how Prime Intellect `vf.Rubric` / `vf.SingleTurnEnv` are supposed to be wired up

## What I kept vs changed

Most of the structure came from AI-assisted drafting. I went through the simulator logic manually screen transitions, invalid action handling, note draft flow and fixed things against the assignment's element tables. Rubric weights and the `finish.report` extension for read tasks were adjusted by hand. README was rewritten to sound less templated.

## What I actually learned

- Verifiers single-turn envs can score a full action sequence by replaying it inside reward functions
- Splitting the simulator from the Verifiers wrapper makes pytest much easier
- Sparse success + shaped aux rewards is easy to write but easy to mis-tune
- Invalid actions need to fail soft; crashing on a bad tap would be unusable for RL rollouts

