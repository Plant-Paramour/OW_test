"""临时调试脚本 — 对比 1v1 和 4p 模式彗星。"""
import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
sys.path.insert(0, r"C:\code\[kaggle]\Orbit Wars")

from kaggle_environments import make

# 1v1 mode
print("=" * 60)
print("1v1 mode")
print("=" * 60)
env2 = make('orbit_wars', debug=True)
env2.reset(2)
print(f"episodeSteps: {env2.configuration['episodeSteps']}")
for _ in range(500):
    if env2.done:
        break
    env2.step([[], []])
print(f"Game ended at step {env2.steps[-1][0].observation['step']}")
print(f"Total env steps: {len(env2.steps)}")

# List comet appearances
comet_steps = set()
for s in env2.steps:
    obs = s[0].observation
    for g in obs.get('comets', []):
        comet_steps.add(obs['step'])
print(f"Steps with active comets: {sorted(comet_steps)}")

print()

# 4p mode
print("=" * 60)
print("4p mode")
print("=" * 60)
env4 = make('orbit_wars', configuration={'episodeSteps': 200}, debug=True)
env4.reset(4)
print(f"episodeSteps: {env4.configuration['episodeSteps']}")
for _ in range(250):
    if env4.done:
        break
    env4.step([[], [], [], []])
print(f"Game ended at step {env4.steps[-1][0].observation['step']}")
print(f"Total env steps: {len(env4.steps)}")

comet_steps4 = set()
for s in env4.steps:
    obs = s[0].observation
    for g in obs.get('comets', []):
        comet_steps4.add(obs['step'])
print(f"Steps with active comets: {sorted(comet_steps4)}")
