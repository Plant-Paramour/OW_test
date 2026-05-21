"""临时调试脚本 — 追踪 4 人模式彗星生成和消失回合。"""
import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
sys.path.insert(0, r"C:\code\[kaggle]\Orbit Wars")

from kaggle_environments import make

env4 = make('orbit_wars', configuration={'episodeSteps': 200}, debug=True)
env4.reset(4)
print(f"Config episodeSteps: {env4.configuration['episodeSteps']}")

# Track comets throughout the game
for step_idx in range(300):
    if env4.done:
        print(f"Game ended at step {step_idx}")
        break
    actions = [[], [], [], []]
    env4.step(actions)
    obs = env4.steps[-1][0].observation
    step = obs['step']
    comets = obs.get('comets', [])
    if comets:
        for g in comets:
            pids = g.get('planet_ids', [])
            pi = g.get('path_index', -1)
            paths = g.get('paths', [])
            # Check if any active comets
            active = []
            for i, pid in enumerate(pids):
                if i < len(paths) and pi < len(paths[i]):
                    active.append(pid)
            if active:
                print(f"  step={step}: active={active}, path_index={pi}")
    if step >= 220:
        break

print(f"\nFinal step: {step}")
print(f"Done: {env4.done}")
print(f"Total steps in env: {len(env4.steps)}")
