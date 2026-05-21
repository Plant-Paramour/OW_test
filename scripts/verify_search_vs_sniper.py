"""搜索算法直接决策 vs Sniper —— 验证搜索本身能否击败 Sniper。
用法: "C:\ProgramData\anaconda3\envs\Orbit_Wars\python.exe" "C:\code\[kaggle]\Orbit Wars\scripts\verify_search_vs_sniper.py"
"""
import io, sys, time
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
sys.path.insert(0, r"C:\code\[kaggle]\Orbit Wars")

import numpy as np
from kaggle_environments import make

from src.world.observation import parse_observation
from src.world.fleet_tracker import build_arrival_ledger
from src.world.combat import simulate_planet_timeline
from src.engine.interception import aim_at
from src.search.search import search_best_actions
from src.env.reward import state_potential


def search_agent(observation, configuration):
    """搜索算法直接决策的 agent。"""
    state = parse_observation(observation, episode_steps=50)

    ledger = build_arrival_ledger(state.fleets, state.planets)
    timelines = {}
    for p in state.planets:
        timelines[p.id] = simulate_planet_timeline(
            p, ledger.get(p.id, []), state.player, state.remaining_steps)

    results = search_best_actions(state, ledger, timelines, top_k=20)

    actions = []
    used_sources = set()

    for r in results:
        if r.source_id in used_sources:
            continue
        if r.value <= 0:
            continue

        src = next((p for p in state.planets if p.id == r.source_id), None)
        tgt = next((p for p in state.planets if p.id == r.target_id), None)
        if src is None or tgt is None:
            continue

        aim = aim_at(src, tgt, r.ships,
                     state.initial_by_id, state.angular_velocity,
                     state.comets, state.comet_ids)
        if aim is None:
            continue

        angle = aim[0]
        actions.append([r.source_id, angle, r.ships])
        used_sources.add(r.source_id)

    return actions


def run_eval(n_games=20):
    """Run search vs sniper and report win rate."""
    from src.opponents.sniper import SniperOpponent

    sniper = SniperOpponent()
    wins = 0
    total_phi = 0.0

    for g in range(n_games):
        env = make("orbit_wars", debug=True,
                   configuration={"episodeSteps": 50})
        env.run([search_agent, sniper])

        final_obs = env.steps[-1][0].observation
        final_state = parse_observation(final_obs, player_override=0)
        phi = state_potential(final_state, player=0)
        total_phi += phi

        my_planets = len([p for p in final_state.planets if p.owner == 0])
        enemy_planets = len([p for p in final_state.planets if p.owner == 1])

        won = phi > 0
        if won:
            wins += 1

        status = "WIN" if won else "LOSE"
        print(f"  Game {g+1:2d}: {status}  Φ={phi:+.2f}  "
              f"my_planets={my_planets} enemy_planets={enemy_planets}")

    wr = wins / n_games
    print(f"\n胜率: {wins}/{n_games} = {wr:.1%}")
    print(f"平均 Φ: {total_phi/n_games:+.2f}")
    return wr


if __name__ == "__main__":
    start = time.time()
    print("=== 搜索算法 vs Sniper (50回合) ===\n")
    run_eval(20)
    print(f"\n耗时: {time.time()-start:.0f}s")
