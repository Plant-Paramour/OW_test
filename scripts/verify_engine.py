"""验证物理引擎模块的正确性。"""
import sys
import io
sys.path.insert(0, r"C:\code\[kaggle]\Orbit Wars")
# Fix GBK encoding issues on Windows
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

import math
import random
import numpy as np
from src.engine.constants import *
from src.engine.physics import *
from src.engine.prediction import *
from src.engine.interception import *

# Fixed seeds for reproducibility
random.seed(42)
np.random.seed(42)

# For testing with the actual game
from kaggle_environments import make
from kaggle_environments.envs.orbit_wars.orbit_wars import Planet

passed = 0
failed = 0

def check(name, actual, expected, tol=0.001):
    global passed, failed
    if abs(actual - expected) <= tol:
        passed += 1
        print(f"  [OK] {name}: {actual:.3f} == {expected:.3f}")
    else:
        failed += 1
        print(f"  [FAIL] {name}: got {actual:.3f}, expected {expected:.3f}")

def check_bool(name, actual, expected):
    global passed, failed
    if actual == expected:
        passed += 1
        print(f"  [OK] {name}: {actual} == {expected}")
    else:
        failed += 1
        print(f"  [FAIL] {name}: got {actual}, expected {expected}")

print("=" * 60)
print("1. Constants")
print("=" * 60)
check("BOARD_SIZE", BOARD_SIZE, 100.0)
check("CENTER_X", CENTER_X, 50.0)
check("SUN_RADIUS", SUN_RADIUS, 10.0)
check("MAX_SPEED", MAX_SPEED, 6.0)
check("TOTAL_STEPS", TOTAL_STEPS, 500)

print()
print("=" * 60)
print("2. Physics: fleet_speed")
print("=" * 60)
check("speed(1)", fleet_speed(1), 1.0)
check_bool("speed(10) > 1", fleet_speed(10) > 1.0, True)
check_bool("speed(10) < MAX_SPEED", fleet_speed(10) < MAX_SPEED, True)
check_bool("speed(1000) ≈ MAX_SPEED", abs(fleet_speed(1000) - MAX_SPEED) < 0.001, True)
check_bool("speed(10) < speed(100)", fleet_speed(10) < fleet_speed(100), True)
check_bool("speed(100) < speed(500)", fleet_speed(100) < fleet_speed(500), True)

print()
print("=" * 60)
print("3. Physics: dist & point_to_segment_distance")
print("=" * 60)
check("dist(0,0,3,4)", dist(0, 0, 3, 4), 5.0)
check("dist(50,50,80,90)", dist(50, 50, 80, 90), 50.0)

# Point to segment: point at (0,0), segment from (3,0) to (3,4)
check("pt_to_seg perpendicular", point_to_segment_distance(0, 0, 3, 0, 3, 4), 3.0)
# Point on segment
check("pt_to_seg on segment", point_to_segment_distance(3, 2, 0, 0, 6, 0), 2.0)

print()
print("=" * 60)
print("4. Physics: segment_hits_sun")
print("=" * 60)
# Segment through the sun center
check_bool("through sun center", segment_hits_sun(30, 50, 70, 50), True)
# Segment far from sun
check_bool("far from sun", segment_hits_sun(10, 10, 90, 10), False)
# Segment near but misses sun
check_bool("near miss", segment_hits_sun(50, 0, 50, 38), False)

print()
print("=" * 60)
print("5. Prediction: is_static_planet")
print("=" * 60)
# Create test planets
inner = Planet(0, -1, 50, 30, 2, 10, 3)   # near center → rotating
outer = Planet(1, -1, 90, 10, 8, 10, 3)    # orbital_r=44.7, +8 > 50 → static
check_bool("inner planet (r=22) is rotating", is_static_planet(inner), False)
check_bool("outer planet (r=44.7, rad=8 > 50) is static", is_static_planet(outer), True)

print()
print("=" * 60)
print("5.5 Prediction: comet functions")
print("=" * 60)

# Mock comet data
mock_comets = [
    {
        "planet_ids": [100, 101],
        "paths": [
            [(60.0, 30.0), (62.0, 31.0), (64.0, 32.0), (66.0, 33.0), (68.0, 34.0)],
            [(40.0, 70.0), (38.0, 69.0), (36.0, 68.0), (34.0, 67.0), (32.0, 66.0)],
        ],
        "path_index": 0,
    }
]
mock_comet_ids = {100, 101}

check("comet_remaining_life(100) = 5", comet_remaining_life(100, mock_comets), 5.0)
check("comet_remaining_life(101) = 5", comet_remaining_life(101, mock_comets), 5.0)
check("comet_remaining_life(999) = 0", comet_remaining_life(999, mock_comets), 0.0)

pos = predict_comet_position(100, mock_comets, 0)
check_bool("predict_comet(100, t=0) is not None", pos is not None, True)
if pos:
    check("predict_comet(100, t=0).x", pos[0], 60.0)
    check("predict_comet(100, t=0).y", pos[1], 30.0)

pos2 = predict_comet_position(100, mock_comets, 2)
check_bool("predict_comet(100, t=2) is not None", pos2 is not None, True)
if pos2:
    check("predict_comet(100, t=2).x", pos2[0], 64.0)
    check("predict_comet(100, t=2).y", pos2[1], 32.0)

pos_oob = predict_comet_position(100, mock_comets, 10)
check_bool("predict_comet(100, t=10) → None (out of bounds)", pos_oob is None, True)

pos_wrong_id = predict_comet_position(999, mock_comets, 0)
check_bool("predict_comet(999) → None (not a comet)", pos_wrong_id is None, True)

# Test comet_remaining_life with path_index advanced
mock_comets_mid = [
    {
        "planet_ids": [200],
        "paths": [[(10.0, 20.0), (11.0, 21.0), (12.0, 22.0), (13.0, 23.0), (14.0, 24.0)]],
        "path_index": 3,
    }
]
check("comet_remaining_life(path_index=3, len=5) = 2",
      comet_remaining_life(200, mock_comets_mid), 2.0)

# Test predict_comet_position with non-zero path_index
pos_mid = predict_comet_position(200, mock_comets_mid, 1)
check_bool("predict_comet(path_index=3, t=1) is not None", pos_mid is not None, True)
if pos_mid:
    check("predict_comet(path_index=3, t=1).x", pos_mid[0], 14.0)
    check("predict_comet(path_index=3, t=1).y", pos_mid[1], 24.0)

# Test predict_comet_position out of bounds with path_index > 0
pos_mid_oob = predict_comet_position(200, mock_comets_mid, 3)
check_bool("predict_comet(path_index=3, t=3) → None", pos_mid_oob is None, True)

print()
print("=" * 60)
print("5.6 Interception: aim_at basics")
print("=" * 60)

# Test aim_at between two planets on the same side of the sun (safe path)
src_planet = Planet(10, -1, 15, 15, 2, 100, 5)
tgt_planet = Planet(11, -1, 85, 20, 2, 50, 5)
initial_by_id = {
    10: Planet(10, -1, 15, 15, 2, 100, 5),
    11: Planet(11, -1, 85, 20, 2, 50, 5),
}

result = aim_at(src_planet, tgt_planet, 50, initial_by_id, 0.0, [], set())
check_bool("aim_at static→static returns result", result is not None, True)
if result:
    angle, turns, pred_x, pred_y = result
    check("angle near 0 (straight right)", abs(angle), 0.0, tol=0.15)
    check_bool("turns > 0", turns > 0, True)

# Test aim_at returns None for sun-blocked path with static planets
# Static planets: orbital_r + radius >= 50 → aim_at returns None directly
sun_blocked_src = Planet(20, -1, 50, 20, 22, 50, 5)
sun_blocked_tgt = Planet(21, -1, 50, 80, 22, 50, 5)
result_blocked = aim_at(sun_blocked_src, sun_blocked_tgt, 50,
                         {20: sun_blocked_src, 21: sun_blocked_tgt},
                         0.0, [], set())
check_bool("aim_at through sun → None", result_blocked is None, True)

print()
print("=" * 60)
print("6. Integration: run a game with engine validation")
print("=" * 60)
env = make("orbit_wars", debug=True)
env.run(["random", "random"])

obs = env.steps[1][0].observation
planets = [Planet(*p) for p in obs.planets]

# Verify fleet_speed produces consistent results
for ships in [1, 10, 30, 50, 100, 200, 500]:
    speed = fleet_speed(ships)
    print(f"  ships={ships:4d} → speed={speed:.3f} units/turn")

# Verify at least one path check works with real planets
if len(planets) >= 2:
    p1, p2 = planets[0], planets[1]
    d = dist(p1.x, p1.y, p2.x, p2.y)
    print(f"  distance(planet{p1.id}, planet{p2.id}) = {d:.1f}")

    # Check sun collision
    hits = segment_hits_sun(p1.x, p1.y, p2.x, p2.y)
    print(f"  segment_hits_sun = {hits}")

print()
print("=" * 60)
print(f"RESULTS: {passed} passed, {failed} failed out of {passed + failed}")
print("=" * 60)

if failed > 0:
    sys.exit(1)
else:
    print("All checks passed! Engine is ready.")
