#!/usr/bin/env bash
# Everything, end to end. Assumes: conda activate fingersim
set -e
# MUJOCO_GL is auto-detected: forced to osmesa only on headless Linux. Override if needed.

echo "--- friction physics check ---"
python3 swarm_sim.py --selftest

echo "--- main demo, 6 fingers, position control ---"
python3 swarm_sim.py --task demo --fingers 6 --out out/swarm_demo.mp4

echo "--- minimise fingers for the corner balance ---"
python3 min_fingers.py --task corner --phase "BALANCE ON CORNER" --samples 4 --per-face 9

echo "--- minimise fingers for the whole corner task, then render it ---"
python3 min_fingers.py --task corner --samples 8 --per-face 4 --out contacts_corner_full.json
python3 swarm_sim.py --task corner --control hybrid \
    --contacts contacts_corner_full.json --out out/corner_balance.mp4
