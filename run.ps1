# Everything, end to end. PowerShell.
#   .\run.ps1
# Leave MUJOCO_GL unset on Windows: MuJoCo uses wgl and your GPU.
$ErrorActionPreference = "Stop"

Write-Host "--- friction physics check ---" -ForegroundColor Cyan
python swarm_sim.py --selftest

Write-Host "--- main demo, 6 fingers, position control ---" -ForegroundColor Cyan
python swarm_sim.py --task demo --fingers 6 --out out/swarm_demo.mp4

Write-Host "--- minimise fingers for the corner balance ---" -ForegroundColor Cyan
python min_fingers.py --task corner --phase "BALANCE ON CORNER" --samples 4 --per-face 9

Write-Host "--- minimise for the whole corner task, then render it ---" -ForegroundColor Cyan
python min_fingers.py --task corner --samples 8 --per-face 4 --out contacts_corner_full.json
python swarm_sim.py --task corner --control hybrid `
    --contacts contacts_corner_full.json --out out/corner_balance.mp4
