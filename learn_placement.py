"""
Learn where to put the fingertips, by search rather than by heuristic or by LP.

The cross-entropy method (CEM) over contact placement. Sample a population of candidate
placements, run each one through the simulator, keep the best fraction, refit the
sampling distribution to them, repeat. The simulator is the fitness function.

Why this and not the linear program in min_fingers.py
-----------------------------------------------------
The LP answers "can these contacts produce the required wrench", which is a necessary
condition and turned out to be a badly insufficient one. It declared three contacts
enough for the corner balance; the simulator jammed the cube against the table. CEM
optimises the thing we actually care about, which is whether the controller holds the
object, and it does not need the criterion to be right a priori. It is slower and it
gives no proof, but it cannot be confidently wrong in the way the LP was.

The parameterisation
--------------------
Each contact is a free 3-vector, ray-cast from the cube centre onto its surface by
`swarm_sim.surface_point`. So n contacts is 3n parameters, continuous, with no face
indices to make discrete. This matters: the map covers edges and vertices, where the
contact normal becomes the bisector of the adjoining faces. The LP's face-grid candidate
set could not express a vertex contact, which is precisely the contact that turned out to
matter for balancing on a corner. This one can find it.

Usage:
    python learn_placement.py --fingers 4 --generations 12 --population 32
    python learn_placement.py --fingers 4 --min-fingers      # then try to go smaller
    python learn_placement.py --replay best_placement_n4.json --render out/learned.mp4
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import time

import numpy as np

os.environ.setdefault("MUJOCO_GL", "osmesa")

LOST_MM = 25.0          # position error above this counts as losing the object
BIG = 1e4               # fitness penalty for a lost or diverged rollout


# --------------------------------------------------------------------------------------
# Fitness
# --------------------------------------------------------------------------------------

def contacts_from_params(params):
    """3n parameters -> n (offset, normal) contacts on the object surface."""
    import swarm_sim as S
    p = np.asarray(params, float).reshape(-1, 3)
    return [S.surface_point(v) for v in p]


def rollout(params, task="probe", collect=False):
    """Run one candidate placement. Returns (cost, info)."""
    import swarm_sim as S
    S.set_task(task)
    try:
        S.set_finger_count(0, contacts=contacts_from_params(params))
    except Exception as exc:                     # degenerate geometry
        return BIG * 10, {"error": repr(exc)}

    try:
        sim = S.Sim("dynamic")
    except Exception as exc:                     # model would not compile
        return BIG * 10, {"error": repr(exc)}

    errs, grips = [], []
    lost = False
    while sim.data.time < sim.ref.duration:
        sim.step()
        t = sim.data.time
        if not np.all(np.isfinite(sim.data.qpos)):
            lost = True
            break
        if t <= S.GRASP_END:
            continue
        e_p, e_r = sim.pose_error(t)
        mm = float(np.linalg.norm(e_p) * 1000.0)
        errs.append((mm, float(np.rad2deg(np.linalg.norm(e_r)))))
        if collect:
            grips.append(sim.grip_forces().copy())
        if mm > LOST_MM:
            lost = True
            break

    if lost or not errs:
        # Rank failures by how far they got, so the search has a gradient to follow
        # instead of a flat wall of identical penalties.
        frac = (sim.data.time / sim.ref.duration) if sim.ref.duration else 0.0
        return BIG * (2.0 - frac), {"lost": True, "survived": round(frac, 3)}

    v = np.array(errs)
    rms_pos = float(np.sqrt((v[:, 0] ** 2).mean()))
    rms_rot = float(np.sqrt((v[:, 1] ** 2).mean()))
    max_pos = float(v[:, 0].max())
    # Position in mm and rotation in degrees are already comparable magnitudes here,
    # so a simple weighted sum is honest enough. Max error is included so a candidate
    # cannot win by being good on average and terrible once.
    cost = rms_pos + 2.0 * rms_rot + 0.3 * max_pos
    info = {"lost": False, "rms_pos_mm": round(rms_pos, 3),
            "rms_rot_deg": round(rms_rot, 3), "max_pos_mm": round(max_pos, 3)}
    if collect:
        g = np.array(grips)
        info["mean_grip_N"] = [round(x, 2) for x in g.mean(axis=0)]
        info["min_contacts"] = int((g > 0.05).sum(axis=1).min())
    return cost, info


def _worker(args):
    params, task = args
    try:
        return rollout(params, task)[0]
    except Exception:
        return BIG * 10


# --------------------------------------------------------------------------------------
# CEM
# --------------------------------------------------------------------------------------

def cem(n_fingers, task="probe", generations=12, population=32, elite_frac=0.25,
        sigma0=0.6, workers=None, seed=0, log=None):
    """Cross-entropy method over 3n placement parameters."""
    rng = np.random.default_rng(seed)
    dim = 3 * n_fingers

    # Seed the mean from the existing heuristic layout, so the search starts from
    # something workable rather than from noise. CEM will move off it if it can do better.
    import swarm_sim as S
    S.set_task(task)
    S.set_finger_count(n_fingers)
    mean = np.array([nrm for _, nrm in S.CONTACTS], float).reshape(-1)
    sigma = np.full(dim, sigma0)

    n_elite = max(2, int(round(population * elite_frac)))
    workers = workers or max(1, (os.cpu_count() or 2))
    history = []

    print(f"CEM  n_fingers={n_fingers}  task={task}  dim={dim}  pop={population}  "
          f"elite={n_elite}  workers={workers}")
    base_cost, base_info = rollout(mean, task)
    print(f"  heuristic baseline: cost {base_cost:9.3f}   {base_info}\n")

    best_cost, best_params, best_info = base_cost, mean.copy(), base_info

    with mp.Pool(workers) as pool:
        for gen in range(generations):
            t0 = time.time()
            pop = rng.normal(mean, sigma, size=(population, dim))
            pop[0] = mean                         # always keep the incumbent
            costs = np.array(pool.map(_worker, [(p, task) for p in pop]))

            order = np.argsort(costs)
            elite = pop[order[:n_elite]]
            mean = elite.mean(axis=0)
            # Floor the spread so the distribution cannot collapse in one good generation
            # and stop exploring.
            sigma = np.maximum(elite.std(axis=0), 0.03)

            if costs[order[0]] < best_cost:
                best_cost = float(costs[order[0]])
                best_params = pop[order[0]].copy()
                best_cost, best_info = rollout(best_params, task)

            n_ok = int((costs < BIG).sum())
            history.append({"gen": gen, "best": float(costs[order[0]]),
                            "elite_mean": float(costs[order[:n_elite]].mean()),
                            "feasible": n_ok, "sigma": float(sigma.mean())})
            print(f"  gen {gen:2d}  best {costs[order[0]]:9.3f}   "
                  f"elite mean {costs[order[:n_elite]].mean():9.3f}   "
                  f"held {n_ok:3d}/{population}   sigma {sigma.mean():.3f}   "
                  f"{time.time()-t0:5.1f}s")
            if log:
                with open(log, "w") as fh:
                    json.dump(history, fh, indent=2)

    print(f"\n  best cost {best_cost:.3f}   {best_info}")
    improved = best_cost < base_cost
    print(f"  {'IMPROVED on' if improved else 'did NOT beat'} the heuristic baseline "
          f"({base_cost:.3f})")
    return best_params, best_cost, best_info, history


def save(params, path, n, task, cost, info):
    cs = contacts_from_params(params)
    spec = {"source": "learn_placement.py (CEM)", "task": task, "n_fingers": n,
            "cost": cost, "info": info,
            "params": [float(x) for x in np.asarray(params).reshape(-1)],
            "contacts": [{"offset": [float(x) for x in o],
                          "normal": [float(x) for x in nn]} for o, nn in cs]}
    with open(path, "w") as fh:
        json.dump(spec, fh, indent=2)
    print(f"  wrote {path}")


# --------------------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fingers", type=int, default=4)
    ap.add_argument("--task", default="probe")
    ap.add_argument("--generations", type=int, default=12)
    ap.add_argument("--population", type=int, default=32)
    ap.add_argument("--workers", type=int, default=None)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--min-fingers", action="store_true",
                    help="after converging at --fingers, keep decrementing while the "
                         "search still finds a placement that holds")
    ap.add_argument("--out", default=None)
    ap.add_argument("--replay", default=None,
                    help="JSON from a previous run; re-score it and report")
    args = ap.parse_args()

    if args.replay:
        spec = json.load(open(args.replay))
        cost, info = rollout(np.array(spec["params"]), spec.get("task", args.task),
                             collect=True)
        print(f"replay {args.replay}: cost {cost:.3f}")
        print(json.dumps(info, indent=2))
        return 0

    n = args.fingers
    while True:
        best, cost, info, _ = cem(n, args.task, args.generations, args.population,
                                  workers=args.workers, seed=args.seed,
                                  log=f"cem_log_n{n}.json")
        out = args.out or f"best_placement_n{n}.json"
        save(best, out, n, args.task, cost, info)
        if not args.min_fingers:
            break
        if cost >= BIG:
            print(f"\n{n} fingers: no placement found that holds. Stopping.")
            break
        print(f"\n{n} fingers works. Trying {n-1}.\n")
        n -= 1
        if n < 2:
            break
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
