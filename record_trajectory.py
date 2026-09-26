"""
Author a trajectory by dragging the object with the mouse, then hand it to the search.

Phase 1, record.  The object is loaded ALONE, as a mocap body, with no fingers and no
    gravity acting on it. A mocap body follows what you tell it exactly, so dragging is
    not a negotiation with the dynamics: the object goes where you put it. Ctrl+drag with
    the left mouse button translates, ctrl+drag with the right rotates (standard MuJoCo
    viewer perturbation bindings). Close the viewer window when you are done.

Phase 2, fit.  A raw mouse path is jittery and arbitrarily fast, and the controller's
    Reference wants C2 continuity. The recording is resampled, smoothed, and reduced to
    keyframes, then checked for dynamic feasibility.

Phase 3, use.  The result is a JSON trajectory that swarm_sim and tiled_view load with
    --trajectory, in place of a hand-coded reference.

The feasibility check is the part that matters and the part people skip. A dragged path
is not physically achievable in general: you can pull the object through the table, or
move it at 5 m/s, or demand a rotation no friction could deliver. If you replay an
infeasible demonstration the swarm will fail and you will not be able to tell whether
the controller is bad or the demonstration was impossible. So this refuses to export one
without telling you what it would require.

Usage:
    python record_trajectory.py --record --out my_traj.json
    python record_trajectory.py --check my_traj.json
    python swarm_sim.py --trajectory my_traj.json --out out/replay.mp4
    python tiled_view.py --grid 4 4 --trajectory my_traj.json --seconds 60
"""

from __future__ import annotations

import argparse
import json
import os
import time

import numpy as np
from scipy.spatial.transform import Rotation as Rot
from scipy.spatial.transform import Slerp

os.environ.setdefault("MUJOCO_GL", "osmesa")

import mujoco                                  # noqa: E402

import swarm_sim as S                          # noqa: E402
from objects import OBJECTS                    # noqa: E402


# --------------------------------------------------------------------------------------
# Phase 1: record
# --------------------------------------------------------------------------------------

def record_xml(obj):
    """The object alone, as a MOCAP body, on a table.

    Mocap because a mocap body is kinematically positioned: the viewer's perturbation
    moves it exactly, with no dynamics resisting. Dragging a dynamic body instead means
    fighting gravity and contact while you try to draw a path.
    """
    inner = obj["xml"]("rec", 0.0, 0.0)
    # strip the freejoint and make it mocap
    inner = inner.replace('<freejoint name="rec_objfree"/>', "")
    inner = inner.replace('<body name="rec_obj"', '<body name="rec_obj" mocap="true"')
    return f"""
<mujoco model="record">
  <compiler angle="radian" autolimits="true"/>
  <option timestep="{S.TIMESTEP}" gravity="0 0 0"/>
  <visual>
    <global offwidth="1280" offheight="720" azimuth="130" elevation="-20"/>
    <quality shadowsize="2048"/>
    <headlight ambient="0.4 0.4 0.42" diffuse="0.5 0.5 0.5"/>
  </visual>
  <asset>
    <texture name="sky" type="skybox" builtin="gradient" rgb1="0.16 0.18 0.24"
             rgb2="0.04 0.05 0.07" width="512" height="512"/>
    <texture name="tt" type="2d" builtin="checker" rgb1="0.30 0.31 0.34"
             rgb2="0.25 0.26 0.29" width="512" height="512"/>
    <material name="tablemat" texture="tt" texrepeat="9 9" specular="0.2"/>
    <material name="cubemat" rgba="0.09 0.09 0.10 1" specular="0.35" shininess="0.5"/>
{obj.get("assets", "")}
  </asset>
  <worldbody>
    <light name="key" pos="0.5 -0.5 1.2" dir="-0.4 0.4 -1" directional="true"
           diffuse="0.75 0.74 0.73" castshadow="true"/>
    <geom name="table" type="box" pos="0 0 -0.02" size="0.45 0.45 0.02"
          material="tablemat"/>
{inner}
  </worldbody>
</mujoco>
"""


def record(obj, rate=60.0):
    from mujoco import viewer as mj_viewer

    model = mujoco.MjModel.from_xml_string(record_xml(obj))
    data = mujoco.MjData(model)
    bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "rec_obj")
    mid = model.body_mocapid[bid]
    data.mocap_pos[mid] = [0.0, 0.0, obj["rest_z"]]
    data.mocap_quat[mid] = [1.0, 0.0, 0.0, 0.0]
    mujoco.mj_forward(model, data)

    print("\n  RECORDING")
    print("  ctrl + left-drag   move the object")
    print("  ctrl + right-drag  rotate it")
    print("  close the window when you are done\n")

    samples = []
    dt = 1.0 / rate
    t0 = time.time()
    last = 0.0
    with mj_viewer.launch_passive(model, data) as v:
        while v.is_running():
            mujoco.mj_forward(model, data)
            v.sync()
            now = time.time() - t0
            if now - last >= dt:
                last = now
                p = np.array(data.mocap_pos[mid], float)
                q = np.array(data.mocap_quat[mid], float)
                samples.append((now, p.copy(), q.copy()))
            time.sleep(0.002)

    print(f"  captured {len(samples)} samples over {samples[-1][0]:.1f}s"
          if samples else "  nothing captured")
    return samples


# --------------------------------------------------------------------------------------
# Phase 2: fit
# --------------------------------------------------------------------------------------

def fit(samples, n_keys=14, smooth=9):
    """Resample, smooth, and reduce to keyframes the Reference can interpolate."""
    if len(samples) < 8:
        raise SystemExit("too few samples to fit a trajectory")
    t = np.array([s[0] for s in samples])
    p = np.array([s[1] for s in samples])
    q = np.array([s[2] for s in samples])           # wxyz

    # moving average on position; the mouse path is noisy at the sample rate
    k = max(1, int(smooth) | 1)
    pad = k // 2
    pp = np.pad(p, ((pad, pad), (0, 0)), mode="edge")
    ker = np.ones(k) / k
    p_s = np.stack([np.convolve(pp[:, i], ker, mode="valid") for i in range(3)], axis=1)

    rots = Rot.from_quat(np.column_stack([q[:, 1], q[:, 2], q[:, 3], q[:, 0]]))
    sl = Slerp(t, rots)

    tk = np.linspace(t[0], t[-1], n_keys)
    keys = []
    for tt in tk:
        i = int(np.clip(np.searchsorted(t, tt) - 1, 0, len(t) - 2))
        a = (tt - t[i]) / max(t[i + 1] - t[i], 1e-9)
        pos = p_s[i] + a * (p_s[i + 1] - p_s[i])
        keys.append((float(tt - t[0]), [float(x) for x in pos],
                     [float(x) for x in sl(tt).as_quat()]))
    return keys


def feasibility(keys, obj):
    """What the demonstration would demand of the contacts, and whether that is sane."""
    from scipy.spatial.transform import Slerp as _S
    tt = np.array([k[0] for k in keys])
    pp = np.array([k[1] for k in keys])
    rr = Rot.from_quat([k[2] for k in keys])
    sl = _S(tt, rr)

    ts = np.linspace(tt[0], tt[-1], 200)
    ps = np.stack([np.interp(ts, tt, pp[:, i]) for i in range(3)], axis=1)
    dt = ts[1] - ts[0]
    vel = np.gradient(ps, dt, axis=0)
    acc = np.gradient(vel, dt, axis=0)
    rs = sl(ts)
    wv = np.array([(rs[i + 1] * rs[i].inv()).as_rotvec() / dt
                   for i in range(len(ts) - 1)])

    m = obj["mass"]
    f_req = m * np.linalg.norm(acc - np.array([0, 0, -S.GRAVITY]), axis=1)
    issues = []
    if ps[:, 2].min() < obj["rest_z"] - 2e-3:
        issues.append(f"goes {(obj['rest_z']-ps[:,2].min())*1000:.0f} mm below the table")
    if np.linalg.norm(vel, axis=1).max() > 1.5:
        issues.append(f"peak speed {np.linalg.norm(vel,axis=1).max():.2f} m/s is fast "
                      f"for a frictional grasp")
    if np.linalg.norm(wv, axis=1).max() > 8.0:
        issues.append(f"peak angular rate {np.linalg.norm(wv,axis=1).max():.1f} rad/s")
    return {
        "duration_s": float(ts[-1]),
        "peak_speed_ms": float(np.linalg.norm(vel, axis=1).max()),
        "peak_accel_ms2": float(np.linalg.norm(acc, axis=1).max()),
        "peak_ang_rate_rads": float(np.linalg.norm(wv, axis=1).max()),
        "peak_contact_force_N": float(f_req.max()),
        "min_height_mm": float(ps[:, 2].min() * 1000),
        "issues": issues,
    }


def report(info):
    print(f"\n  duration              {info['duration_s']:.2f} s")
    print(f"  peak speed            {info['peak_speed_ms']:.3f} m/s")
    print(f"  peak acceleration     {info['peak_accel_ms2']:.2f} m/s^2")
    print(f"  peak angular rate     {info['peak_ang_rate_rads']:.2f} rad/s")
    print(f"  peak contact force    {info['peak_contact_force_N']:.2f} N")
    print(f"  minimum height        {info['min_height_mm']:.1f} mm")
    if info["issues"]:
        print("\n  NOT CLEANLY FEASIBLE:")
        for s in info["issues"]:
            print(f"    - {s}")
        print("\n  Replaying this will probably fail, and you will not be able to tell\n"
              "  whether the controller or the demonstration is at fault. Re-record it\n"
              "  more slowly, or accept that the failure is expected.")
    else:
        print("\n  looks feasible")


# --------------------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--record", action="store_true")
    ap.add_argument("--check", default=None, help="report on an existing trajectory")
    ap.add_argument("--object", default="cube", choices=sorted(OBJECTS))
    ap.add_argument("--mesh", default=None)
    ap.add_argument("--mesh-scale", type=float, default=1.0)
    ap.add_argument("--keys", type=int, default=14)
    ap.add_argument("--smooth", type=int, default=9)
    ap.add_argument("--out", default="recorded_trajectory.json")
    args = ap.parse_args()

    obj = OBJECTS[args.object](mesh=args.mesh, scale=args.mesh_scale)

    if args.check:
        spec = json.load(open(args.check))
        report(feasibility([(k["t"], k["pos"], k["quat"]) for k in spec["keys"]], obj))
        return 0

    if not args.record:
        ap.error("pass --record to capture, or --check FILE to inspect one")

    samples = record(obj)
    if not samples:
        return 1
    keys = fit(samples, args.keys, args.smooth)
    info = feasibility(keys, obj)
    report(info)

    json.dump({"source": "record_trajectory.py", "object": obj["name"],
               "feasibility": info,
               "keys": [{"t": t, "pos": p, "quat": q} for t, p, q in keys]},
              open(args.out, "w"), indent=2)
    print(f"\n  wrote {args.out}")
    print(f"  replay it:  python swarm_sim.py --trajectory {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
