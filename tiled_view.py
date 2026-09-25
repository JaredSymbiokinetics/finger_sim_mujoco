"""
A grid of simulations running at once, in plain MuJoCo.

MuJoCo has no tiled renderer the way Isaac Lab does, but it does not need one for this:
you put N copies of the scene into ONE model, laid out on a grid, and then a single
physics step advances all of them and a single render pass draws all of them. The copies
are independent (no shared bodies, no shared contacts), the solver treats them as
separate contact islands, and each copy can run its own contact placement because
placement lives in the controller, not the XML.

What this costs: every copy adds bodies and contacts to one solver, so wall time scales
roughly linearly and you will not stay real time much past a couple of dozen copies of
this scene. It is a visualisation tool, not a throughput tool. For throughput you want
independent processes (learn_placement.py already does that) or MJX. For hundreds of
tiles rendered live, Isaac Lab's tiled rendering is the right tool and this is not.

Usage:
    python tiled_view.py --grid 3 3 --from-cem best_placement_n4.json --out out/tiled.mp4
    python tiled_view.py --grid 4 4 --random 16 --fingers 4 --out out/tiled.mp4
    python tiled_view.py --grid 3 3 --random 9 --viewer      # interactive, no file
"""

from __future__ import annotations

import argparse
import json
import os

import numpy as np

os.environ.setdefault("MUJOCO_GL", "osmesa")

import mujoco                                          # noqa: E402
from scipy.spatial.transform import Rotation as Rot    # noqa: E402

import swarm_sim as S                                  # noqa: E402

PITCH = 0.50            # m between tile centres. Fingers start ~0.24 m
                        # out, so tiles closer than ~0.48 m would let
                        # neighbouring swarms collide during fly-in.


# --------------------------------------------------------------------------------------

def tiled_xml(cells, n_fingers):
    """One MJCF containing len(cells) independent copies of the scene on a grid."""
    a = S.CUBE_HALF
    sticker = a * 0.78
    inset = a + 0.0009
    faces = [
        ("px", f"{inset} 0 0", f"0.001 {sticker} {sticker}", "0.85 0.16 0.14 1"),
        ("nx", f"-{inset} 0 0", f"0.001 {sticker} {sticker}", "0.95 0.48 0.09 1"),
        ("py", f"0 {inset} 0", f"{sticker} 0.001 {sticker}", "0.10 0.38 0.75 1"),
        ("ny", f"0 -{inset} 0", f"{sticker} 0.001 {sticker}", "0.11 0.55 0.25 1"),
        ("pz", f"0 0 {inset}", f"{sticker} {sticker} 0.001", "0.96 0.96 0.94 1"),
        ("nz", f"0 0 -{inset}", f"{sticker} {sticker} 0.001", "0.97 0.85 0.10 1"),
    ]

    body = []
    for k, (cx, cy) in enumerate(cells):
        stickers = "\n".join(
            f'        <geom name="s{k}_{nm}" type="box" pos="{p}" size="{sz}" '
            f'rgba="{c}" contype="0" conaffinity="0" group="1" mass="0"/>'
            for nm, p, sz, c in faces)
        fingers = []
        for i in range(n_fingers):
            col = S.BASE_COLORS[i % len(S.BASE_COLORS)]
            fingers.append(f"""
    <body name="e{k}_finger{i}" pos="{cx + 0.25} {cy} 0.12" gravcomp="1">
      <freejoint name="e{k}_fj{i}"/>
      <inertial pos="0 0 0" mass="{S.FINGER_MASS}"
                diaginertia="{S.FINGER_INERTIA} {S.FINGER_INERTIA} {S.FINGER_INERTIA}"/>
      <geom name="e{k}_tip{i}" type="sphere" size="{S.TIP_RADIUS}" rgba="{col}"
            friction="{S.MU_TIP} 0.02 0.002"
            solref="-{S.TIP_STIFFNESS} -{S.TIP_DAMPING}" solimp="0.92 0.97 0.001"
            condim="4" priority="2" mass="0"/>
      <geom name="e{k}_shaft{i}" type="capsule" fromto="0.004 0 0 {S.SHAFT_LEN} 0 0"
            size="0.0052" rgba="0.30 0.32 0.36 1" friction="0.6 0.01 0.001"
            condim="3" mass="0"/>
    </body>""")
        body.append(f"""
    <geom name="e{k}_table" type="box" pos="{cx} {cy} -0.02" size="0.20 0.20 0.02"
          material="tablemat" friction="{S.MU_TABLE} 0.01 0.001" condim="4"/>
    <body name="e{k}_cube" pos="{cx} {cy} {a}">
      <freejoint name="e{k}_cubefree"/>
      <geom name="e{k}_cubegeom" type="box" size="{a} {a} {a}" material="cubemat"
            mass="{S.CUBE_MASS}" friction="{S.MU_TABLE} 0.01 0.001" condim="4"
            solref="-{S.TIP_STIFFNESS} -{S.TIP_DAMPING}" solimp="0.92 0.97 0.001"/>
{stickers}
    </body>{''.join(fingers)}""")

    return f"""
<mujoco model="tiled">
  <compiler angle="radian" autolimits="true"/>
  <option timestep="{S.TIMESTEP}" gravity="0 0 -{S.GRAVITY}" integrator="implicitfast"
          cone="elliptic" impratio="10" noslip_iterations="3"/>
  <visual>
    <global offwidth="1920" offheight="1080"/>
    <quality shadowsize="2048" offsamples="2"/>
    <map znear="0.02" zfar="60"/>
    <headlight ambient="0.42 0.42 0.44" diffuse="0.5 0.5 0.5" specular="0.2 0.2 0.2"/>
  </visual>
  <asset>
    <texture name="sky" type="skybox" builtin="gradient" rgb1="0.16 0.18 0.24"
             rgb2="0.04 0.05 0.07" width="512" height="512"/>
    <texture name="tt" type="2d" builtin="checker" rgb1="0.30 0.31 0.34"
             rgb2="0.25 0.26 0.29" width="512" height="512"/>
    <material name="tablemat" texture="tt" texrepeat="5 5" specular="0.2"
              shininess="0.3"/>
    <material name="cubemat" rgba="0.09 0.09 0.10 1" specular="0.35" shininess="0.5"/>
  </asset>
  <worldbody>
    <light name="key" pos="1.2 -1.2 3.0" dir="-0.3 0.3 -1" directional="true"
           diffuse="0.75 0.74 0.73" castshadow="true"/>
    <light name="fill" pos="-1.4 1.2 2.4" dir="0.4 -0.35 -1" directional="true"
           diffuse="0.25 0.26 0.29" castshadow="false"/>
{''.join(body)}
  </worldbody>
</mujoco>
"""


class TiledSim:
    """N independent copies of the scene in one model, each with its own placement."""

    def __init__(self, param_sets, task="probe"):
        self.n_env = len(param_sets)
        self.n_fing = len(param_sets[0]) // 3
        cols = int(np.ceil(np.sqrt(self.n_env)))
        rows = int(np.ceil(self.n_env / cols))
        self.cells = [((c - (cols - 1) / 2) * PITCH, ((rows - 1) / 2 - r) * PITCH)
                      for r in range(rows) for c in range(cols)][:self.n_env]
        self.cols, self.rows = cols, rows

        S.set_task(task)
        self.ref, self.segments = S.TASKS[task][0]()
        self.contacts = [[S.surface_point(v) for v in np.asarray(p).reshape(-1, 3)]
                         for p in param_sets]

        self.model = mujoco.MjModel.from_xml_string(tiled_xml(self.cells, self.n_fing))
        self.data = mujoco.MjData(self.model)

        nm = lambda s: mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, s)
        jt = lambda s: mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, s)
        self.cube_b = [nm(f"e{k}_cube") for k in range(self.n_env)]
        self.cube_q = [self.model.jnt_qposadr[jt(f"e{k}_cubefree")]
                       for k in range(self.n_env)]
        self.cube_v = [self.model.jnt_dofadr[jt(f"e{k}_cubefree")]
                       for k in range(self.n_env)]
        self.fing_b = [[nm(f"e{k}_finger{i}") for i in range(self.n_fing)]
                       for k in range(self.n_env)]
        self.fing_q = [[self.model.jnt_qposadr[jt(f"e{k}_fj{i}")]
                        for i in range(self.n_fing)] for k in range(self.n_env)]
        self.fing_v = [[self.model.jnt_dofadr[jt(f"e{k}_fj{i}")]
                        for i in range(self.n_fing)] for k in range(self.n_env)]
        self.reset()

    def origin(self, k):
        return np.array([self.cells[k][0], self.cells[k][1], 0.0])

    def reset(self):
        mujoco.mj_resetData(self.model, self.data)
        for k in range(self.n_env):
            o = self.origin(k)
            q = self.cube_q[k]
            self.data.qpos[q:q + 3] = o + [0, 0, S.REST_Z]
            self.data.qpos[q + 3:q + 7] = [1, 0, 0, 0]
            home = S.make_home(self.contacts[k])
            for i in range(self.n_fing):
                fq = self.fing_q[k][i]
                self.data.qpos[fq:fq + 3] = o + home[i]
                self.data.qpos[fq + 3:fq + 7] = S.finger_quat(self.contacts[k][i][1])
        mujoco.mj_forward(self.model, self.data)
        self._prev = [S.make_home(self.contacts[k]) + self.origin(k)
                      for k in range(self.n_env)]
        self._int_p = [np.zeros(3) for _ in range(self.n_env)]
        self._int_r = [np.zeros(3) for _ in range(self.n_env)]
        self.alive = [True] * self.n_env

    def pose(self, k):
        p = self.data.xpos[self.cube_b[k]].copy() - self.origin(k)
        w = self.data.xquat[self.cube_b[k]].copy()
        return p, Rot.from_quat([w[1], w[2], w[3], w[0]])

    def step(self):
        """One shared physics step; every tile controlled independently."""
        t = self.data.time
        p_d, R_d = self.ref(min(t, self.ref.duration))
        self.data.xfrc_applied[:] = 0.0

        for k in range(self.n_env):
            p, R = self.pose(k)
            e_p, e_r = p_d - p, (R_d * R.inv()).as_rotvec()
            vadr = self.cube_v[k]
            v = self.data.qvel[vadr:vadr + 3]
            w = R.apply(self.data.qvel[vadr + 3:vadr + 6])

            self._int_p[k] = np.clip(self._int_p[k] + e_p * S.TIMESTEP,
                                     -S.IMAX_POS, S.IMAX_POS)
            self._int_r[k] = np.clip(self._int_r[k] + e_r * S.TIMESTEP,
                                     -S.IMAX_ROT, S.IMAX_ROT)
            lin = np.clip(e_p + S.KI_POS * self._int_p[k] - S.KD_POS * v,
                          -S.MAX_LAG, S.MAX_LAG)
            rv = e_r + S.KI_ROT * self._int_r[k] - S.KD_ROT * w
            nn = np.linalg.norm(rv)
            lag_rot = max(S.MAX_LAG_ROT, 1.25)
            if nn > lag_rot:
                rv = rv * (lag_rot / nn)
            p_c, R_c = p + lin, Rot.from_rotvec(rv) * R

            delta = S.squeeze_depth(t, self.ref.duration)
            o = self.origin(k)
            for i, (off, nrm) in enumerate(self.contacts[k]):
                n_w = R_c.apply(nrm)
                tgt = p_c + R_c.apply(off) + n_w * (S.TIP_RADIUS - delta)
                if t < S.GRASP_START:
                    s = S.smoothstep(t / S.GRASP_START)
                    home = S.make_home(self.contacts[k])[i]
                    tgt = home + s * (tgt - home)
                tgt = tgt + o
                step_max = S.SLEW * S.TIMESTEP
                d = tgt - self._prev[k][i]
                nd = np.linalg.norm(d)
                if nd > step_max:
                    tgt = self._prev[k][i] + d * (step_max / nd)
                self._prev[k][i] = tgt

                b = self.fing_b[k][i]
                fv = self.fing_v[k][i]
                fp = self.data.xpos[b]
                fw = self.data.xquat[b]
                R_now = Rot.from_quat([fw[1], fw[2], fw[3], fw[0]])
                wq = S.finger_quat(n_w)
                R_des = Rot.from_quat([wq[1], wq[2], wq[3], wq[0]])
                f = S.MOUNT_KP * (tgt - fp) - S.MOUNT_KD * self.data.qvel[fv:fv + 3]
                nf = np.linalg.norm(f)
                if nf > S.MOUNT_FMAX:
                    f = f * (S.MOUNT_FMAX / nf)
                w_world = R_now.apply(self.data.qvel[fv + 3:fv + 6])
                tau = (S.MOUNT_KR * (R_des * R_now.inv()).as_rotvec()
                       - S.MOUNT_KW * w_world)
                nt = np.linalg.norm(tau)
                if nt > S.MOUNT_TMAX:
                    tau = tau * (S.MOUNT_TMAX / nt)
                self.data.xfrc_applied[b, :3] = f
                self.data.xfrc_applied[b, 3:] = tau

        mujoco.mj_step(self.model, self.data)
        for k in range(self.n_env):
            p, _ = self.pose(k)
            if np.linalg.norm(p - p_d) > 0.05:
                self.alive[k] = False

    def errors(self):
        p_d, R_d = self.ref(min(self.data.time, self.ref.duration))
        out = []
        for k in range(self.n_env):
            p, R = self.pose(k)
            out.append((np.linalg.norm(p_d - p) * 1000.0,
                        np.rad2deg(np.linalg.norm((R_d * R.inv()).as_rotvec()))))
        return out


# --------------------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--grid", type=int, nargs=2, default=[3, 3])
    ap.add_argument("--fingers", type=int, default=4)
    ap.add_argument("--random", type=int, default=None,
                    help="sample N random placements around the heuristic")
    ap.add_argument("--from-cem", default=None,
                    help="best_placement JSON; tiles are perturbations around it")
    ap.add_argument("--sigma", type=float, default=0.35)
    ap.add_argument("--task", default="probe")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="out/tiled.mp4")
    ap.add_argument("--viewer", action="store_true", help="interactive instead of mp4")
    ap.add_argument("--width", type=int, default=1280)
    ap.add_argument("--height", type=int, default=720)
    ap.add_argument("--fps", type=int, default=30)
    args = ap.parse_args()

    n_env = args.grid[0] * args.grid[1]
    rng = np.random.default_rng(args.seed)

    S.set_task(args.task)
    if args.from_cem:
        spec = json.load(open(args.from_cem))
        base = np.array(spec["params"], float)
        n_f = spec["n_fingers"]
    else:
        n_f = args.fingers
        S.set_finger_count(n_f)
        base = np.array([nrm for _, nrm in S.CONTACTS], float).reshape(-1)

    sets = [base] + [rng.normal(base, args.sigma) for _ in range(n_env - 1)]
    print(f"tiling {n_env} environments, {n_f} fingers each "
          f"({n_env * (n_f + 1)} bodies), task={args.task}")
    print("tile 0 is the reference placement; the rest are perturbations of it")

    sim = TiledSim(sets, args.task)
    print(f"model: nq={sim.model.nq} nv={sim.model.nv} ngeom={sim.model.ngeom}")

    if args.viewer:
        # NB: `import mujoco.viewer` here would rebind `mujoco` as a function-local name
        # for the whole function body, shadowing the module-level import.
        from mujoco import viewer as mj_viewer
        with mj_viewer.launch_passive(sim.model, sim.data) as v:
            while v.is_running() and sim.data.time < sim.ref.duration:
                sim.step()
                v.sync()
        return 0

    import imageio.v2 as imageio
    import pathlib
    pathlib.Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    r = mujoco.Renderer(sim.model, args.height, args.width)
    cam = mujoco.MjvCamera()
    cam.type = mujoco.mjtCamera.mjCAMERA_FREE
    cam.distance = PITCH * max(sim.cols, sim.rows) * 1.18
    cam.elevation = -38.0
    cam.azimuth = 90.0
    cam.lookat[:] = [0.0, 0.0, 0.07]

    wr = imageio.get_writer(args.out, fps=args.fps, codec="libx264", quality=8,
                            macro_block_size=1, ffmpeg_params=["-pix_fmt", "yuv420p"])
    spf = max(1, int(round((1.0 / args.fps) / S.TIMESTEP)))
    nframe = int(sim.ref.duration * args.fps)
    for n in range(nframe):
        for _ in range(spf):
            sim.step()
        r.update_scene(sim.data, camera=cam)
        wr.append_data(r.render())
        if n % 30 == 0:
            e = sim.errors()
            print(f"  frame {n:4d}/{nframe}  alive {sum(sim.alive)}/{n_env}  "
                  f"best {min(x[0] for x in e):.2f} mm", flush=True)
    wr.close()
    r.close()

    e = sim.errors()
    print(f"\nwrote {args.out}")
    print("\n  tile   final pos err   final rot err   held")
    for k in range(n_env):
        print(f"  {k:4d}   {e[k][0]:10.2f} mm {e[k][1]:12.2f} deg   "
              f"{'yes' if sim.alive[k] else 'no'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
