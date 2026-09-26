"""
A grid of simulations running a live search, in plain MuJoCo.

MuJoCo has no tiled renderer like Isaac Lab's, and does not need one for this: put N
copies of the scene into ONE model on a grid, and a single physics step advances all of
them while a single render pass draws all of them. The copies are independent (separate
bodies, separate contact islands) and each runs its own contact placement, because
placement lives in the controller rather than in the XML.

Each tile runs its own episode on its own clock. When a tile finishes, or drops the
object, it is scored, reset on the spot, and handed the next candidate from an
asynchronous cross-entropy search. So the grid keeps working indefinitely and you watch
the population improve, rather than watching one fixed batch play out once.

Every tile runs the SAME controller as swarm_sim.Sim, through the shared functions in
that module. An earlier version of this file carried its own transcription of the
control law and had already drifted from it. Two copies of a controller that are meant
to be identical will diverge, and then a result measured in one is not a result about
the other.

Cost scales roughly linearly in tiles, so this tops out in the low tens on a laptop and
maybe a hundred on a workstation. It is a visualisation tool. For search throughput use
independent processes (learn_placement.py) or MJX; for hundreds of tiles rendered live,
Isaac Lab is the right tool and this is not.

Usage:
    python tiled_view.py --grid 6 6 --seconds 90 --out out/search.mp4
    python tiled_view.py --grid 4 4 --from-cem best_placement_n4.json --viewer
    python tiled_view.py --grid 5 5 --object cylinder --fingers 5 --seconds 60
"""

from __future__ import annotations

import argparse
import json
import os
import pathlib
import time

import numpy as np

os.environ.setdefault("MUJOCO_GL", "osmesa")

import mujoco                                          # noqa: E402
from scipy.spatial.transform import Rotation as Rot    # noqa: E402

import swarm_sim as S                                  # noqa: E402
from objects import OBJECTS, object_xml                # noqa: E402

PITCH = 0.50            # m between tile centres. Fingers start ~0.24 m out, so tiles
                        # closer than ~0.48 m would let neighbouring swarms collide.
FAIL_MM = 25.0          # position error that counts as having dropped the object


# --------------------------------------------------------------------------------------
# Asynchronous cross-entropy search
# --------------------------------------------------------------------------------------

class CEMDriver:
    """Hands out candidates and refits once enough results are back.

    Asynchronous on purpose: tiles finish at different times (a tile that drops the
    object early frees up sooner than one that completes), so insisting on lockstep
    generations would leave most of the grid idle most of the time. Instead results go
    into a rolling buffer and the distribution refits whenever `population` of them have
    arrived.
    """

    def __init__(self, mean, sigma0=0.45, population=24, elite_frac=0.25, seed=0):
        self.mean = np.asarray(mean, float).copy()
        self.sigma = np.full(self.mean.shape, float(sigma0))
        self.population = population
        self.n_elite = max(2, int(round(population * elite_frac)))
        self.rng = np.random.default_rng(seed)
        self.buffer = []
        self.generation = 0
        self.best_cost = float("inf")
        self.best_params = self.mean.copy()
        self.history = []
        self.n_eval = 0
        self.first = True

    def next_candidate(self):
        if self.first:                      # always evaluate the incumbent once
            self.first = False
            return self.mean.copy()
        return self.rng.normal(self.mean, self.sigma)

    def report(self, params, cost):
        self.n_eval += 1
        self.buffer.append((float(cost), np.asarray(params, float)))
        if cost < self.best_cost:
            self.best_cost = float(cost)
            self.best_params = np.asarray(params, float).copy()
        if len(self.buffer) >= self.population:
            self.buffer.sort(key=lambda x: x[0])
            elite = np.array([p for _, p in self.buffer[:self.n_elite]])
            self.mean = elite.mean(axis=0)
            self.sigma = np.maximum(elite.std(axis=0), 0.03)
            costs = [c for c, _ in self.buffer]
            self.generation += 1
            self.history.append({
                "gen": self.generation, "evals": self.n_eval,
                "best": min(costs), "median": float(np.median(costs)),
                "elite_mean": float(np.mean(costs[:self.n_elite])),
                "sigma": float(self.sigma.mean()),
                "best_ever": self.best_cost})
            self.buffer = []
            return True
        return False


# --------------------------------------------------------------------------------------
# Tiled model
# --------------------------------------------------------------------------------------

def tiled_xml(cells, n_fingers, obj):
    body = []
    for k, (cx, cy) in enumerate(cells):
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
{object_xml(obj, f"e{k}", cx, cy)}{''.join(fingers)}""")

    return f"""
<mujoco model="tiled">
  <compiler angle="radian" autolimits="true"/>
  <option timestep="{S.TIMESTEP}" gravity="0 0 -{S.GRAVITY}" integrator="implicitfast"
          cone="elliptic" impratio="10" noslip_iterations="3"/>
  <visual>
    <global offwidth="1920" offheight="1080"/>
    <quality shadowsize="1024" offsamples="2"/>
    <map znear="0.02" zfar="80"/>
    <headlight ambient="0.45 0.45 0.47" diffuse="0.5 0.5 0.5" specular="0.15 0.15 0.15"/>
  </visual>
  <asset>
    <texture name="sky" type="skybox" builtin="gradient" rgb1="0.16 0.18 0.24"
             rgb2="0.04 0.05 0.07" width="512" height="512"/>
    <texture name="tt" type="2d" builtin="checker" rgb1="0.30 0.31 0.34"
             rgb2="0.25 0.26 0.29" width="256" height="256"/>
    <material name="tablemat" texture="tt" texrepeat="4 4" specular="0.15"/>
    <material name="cubemat" rgba="0.09 0.09 0.10 1" specular="0.35" shininess="0.5"/>
{obj.get("assets", "")}
  </asset>
  <worldbody>
    <light name="key" pos="1.5 -1.5 4.0" dir="-0.3 0.3 -1" directional="true"
           diffuse="0.75 0.74 0.73" castshadow="true"/>
    <light name="fill" pos="-1.8 1.5 3.0" dir="0.4 -0.35 -1" directional="true"
           diffuse="0.25 0.26 0.29" castshadow="false"/>
{''.join(body)}
  </worldbody>
</mujoco>
"""


class TiledSearch:
    def __init__(self, n_env, n_fingers, obj, task="probe", driver=None):
        self.n_env = n_env
        self.n_fing = n_fingers
        self.obj = obj
        cols = int(np.ceil(np.sqrt(n_env)))
        rows = int(np.ceil(n_env / cols))
        self.cols, self.rows = cols, rows
        self.cells = [((c - (cols - 1) / 2) * PITCH, ((rows - 1) / 2 - r) * PITCH)
                      for r in range(rows) for c in range(cols)][:n_env]

        S.set_task(task)
        self.task = task
        self.ref, self.segments = S.TASKS[task][0]()
        self.rest_z = obj["rest_z"]
        self.driver = driver

        self.model = mujoco.MjModel.from_xml_string(
            tiled_xml(self.cells, n_fingers, obj))
        self.data = mujoco.MjData(self.model)

        nm = lambda s: mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, s)
        jt = lambda s: mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, s)
        gm = lambda s: mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, s)
        self.obj_b = [nm(f"e{k}_obj") for k in range(n_env)]
        self.obj_q = [self.model.jnt_qposadr[jt(f"e{k}_objfree")] for k in range(n_env)]
        self.obj_v = [self.model.jnt_dofadr[jt(f"e{k}_objfree")] for k in range(n_env)]
        self.tab_g = [gm(f"e{k}_table") for k in range(n_env)]
        self.fb = [[nm(f"e{k}_finger{i}") for i in range(n_fing_i)]
                   for k, n_fing_i in ((k, n_fingers) for k in range(n_env))]
        self.fq = [[self.model.jnt_qposadr[jt(f"e{k}_fj{i}")] for i in range(n_fingers)]
                   for k in range(n_env)]
        self.fv = [[self.model.jnt_dofadr[jt(f"e{k}_fj{i}")] for i in range(n_fingers)]
                   for k in range(n_env)]

        self.params = [None] * n_env
        self.contacts = [None] * n_env
        self.t = np.zeros(n_env)
        self.int_p = [np.zeros(3) for _ in range(n_env)]
        self.int_r = [np.zeros(3) for _ in range(n_env)]
        self.prev = [None] * n_env
        self.err = [[] for _ in range(n_env)]
        self.done_count = 0
        self.results = []
        for k in range(n_env):
            self.load(k, driver.next_candidate() if driver else None)
        mujoco.mj_forward(self.model, self.data)   # xpos/xquat are zero until this runs

    # -- per-tile lifecycle ---------------------------------------------------------

    def origin(self, k):
        return np.array([self.cells[k][0], self.cells[k][1], 0.0])

    def load(self, k, params, outcome=None):
        """Reset ONE tile and give it a new candidate. Other tiles keep running."""
        if params is None:
            params = np.array([n for _, n in self.obj["default_contacts"](self.n_fing)]
                              ).reshape(-1)
        self.params[k] = np.asarray(params, float)
        self.contacts[k] = [self.obj["surface_point"](v)
                            for v in self.params[k].reshape(-1, 3)]
        o = self.origin(k)
        q = self.obj_q[k]
        self.data.qpos[q:q + 3] = o + [0, 0, self.rest_z]
        self.data.qpos[q + 3:q + 7] = [1, 0, 0, 0]
        self.data.qvel[self.obj_v[k]:self.obj_v[k] + 6] = 0.0
        home = S.make_home(self.contacts[k])
        for i in range(self.n_fing):
            fq = self.fq[k][i]
            self.data.qpos[fq:fq + 3] = o + home[i]
            self.data.qpos[fq + 3:fq + 7] = S.finger_quat(self.contacts[k][i][1])
            self.data.qvel[self.fv[k][i]:self.fv[k][i] + 6] = 0.0
        self.t[k] = 0.0
        self.int_p[k] = np.zeros(3)
        self.int_r[k] = np.zeros(3)
        self.prev[k] = home + o
        self.err[k] = []
        if outcome is not None:
            c = ([0.13, 0.30, 0.16, 1] if outcome == "held"
                 else [0.34, 0.11, 0.11, 1])
            self.model.geom_rgba[self.tab_g[k]] = c

    def pose(self, k):
        p = self.data.xpos[self.obj_b[k]].copy() - self.origin(k)
        w = self.data.xquat[self.obj_b[k]].copy()
        return p, Rot.from_quat([w[1], w[2], w[3], w[0]])

    def touching_table(self, k):
        og = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, f"e{k}_objgeom")
        for ci in range(self.data.ncon):
            c = self.data.contact[ci]
            if og in (c.geom1, c.geom2) and self.tab_g[k] in (c.geom1, c.geom2):
                return True
        return False

    def score(self, k, lost):
        if lost or not self.err[k]:
            frac = self.t[k] / self.ref.duration
            return 1e4 * (2.0 - frac), "lost"
        v = np.array(self.err[k])
        rms_p = float(np.sqrt((v[:, 0] ** 2).mean()))
        rms_r = float(np.sqrt((v[:, 1] ** 2).mean()))
        return rms_p + 2.0 * rms_r + 0.3 * float(v[:, 0].max()), "held"

    # -- physics --------------------------------------------------------------------

    def step(self):
        self.data.xfrc_applied[:] = 0.0
        for k in range(self.n_env):
            t = self.t[k]
            p, R = self.pose(k)
            vadr = self.obj_v[k]
            v = self.data.qvel[vadr:vadr + 3].copy()
            w = R.apply(self.data.qvel[vadr + 3:vadr + 6])
            p_d, R_d = self.ref(min(t, self.ref.duration))

            if t > S.GRASP_START:
                h = 2e-3
                p1, r1 = self.ref(min(t + h, self.ref.duration))
                p0, r0 = self.ref(max(t - h, 0.0))
                dt = min(t + h, self.ref.duration) - max(t - h, 0.0)
                v_d = (p1 - p0) / dt
                w_d = (r1 * r0.inv()).as_rotvec() / dt
                p_c, R_c, self.int_p[k], self.int_r[k] = S.solve_commanded_pose(
                    p, R, p_d, R_d, v, w, v_d, w_d, self.int_p[k], self.int_r[k],
                    S.MAX_LAG_ROT, integrate_pos=not self.touching_table(k))
            else:
                p_c, R_c = p, R

            deltas = [S.squeeze_depth(t, self.ref.duration)] * self.n_fing
            tgt, quats = S.targets_from_pose(p_c, R_c, self.contacts[k], deltas)
            if t < S.GRASP_START:
                s = S.smoothstep(t / S.GRASP_START)
                home = S.make_home(self.contacts[k])
                tgt = home + s * (tgt - home)
            tgt = S.slew_limit(tgt + self.origin(k), self.prev[k])
            self.prev[k] = tgt

            for i in range(self.n_fing):
                b = self.fb[k][i]
                fv = self.fv[k][i]
                wq = self.data.xquat[b]
                R_f = Rot.from_quat([wq[1], wq[2], wq[3], wq[0]])
                f, tau = S.solve_mount_wrench(
                    tgt[i], quats[i], self.data.xpos[b], R_f,
                    self.data.qvel[fv:fv + 3],
                    R_f.apply(self.data.qvel[fv + 3:fv + 6]))
                self.data.xfrc_applied[b, :3] = f
                self.data.xfrc_applied[b, 3:] = tau

        mujoco.mj_step(self.model, self.data)
        self.t += S.TIMESTEP

        # Episode bookkeeping, per tile
        reloaded = False
        for k in range(self.n_env):
            p, R = self.pose(k)
            p_d, R_d = self.ref(min(self.t[k], self.ref.duration))
            mm = float(np.linalg.norm(p_d - p) * 1000.0)
            deg = float(np.rad2deg(np.linalg.norm((R_d * R.inv()).as_rotvec())))
            if self.t[k] > S.GRASP_END:
                self.err[k].append((mm, deg))
            bad = (mm > FAIL_MM) or not np.all(
                np.isfinite(self.data.qpos[self.obj_q[k]:self.obj_q[k] + 7]))
            if bad or self.t[k] >= self.ref.duration:
                cost, outcome = self.score(k, bad)
                self.results.append(cost)
                self.done_count += 1
                if self.driver:
                    self.driver.report(self.params[k], cost)
                    self.load(k, self.driver.next_candidate(), outcome)
                else:
                    self.load(k, self.params[k], outcome)
                reloaded = True
        if reloaded:
            # Refresh derived quantities so the next control step sees the reset pose
            # rather than the pre-reset one.
            mujoco.mj_forward(self.model, self.data)


# --------------------------------------------------------------------------------------

def hud(frame, sim, driver, wall):
    try:
        from PIL import Image, ImageDraw, ImageFont
    except ImportError:
        return frame
    img = Image.fromarray(frame)
    d = ImageDraw.Draw(img, "RGBA")
    W, H = img.size

    def font(sz):
        for p in ("/usr/share/fonts/truetype/dejavu/DejaVuSansMono-Bold.ttf",
                  "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"):
            if os.path.exists(p):
                return ImageFont.truetype(p, sz)
        return ImageFont.load_default()

    fb, fm, fs = font(int(H * 0.030)), font(int(H * 0.023)), font(int(H * 0.018))
    pad = int(H * 0.025)
    d.rounded_rectangle([pad, pad, pad + int(W * 0.30), pad + int(H * 0.215)],
                        radius=10, fill=(12, 14, 18, 210))
    x, y = pad + int(W * 0.015), pad + int(H * 0.018)
    d.text((x, y), "PLACEMENT SEARCH", font=fb, fill=(235, 238, 245, 255))
    y += int(H * 0.040)
    d.text((x, y), f"{sim.n_env} tiles, {sim.n_fing} fingers, {sim.obj['name']}",
           font=fs, fill=(150, 158, 175, 255))
    y += int(H * 0.034)
    d.text((x, y), f"generation {driver.generation}", font=fm,
           fill=(200, 206, 218, 255))
    y += int(H * 0.030)
    d.text((x, y), f"episodes    {driver.n_eval}", font=fm, fill=(200, 206, 218, 255))
    y += int(H * 0.030)
    bc = driver.best_cost
    d.text((x, y), f"best cost   {bc:7.2f}" if bc < 1e4 else "best cost      none",
           font=fm, fill=(120, 220, 150, 255) if bc < 1e4 else (245, 190, 90, 255))
    y += int(H * 0.030)
    d.text((x, y), f"spread      {driver.sigma.mean():.3f}", font=fm,
           fill=(200, 206, 218, 255))

    lab = "green table = held    red = dropped"
    tw = d.textlength(lab, font=fs)
    d.rounded_rectangle([(W - tw) / 2 - 12, H - pad - int(H * 0.042),
                         (W + tw) / 2 + 12, H - pad], radius=8,
                        fill=(12, 14, 18, 200))
    d.text(((W - tw) / 2, H - pad - int(H * 0.034)), lab, font=fs,
           fill=(200, 206, 218, 255))
    return np.array(img.convert("RGB"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--grid", type=int, nargs=2, default=[4, 4])
    ap.add_argument("--fingers", type=int, default=4)
    ap.add_argument("--object", default="cube", choices=sorted(OBJECTS))
    ap.add_argument("--mesh", default=None, help="STL/OBJ path; sets --object mesh")
    ap.add_argument("--mesh-scale", type=float, default=1.0)
    ap.add_argument("--task", default="probe")
    ap.add_argument("--seconds", type=float, default=40.0, help="sim seconds to record")
    ap.add_argument("--population", type=int, default=24)
    ap.add_argument("--sigma", type=float, default=0.45)
    ap.add_argument("--from-cem", default=None)
    ap.add_argument("--trajectory", default=None,
                    help="JSON from record_trajectory.py; searched against instead of "
                         "the built-in task")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="out/tiled_search.mp4")
    ap.add_argument("--viewer", action="store_true")
    ap.add_argument("--width", type=int, default=1280)
    ap.add_argument("--height", type=int, default=720)
    ap.add_argument("--fps", type=int, default=30)
    args = ap.parse_args()

    obj = OBJECTS[args.object](mesh=args.mesh, scale=args.mesh_scale)
    n_env = args.grid[0] * args.grid[1]
    S.set_task(args.task)
    if args.trajectory:
        S.TASKS[args.task] = (lambda p=args.trajectory: S.load_trajectory(p),
                              S.TASKS[args.task][1])
        print(f"searching against recorded trajectory {args.trajectory}")

    if args.from_cem:
        spec = json.load(open(args.from_cem))
        mean = np.array(spec["params"], float)
        n_f = spec["n_fingers"]
    else:
        n_f = args.fingers
        mean = np.array([n for _, n in obj["default_contacts"](n_f)]).reshape(-1)

    driver = CEMDriver(mean, args.sigma, args.population, seed=args.seed)
    print(f"tiling {n_env} environments x {n_f} fingers, object={obj['name']}, "
          f"task={args.task}")
    t0 = time.time()
    sim = TiledSearch(n_env, n_f, obj, args.task, driver)
    print(f"model nq={sim.model.nq} nv={sim.model.nv} ngeom={sim.model.ngeom} "
          f"(built in {time.time()-t0:.1f}s)")
    print(f"episode length {sim.ref.duration:.1f}s; recording {args.seconds:.0f}s "
          f"=> about {int(args.seconds / sim.ref.duration * n_env)} episodes")

    if args.viewer:
        from mujoco import viewer as mj_viewer
        with mj_viewer.launch_passive(sim.model, sim.data) as v:
            while v.is_running():
                sim.step()
                v.sync()
        return 0

    import imageio.v2 as imageio
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
    nframe = int(args.seconds * args.fps)
    t0 = time.time()
    for n in range(nframe):
        for _ in range(spf):
            sim.step()
        r.update_scene(sim.data, camera=cam)
        wr.append_data(hud(r.render(), sim, driver, time.time() - t0))
        if n % 60 == 0:
            print(f"  {n:4d}/{nframe}  gen {driver.generation}  "
                  f"episodes {driver.n_eval}  best {driver.best_cost:8.2f}  "
                  f"{time.time()-t0:5.0f}s", flush=True)
    wr.close()
    r.close()

    print(f"\nwrote {args.out}")
    print(f"{driver.n_eval} episodes, {driver.generation} generations, "
          f"best cost {driver.best_cost:.3f}")
    if driver.history:
        print("\n  gen  evals      best    median  elite mean   sigma")
        for h in driver.history:
            print(f"  {h['gen']:3d}  {h['evals']:5d} {h['best']:9.2f} "
                  f"{h['median']:9.2f} {h['elite_mean']:11.2f}  {h['sigma']:.3f}")
    out = pathlib.Path(args.out).with_suffix(".best.json")
    json.dump({"source": "tiled_view.py", "object": obj["name"], "task": args.task,
               "n_fingers": n_f, "cost": driver.best_cost,
               "params": [float(x) for x in driver.best_params],
               "contacts": [{"offset": [float(v) for v in o],
                             "normal": [float(v) for v in nn]}
                            for o, nn in [obj["surface_point"](v)
                                          for v in driver.best_params.reshape(-1, 3)]],
               "history": driver.history}, open(out, "w"), indent=2)
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
