# Fingertip swarm object-trajectory tracking

A swarm of free-floating, unactuated visuotactile fingertips grasps a 60 mm cube and
drives it along a prescribed SE(3) trajectory through real frictional contact. MuJoCo.

## Setup

```powershell
conda env remove -n fingersim        # only if a previous attempt left a partial env
conda env create -f environment.yml
conda activate fingersim
```

One-time, if conda refuses with a Terms of Service error: it runs a ToS precheck on the
channels in your `.condarc` before it ever reads this file, so `nodefaults` here cannot
suppress it.

```powershell
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/msys2
```

Smoke test:

```powershell
python -c "import mujoco, scipy, imageio, PIL; print('mujoco', mujoco.__version__)"
python swarm_sim.py --selftest
```

**If a package's post-link script fails**, the cause is almost certainly a space in the
environment prefix. Conda warns about this but carries on, and some packages' post-link
scripts do not quote paths, so `C:\Users\Jared Grinberg\miniconda3\envs\...` breaks
them. The environment file avoids the known offender by not installing conda's ffmpeg. If
something else trips on it, put the env somewhere without spaces:

```powershell
conda env create -f environment.yml -p C:\condaenvs\fingersim
conda activate C:\condaenvs\fingersim
```

Create the env from a base prompt, not from inside `isaacsim`, and do not install into
the `isaacsim` env itself: Isaac pins a lot of packages and it is not worth disturbing.
There is also a `requirements.txt` for a plain venv.

### Windows Application Control blocks MuJoCo's DLLs

```
OSError: [WinError 4551] An Application Control policy has blocked this file
```

raised from `_load_all_bundled_plugins()` during `import mujoco`. Smart App Control or a
corporate WDAC policy is refusing MuJoCo's unsigned PyPI DLLs. The core library loads
fine; only the optional bundled plugins (elasticity, SDF, sensor) trip it, and this
project uses none of them.

```powershell
# 1. Clear the mark-of-the-web on the downloaded binaries.
Get-ChildItem "$env:CONDA_PREFIX\Lib\site-packages\mujoco" -Recurse `
    -Include *.dll,*.pyd,*.exe | Unblock-File

# 2. If still blocked, disable the plugins. _load_all_bundled_plugins walks this
#    directory; if it is absent it finds nothing and the import proceeds.
Rename-Item "$env:CONDA_PREFIX\Lib\site-packages\mujoco\plugin" plugin_disabled
```

Check what is enforcing it (1 = enforced, 2 = evaluation, 0 = off):

```powershell
Get-ItemProperty "HKLM:\SYSTEM\CurrentControlSet\Control\CI\Policy" `
    -Name VerifiedAndReputablePolicyState
```

Prefer the plugin rename over changing the policy. Smart App Control cannot be
re-enabled after being turned off without reinstalling Windows, and a managed corporate
policy will not be changeable anyway. If the rename does not work either, the fix is an
IT allowlist entry for the environment, not a local workaround.

`imageio-ffmpeg` ships its own `ffmpeg.exe` and may hit the same policy when rendering.
If it does, `Unblock-File` that binary too; its location is printed by
`python -c "import imageio_ffmpeg; print(imageio_ffmpeg.get_ffmpeg_exe())"`.

**Rendering backend.** On Windows or any machine with a GPU or display, leave `MUJOCO_GL`
unset; MuJoCo picks wgl / egl / cgl and uses the GPU. Only a headless Linux box needs
software rendering, which also needs a system package:

```bash
sudo apt-get install libosmesa6 libosmesa6-dev
export MUJOCO_GL=osmesa
```

`swarm_sim.py` sets `osmesa` itself only when it detects headless Linux, so you should not
have to touch this.

## Running

```powershell
.\run.ps1        # Windows
./run.sh         # Linux / macOS
```

Individually:

```bash
python swarm_sim.py --selftest
python swarm_sim.py --task demo --fingers 6 --out out/swarm_demo.mp4
python swarm_sim.py --task corner --control hybrid \
       --contacts contacts_corner_full.json --out out/corner_balance.mp4
python swarm_sim.py --task corner --sweep

python min_fingers.py --task corner --phase "BALANCE ON CORNER" --samples 4 --per-face 9
python min_fingers.py --task demo --samples 8 --per-face 4 --out contacts_demo.json
```

Key flags: `--fingers N` (3-12), `--contacts FILE.json` (an optimised set, overrides
`--fingers`), `--control {position,hybrid}`, `--task {demo,corner}`,
`--mode {dynamic,kinematic}`.

## Control: what it actually is

Fingers are dynamic bodies with gravity compensation, held by a stiff virtual 6-DOF
spring-damper (3000 N/m, 0.6 N·m/rad). You command a pose setpoint; the mount tracks it.

**`--control position`** (default). Setpoints come from the reference object pose with a
commanded penetration. Contact forces are whatever the penetration geometry produces.
This is impedance control. It works well for a redundant, roughly symmetric grasp.

**`--control hybrid`**. Each control tick the controller forms a commanded object wrench
(feedforward from the trajectory plus PD on object pose error), allocates it across the
contacts by an LP over friction-cone edges with a floor on each normal force, then
converts each contact force into a setpoint offset via `target = touch + f / MOUNT_KP`.
This is needed for minimal contact sets, for the reason in the next section.

## Minimising the number of fingers

`min_fingers.py` is the analysis layer and runs with no simulator. Newton-Euler on the
reference gives the exact wrench the contacts must supply at every instant. Each
candidate contact force is written on the edges of a polyhedral cone inscribed in its
Coulomb cone with nonnegative coefficients, so unilaterality and the friction limit hold
by construction and the whole thing stays linear. One cap variable per contact
upper-bounds its normal force across all conditions; minimising the sum of caps is the L1
surrogate for cardinality, reweighted by `1/(s+eps)` to approach true cardinality, then
backward elimination removes anything still redundant.

**The disturbance margin is the load-bearing part.** A cube balanced on a vertex has its
centre of mass exactly over the contact point, so the nominal required wrench is zero and
pure feasibility says *no fingers are needed*. That is correct and useless: the
equilibrium is neutral in force and unstable in attitude. So the contact set must also be
able to supply the nominal wrench perturbed along each of the six wrench axes in both
directions. What the fingers are for is rejecting the disturbance that tips it.

Minimum fingers to hold the corner balance, against a 0.5 N force disturbance:

| disturbance torque | min fingers |
|---|---|
| 0.005 N·m | 2 |
| 0.010 N·m | 2 |
| 0.025 N·m | 2 |
| 0.050 N·m | 3 |
| 0.100 N·m | 3 |

Whole-trajectory minima, 0.5 N and 0.010 N·m margin, mu = 1.2:

| task | faces available | min fingers |
|---|---|---|
| demo (lift, flip, drag, reorient) | all 6 | 3 |
| corner (lift, rotate onto vertex, balance) | 3 upper only | 3 |

Three is the classical answer for frictional force closure in 3D, which is a reasonable
sign the formulation is not broken. The cone is inscribed in the true cone, so a set this
declares feasible genuinely is; it is conservative, never optimistic.

## The gap between feasible and achievable

The LP says 3 contacts can produce the required wrench. Under **position control the
simulator drops the object in both tasks** (54 mm / 44° on corner, 118 mm / 68° on demo).

That is not a bug in either layer. A position-controlled finger cannot choose its contact
force; the force is whatever the penetration geometry gives. A minimal set needs a
specific, unbalanced force distribution that penetration geometry does not produce. The
clearest case: the corner layout uses only the three upward faces, so it has no opposing
pairs at all, and a purely normal squeeze on three mutually orthogonal faces gives a net
force toward the shared corner that nothing can cancel. The grasp shoves itself apart.
Those same three contacts *can* hold a zero-net-force internal squeeze, but only by
tilting each force inside its cone, which requires solving for the forces.

**So minimisation and force control are the same problem.** You do not get one without the
other. That is the direct answer to whether the fingertips should be force controlled:
for a redundant grasp it does not matter, and for a minimal one it is mandatory.

## Current status, honestly

| | result |
|---|---|
| demo, 6 fingers, position control | **works, and is robust** |
| friction physics check | **passes.** predicted slip mu\* = 0.1308, measured 0.13-0.14 |
| minimiser, as a wrench-feasibility tool | **runs**, but its criterion is wrong for the corner |
| corner balance | **does not work. Do not use it or show it.** |

### The demo is robust

Sweeping the fingertip friction coefficient on the 6-finger demo:

| mu | RMS pos | max pos | RMS rot | min fingers in contact |
|---|---|---|---|---|
| 1.2 | 0.40 mm | 0.99 mm | 1.23° | 6/6 |
| 0.6 | 0.44 mm | 1.00 mm | 1.79° | 6/6 |
| 0.4 | 0.48 mm | 1.23 mm | 2.18° | 6/6 |
| 0.25 | 0.98 mm | 3.75 mm | 2.86° | 5/6 |

All six contacts stay engaged throughout and a 3x cut in friction barely moves the
tracking. The equator layout has genuinely opposing pairs, so the squeeze balances
internally and friction only has to carry the 1.47 N weight rather than hold the grasp
together. Degradation only begins at mu = 0.25, below any plausible gel pad.

### Corner balance by handoff: six fingers place it, one holds it

`--task handoff` is the working corner demo. Six equator contacts (the layout proven on
the main demo) lift the cube, rotate it 54.7° onto its body diagonal and set it down on a
vertex. A seventh fingertip then descends onto the OPPOSITE vertex, takes the load, and
the six let go and fly clear.

| segment | RMS pos | max pos | RMS rot | max rot |
|---|---|---|---|---|
| lift clear | 0.44 mm | 0.57 mm | 0.23° | 0.39° |
| rotate onto the diagonal | 0.21 mm | 0.37 mm | 1.06° | 1.32° |
| lower onto the vertex | 0.17 mm | 0.26 mm | 0.33° | 0.64° |
| settle | 0.10 mm | 0.13 mm | 0.78° | 0.97° |
| HANDOFF, six let go | 0.45 mm | 0.67 mm | 0.79° | 1.10° |
| ONE FINGER HOLDING | 0.72 mm | 1.01 mm | 0.65° | 0.76° |

Contact count goes 6 → 7 → 1 across the handoff and the tilt never exceeds 0.03°. The
single vertex contact carries a steady 3.25 N. The clip ends with a deliberate release,
and the cube topples immediately, which is the evidence the equilibrium was unstable and
being actively held rather than merely resting.

**Why the vertex.** Both vertices lie on the body diagonal, so pressing down on the top
one pins the cube between two points and any tilt must drag the top vertex sideways
against the fingertip's friction. Lever arm is the full 104 mm diagonal, oriented
vertically, which is precisely the geometry that resists the two tipping axes. A
face-centre contact gets 73 mm pointing the wrong way.

**Disturbance rejection.** Single-axis tipping torque applied while only the vertex finger
is on:

| tipping torque | result |
|---|---|
| 0.05 to 0.21 N·m | held, peak tilt 0.14° to 0.52° |
| 0.25 N·m | contact breaks, object lost |

For scale, gravity itself applies 18.5 mN·m at a 14° tilt, so the margin is roughly 11x.
The tilt response is tiny because the two-point pin is stiff: the system holds nearly
rigidly right up to the point the contact breaks, with no visible wobble in between.
Watch the grip-force bar rather than the cube.

### Choosing the handoff swarm size

`--fingers N` sets the PLACEMENT swarm for `--task handoff`. The vertex finger is always
added on top, so `--fingers 4` puts five bodies in the scene. `--task handoff --sweep`
runs the range.

| placers | total | placement RMS | hold max tilt | outcome |
|---|---|---|---|---|
| 3 | 4 | diverges | 176.6° | lost |
| 4 | 5 | diverges | 153.2° | lost |
| 5 | 6 | 0.40 mm | 0.38° | **held** |
| 6 | 7 | 0.50 mm | 0.38° | **held** |
| 8 | 9 | 42.97 mm | 178.3° | lost |
| 10 | 11 | diverges | 153.4° | lost |

Two things worth knowing before reading much into this.

**Both failure modes happen after placement, not during it.** With 4 placers the cube is
placed fine and the handoff itself fails at t = 12.3. With 8 it survives the handoff and
fails at t = 17.5, which is the third and largest disturbance. So the swarm size is not
limiting the placement.

**What correlates with success is the pose accuracy at the instant of transfer**, not the
finger count:

| placers | tilt when the placers let go | outcome |
|---|---|---|
| 3 | 0.412° | lost |
| 5 | 0.040° | held |
| 6 | 0.020° | held |
| 8 | 1.121° | lost |

Four data points, so treat it as a direction to investigate rather than an established
result. If it holds up, the lever for making the handoff robust is tightening the pose at
transfer (settle longer, or close the loop on tilt before releasing), not adding fingers.
That would also explain why 8 and 10 placers are worse than 6, which otherwise looks
backwards.

### What this cost, and what it says about the minimiser

`min_fingers.py` returned three contacts for the corner and it was wrong twice over:

1. **The candidate set excluded the answer.** It gridded face planes only; vertices and
   edges were not candidates, so "3" was the minimum over the wrong set. Fixed —
   `candidate_contacts` now includes vertices and edge midpoints.
2. **The criterion is still wrong**, and fixing the candidates did not rescue it. With the
   table credited in the disturbance conditions the bar is far too low. With the table
   excluded it is too high: it demands the fingers resist a pull upward, which nothing
   applies, and returns 3 to 4.

An isotropic ball of wrench disturbances at the centre of mass is the wrong disturbance
model for an object resting on a support point. The right criterion is rejection of
tipping torque about the support point, with the disturbance set drawn from the task's
physics. That is a modelling judgement, not an LP parameter. **For balance tasks, trust
the simulator over the minimiser.** The tool remains sound for grasps in free space where
the disturbance really is roughly isotropic.

`out/corner_balance.mp4` is the earlier broken three-finger attempt. Do not show it;
`out/handoff_corner.mp4` supersedes it.

### Two control fixes the handoff forced

**Integrator windup walks a finger off the object.** The cube rests on the table slightly
below the commanded height, so the vertical error never clears, the integrator winds up,
and the commanded pose rises until the fingertip loses contact. The vertex finger's grip
bled from 3.3 N to zero over three seconds and the cube fell. Two changes: `KI_POS *
IMAX_POS` is now bounded well below the grip penetration (1.5 mm against a 3.0 mm
squeeze), so the integrator can never command a finger clear of the object; and
integration is frozen while the object is touching the table, because the table is what
sets the height and no amount of pushing will raise it to the reference.

**Released fingers must fly clear.** A finger that has let go still hovers 35 mm off the
surface, which is close enough for a toppling object to land on. Released fingers now
blend back to their home poses.

## Learning: CEM over contact placement

`learn_placement.py`. Cross-entropy method over where the fingertips sit. Sample a
population of placements, run each through the simulator, keep the best quarter, refit
the sampling distribution, repeat. The simulator is the fitness function.

```
python learn_placement.py --fingers 4 --generations 12 --population 32
python learn_placement.py --fingers 5 --min-fingers      # decrement while it still holds
python learn_placement.py --replay best_placement_n4.json
python swarm_sim.py --contacts best_placement_n4.json --out out/learned.mp4
```

**Why this and not the LP.** `min_fingers.py` answers "can these contacts produce the
required wrench", which turned out to be badly insufficient: it declared three contacts
enough for the corner balance and the simulator jammed the cube against the table. CEM
optimises whether the controller actually holds the object. It is slower and proves
nothing, but it cannot be confidently wrong the way the LP was.

**Parameterisation.** Each contact is a free 3-vector, ray-cast from the cube centre onto
the surface by `swarm_sim.surface_point`. So n contacts is 3n continuous parameters with
no discrete face indices. The map covers edges and vertices, where the normal becomes the
bisector of the adjoining faces. The LP's face-grid candidate set could not express a
vertex contact, which is exactly the contact that matters for balancing on a corner. This
one can reach it.

**Result, 4 fingers on the probe task**, 8 generations of 16:

| | cost | RMS pos | RMS rot |
|---|---|---|---|
| heuristic layout | 8.447 | 0.487 mm | 3.808° |
| CEM best | **1.704** | 0.486 mm | **0.452°** |

The entire gain is in orientation. Position tracking was already fine; the heuristic's
four-finger layout simply had poor torque authority, and the search found lever arms that
did not. Feasible fraction of the population went from 12/16 to 16/16 by generation 1.

## Watching many at once

`tiled_view.py`. MuJoCo has no tiled renderer like Isaac Lab's, and does not need one for
this: put N copies of the scene into ONE model on a grid, and a single physics step
advances all of them while a single render pass draws all of them. Copies are fully
independent (separate bodies, separate contact islands) and each runs its own contact
placement, because placement lives in the controller rather than the XML.

```
python tiled_view.py --grid 4 4 --from-cem best_placement_n4.json --out out/tiled.mp4
python tiled_view.py --grid 3 3 --random 9 --fingers 4 --viewer     # interactive
```

16 tiles is nq=560, nv=480. On 2 cores with software rendering that runs at about 22x
slower than real time; on a GPU box it is watchable. The honest limit: cost scales
roughly linearly in tiles, so this tops out in the low tens, not hundreds. It is a
visualisation tool. For throughput use independent processes (which `learn_placement.py`
already does) or MJX; for hundreds of tiles rendered live, Isaac Lab is the right tool
and this is not.

## Three harness bugs the optimiser exposed

All three made the fitness function silently meaningless rather than throwing, which is
the failure mode to watch for when the simulator becomes an objective function.

**The release schedule fired mid-episode.** `squeeze_depth` releases the grip in the
final 0.9 s of any task. The probe episode is 4.9 s, so the release fired at t=4.0 in the
middle of the rotation and every candidate was scored as having dropped the object,
identically. The fitness function was blind and returned plausible-looking numbers.
Release timing is now per task (`TASK_RELEASE`), zero for the probe.

**The lag clamp saturated, for the third time.** The probe asks for a 50° rotation
against a 34.4° clamp. Rather than fix it per task again, `Sim.__init__` now measures the
largest single rotation in the reference and raises the clamp automatically, printing
when it does. This bug had already appeared on the 90° flip and the 54.7° corner
rotation and is not diagnosable from the symptom.

**`Reference.__call__` overshot its own last keyframe.** It computed the slerp argument as
`t0 + s*(t1-t0)`, which is not exactly `t1` in floating point, and scipy's `Slerp` raises
on an argument one ulp past its last knot. Latent in every task, would fire at the final
timestep. Now clamped.

## Files

| file | what it is |
|---|---|
| `swarm_sim.py` | physics, the controller, the built-in tasks, single-env rendering |
| `objects.py` | the manipulable objects and their surface maps |
| `learn_placement.py` | CEM placement search, parallel across processes, headless |
| `tiled_view.py` | a grid of environments running a live rolling search |
| `record_trajectory.py` | author a trajectory by dragging the object with the mouse |
| `min_fingers.py` | the LP minimiser. Sound for free-space grasps, wrong for balances |

## Which controller runs where

All of them run the SAME controller, through shared free functions in `swarm_sim`:
`solve_commanded_pose`, `targets_from_pose`, `solve_mount_wrench`, `slew_limit`.

This was not true before. `tiled_view.py` carried its own transcription of the control
law and had already drifted from it: no auto-sized lag clamp, no table-contact
integrator freeze. Two copies of a controller that are supposed to be identical will
diverge, and then a result measured in one is not a result about the other. That matters
a lot more once the simulator is an objective function.

## Rolling search in the grid

Each tile runs its own episode on its own clock. When a tile finishes or drops the
object it is scored, reset on the spot, and handed the next candidate from an
asynchronous CEM. The grid therefore keeps working indefinitely and you watch the
population improve, instead of watching one fixed batch play out once. Table colour
shows the last outcome: green held, red dropped.

```
python tiled_view.py --grid 4 4 --seconds 60 --out out/search.mp4
python tiled_view.py --grid 3 3 --viewer                      # interactive
python tiled_view.py --grid 4 4 --object cylinder --fingers 5
python tiled_view.py --grid 4 4 --trajectory my_traj.json --seconds 60
```

The search is asynchronous on purpose. Tiles finish at different times (one that drops
the object early frees up sooner than one that completes), so lockstep generations would
leave most of the grid idle. Results go into a rolling buffer and the distribution
refits whenever `population` of them have arrived.

### How many tiles, honestly

Measured on 2 cores, physics only, per 1 ms step:

| tiles | ms/step | speed vs real time |
|---|---|---|
| 4 | 8.96 | 0.11x |
| 9 | 18.69 | 0.05x |
| 16 | 33.00 | 0.03x |
| 25 | 50.58 | 0.02x |

Cost is linear in tiles, about 2 ms per tile per step, and **it does not get better with
more cores**: `mj_step` on one model is essentially single-threaded, so the whole grid is
one serial solve. A faster machine buys single-core speed, maybe 2 to 3x, not 10x.

What that means in practice: the interactive viewer is comfortable at 4 to 9 tiles,
offline rendering is fine at 16 to 36, and hundreds live is not happening this way. That
is a stronger argument for Isaac Lab than the rendering one I made earlier — Isaac and
MJX parallelise the physics across environments on the GPU, where MuJoCo's single model
does not. Use `learn_placement.py` (independent processes, genuinely parallel) when you
want search throughput rather than a picture.

## Swapping the object

`objects.py`. Built in: `cube`, `box`, `sphere`, `cylinder`, `capsule`, `ellipsoid`, and
`mesh` for your own CAD.

```
python tiled_view.py --grid 4 4 --object ellipsoid
python tiled_view.py --grid 4 4 --object mesh --mesh part.stl --mesh-scale 0.001
```

Adding a shape means supplying three things: the MJCF body, a `surface_point(direction)`
that ray-casts from the centre and returns `(offset, outward normal)`, and the rest
height. The surface map is the interesting one, because it is what the optimiser
searches over, so it has to be continuous and cover edges and vertices as well as faces.
All the built-ins return the true outward normal, which at an edge or vertex is the
bisector of the adjoining faces.

Mesh caveats, in order of how likely they are to bite:

- **Collision is the convex hull.** That is MuJoCo's default and it is what
  `surface_point` probes, so the optimiser and the physics agree with each other, but
  both may disagree with your actual part. For a meaningfully concave part, split it
  into convex pieces, or keep primitives for collision and use the mesh for visual only.
- **Units.** STL is unitless and MuJoCo is metres, so CAD in mm needs
  `--mesh-scale 0.001`.
- **Inertia** is computed from the mesh assuming uniform density. For a finger with a
  camera in it that is wrong; override `<inertial>`.
- The mesh normal is estimated from nearby ray hits, because `mj_ray` returns a distance
  and a geom id but no normal.

## Authoring a trajectory by dragging it

`record_trajectory.py`. Three phases.

```
python record_trajectory.py --record --out my_traj.json
python record_trajectory.py --check my_traj.json
python swarm_sim.py --trajectory my_traj.json --out out/replay.mp4
python tiled_view.py --grid 4 4 --trajectory my_traj.json --seconds 60
```

**Record.** The object loads alone, as a mocap body, with no fingers and no gravity.
Mocap because a mocap body is kinematically positioned, so dragging is not a negotiation
with the dynamics: it goes where you put it. Ctrl and left-drag translates, ctrl and
right-drag rotates. Close the window when done.

**Fit.** The path is resampled, smoothed, and reduced to keyframes.

**Check.** This is the part worth not skipping. A dragged path is not dynamically
feasible in general: you can pull the object through the table, or move it at 5 m/s, or
demand a rotation no friction could deliver. Replay an infeasible demonstration and the
swarm fails, and you cannot tell whether the controller is bad or the demonstration was
impossible. The checker reports peak speed, acceleration, angular rate, required contact
force and minimum height, and names the problems it finds.

Two things this needed that are worth knowing:

- **A lead-in hold.** A recording starts moving at t=0, but the swarm needs time to fly
  in and close. Without a prepended hold the object is commanded to move while the
  fingers are mid-air, error runs away immediately, and it looks like a controller
  failure. `load_trajectory` prepends one covering the grasp.
- **Linear interpolation, not smoothstep.** `Reference` eases between keyframes, which
  forces zero velocity at every key. That is right for hand-authored hold/move keyframes
  and wrong for a densely sampled recording, where it becomes one stop-start per key.
  Recorded paths are already smoothed and interpolate linearly (`Reference(..., ease=False)`).

With both fixes a recorded lift-and-twist replays at 0.30 mm and 0.32 deg RMS. Without
them it was 72 mm.

**Not tested interactively.** The sandbox this was written in has no display, so the
record loop itself has never been driven by a real mouse. The fit, feasibility and
replay stages are tested against a synthetic recording. Expect to shake something out of
the viewer loop on first run.

## Four things that bit, and will bite again

**MuJoCo mocap bodies cannot grip.** A mocap body has no degrees of freedom, so the
solver reads its velocity as zero, and Coulomb friction opposes relative velocity. A
mocap finger generates no friction at all: it can push by resolving penetration but can
never carry. First build had six fingers gripping at 1762 N with the cube motionless on
the table. Fingers must be dynamic bodies.

**A saturated lag clamp deadlocks.** If the commanded pose is clamped to a bounded lead
over the actual pose, once the clamp saturates the fingers stop moving, relative slip goes
to zero, friction goes to zero, the object never starts moving and the clamp never
releases. This is why `MAX_LAG_ROT` is per task: the corner rotates 54.7° onto the body
diagonal and the demo's 28.6° clamp cannot accommodate it. Every corner run failed
identically at ~56° until this was fixed.

**Freejoint angular velocity is in the body frame**, `xfrc_applied` torque is in the world
frame. Mixing them makes the mount damping positive feedback for any finger not at
identity orientation. One finger reached 75,000 rad/s.

**Geometry defined along a rotated axis gets rotated twice.** `finger_quat()` already maps
body +x onto the outward contact normal, so writing the shaft geometry along `normal`
double-rotated it and put the whole shaft inside the cube on the −x face.

## Physics coverage

Colliding: tip/object, tip/table, tip/tip, shaft/object, shaft/table, shaft/shaft.
Coulomb friction with an elliptic cone throughout. Gravity on the object, off the fingers
(`gravcomp="1"`). Finite contact compliance, 4000 N/m, standing in for gel pad softness.
The only non-colliding geometry is the cube's coloured face stickers, which are flush with
its surface and purely cosmetic.

Still simplified: fingertips are rigid spheres, so no contact patch and no torsional
friction from pad deformation. No gel camera rendering. No contact scheduling or
regrasping; each finger keeps one contact for the whole run.

## If this moves to Isaac

Controller, trajectory and minimiser are engine-independent; only `build_xml` and the
`Sim` step/state methods touch MuJoCo. The virtual mount becomes a 6-DOF articulation root
with drive stiffness and damping, gravity compensation becomes a per-body gravity flag,
and the negative-`solref` contact stiffness becomes PhysX compliant contact. Re-run
`--selftest` after porting: PhysX contact differs enough that the friction threshold is
worth re-checking, and GPU contact solving is not bit-reproducible, which matters once
anything optimises in the loop.
