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

### The corner balance: one finger, and the minimiser cannot find it

A single fingertip pressed onto the vertex OPPOSITE the support vertex holds the balance.
Cube starting balanced, tipping torques applied about horizontal axes:

| tipping torque | no finger | one finger on top vertex |
|---|---|---|
| 5 mN·m | topples (56° peak, settles flat) | **0.00° tilt**, min grip 0.37 N |
| 10 mN·m | topples | **0.00° tilt** |
| 20 mN·m | topples | **0.00° tilt** |

The no-finger column confirms the equilibrium really is unstable and the disturbance
really does tip it. Both vertices lie on the body diagonal, so pressing down at the top
pins the cube between two points and any tilt has to drag the top vertex sideways against
the fingertip's friction. The lever arm is the full 104 mm diagonal, oriented purely
vertically, so a lateral force there produces torque purely about the axes that tip it. A
face-centre contact gets 73 mm, pointing the wrong way.

`min_fingers.py` returned three contacts for this, and it was wrong twice over:

1. **The candidate set excluded the answer.** It gridded face planes only. Vertices and
   edges were not candidates at all, so "3" was the minimum over the wrong set. Fixed:
   `candidate_contacts` now includes vertices and edge midpoints.
2. **The criterion is still wrong**, and fixing the candidates did not rescue it. With the
   table credited in the disturbance conditions the bar is far too low (it returned two
   face contacts, and the simulator jammed the cube against the table at a 14° lean).
   With the table excluded the bar is too high (it demands the fingers resist a pull
   upward, which nothing applies, and returns 3 to 4).

The isotropic wrench ball is simply the wrong disturbance model for an object resting on
a support point. The right criterion is rejection of tipping torque about the support
point, with the disturbance set drawn from the task's physics. That is a modelling
judgement, not an LP parameter. **For balance tasks, trust the simulator over the
minimiser.** The tool remains sound for grasps in free space where the disturbance is
roughly isotropic.

The old three-finger corner clip (`out/corner_balance.mp4`) is the broken version. Do not
show it.

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
