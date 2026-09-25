"""
Swarm of free-floating visuotactile fingertips tracking a prescribed object trajectory.

A rigid object (Rubik's-cube-like, 60 mm) rests on a table. A reference SE(3)
trajectory is prescribed for it. A swarm of unconnected, non-actuated fingertips
flies in, grasps the object, and drives it along that trajectory through real
frictional contact.

Two modes:

  dynamic   (option B, the default)
            The object is a dynamic rigid body under gravity. The fingertips are
            position-commanded kinematic bodies that genuinely push it. An
            object-level proportional-derivative law displaces the fingertip
            setpoints to drive the object onto the reference.

  kinematic (option A, the fallback)
            The object pose is written directly from the reference each step and
            the fingertips are pinned to its surface. Physics is decorative.
            Cannot fail, proves nothing.

Gravity acts on the object and on nothing else: the fingertips are kinematic, so
they are weightless and free-floating by construction, but they still collide
with the object, the table and each other.

Usage:
    MUJOCO_GL=osmesa python3 swarm_sim.py --mode dynamic --out out/swarm_demo.mp4
    MUJOCO_GL=osmesa python3 swarm_sim.py --mode kinematic --out out/fallback.mp4
    MUJOCO_GL=osmesa python3 swarm_sim.py --selftest
"""

from __future__ import annotations

import argparse
import os
import pathlib
import sys

import numpy as np

if "MUJOCO_GL" not in os.environ:
    # On a desktop with a display, let MuJoCo pick its native backend (wgl on Windows,
    # glfw/egl on Linux, cgl on macOS). Only force software rendering when headless,
    # otherwise this would cripple a machine that has a GPU.
    if sys.platform.startswith("linux") and not os.environ.get("DISPLAY"):
        os.environ["MUJOCO_GL"] = "osmesa"

import mujoco  # noqa: E402
from scipy.spatial.transform import Rotation as Rot  # noqa: E402
from scipy.spatial.transform import Slerp  # noqa: E402

# --------------------------------------------------------------------------------------
# Physical parameters. Everything the model depends on lives here.
# --------------------------------------------------------------------------------------

CUBE_HALF = 0.030          # m, half-edge of the object (60 mm cube)
CUBE_MASS = 0.150          # kg
TIP_RADIUS = 0.009         # m, spherical fingertip radius
SHAFT_LEN = 0.032          # m, finger shaft (collides)
MU_TIP = 1.20              # fingertip/object Coulomb friction (compliant gel)
MU_TABLE = 0.45            # object/table Coulomb friction
TIMESTEP = 0.001           # s
GRAVITY = 9.81

# Contact layout in the object body frame. Contacts are spread evenly around the cube's
# equator and snapped onto whichever side face the ray hits, so every fingertip
# contributes inward normal force and therefore lift friction. The z offsets alternate so
# the set has some authority about the horizontal axes instead of lying in one plane.
#
# This is a placement HEURISTIC, not an optimiser. Nothing here searches for good contact
# locations or for the smallest set that works.

def make_contacts(n):
    """n contact (offset, outward normal) pairs on the four side faces."""
    a = CUBE_HALF
    margin = a - TIP_RADIUS - 0.003      # keep the tip sphere clear of the face edges
    out = []
    for k in range(n):
        ang = 2.0 * np.pi * k / n
        c, s = np.cos(ang), np.sin(ang)
        if abs(abs(c) - abs(s)) < 1e-6:  # ray through a vertical edge: nudge off it
            ang += 0.09
            c, s = np.cos(ang), np.sin(ang)
        t = a / max(abs(c), abs(s))
        px, py = t * c, t * s
        if abs(abs(px) - a) < 1e-9:      # landed on a +/-x face
            nrm = np.array([np.sign(px), 0.0, 0.0])
            py = float(np.clip(py, -margin, margin))
            px = np.sign(px) * a
        else:                            # landed on a +/-y face
            nrm = np.array([0.0, np.sign(py), 0.0])
            px = float(np.clip(px, -margin, margin))
            py = np.sign(py) * a
        pz = 0.009 if k % 2 == 0 else -0.009
        out.append((np.array([px, py, pz]), nrm))
    return out


def make_home(contacts):
    """Start poses in free space, each one out along its own contact normal.

    Each finger must start on the same side of the object as the contact it is going to
    take. Scattering the start poses arbitrarily assigns some fingers a home across the
    cube from their target, and the fly-in then drives them straight through it.
    """
    out = []
    for k, (off, nrm) in enumerate(contacts):
        base = np.array([off[0], off[1], CUBE_HALF + off[2]])
        tang = np.array([-nrm[1], nrm[0], 0.0])        # horizontal, across the normal
        phase = (k * 0.618) % 1.0
        home = (base
                + nrm * 0.21
                + tang * (0.055 * (2.0 * phase - 1.0))
                + np.array([0.0, 0.0, 0.02 + 0.15 * phase]))
        out.append(home)
    return np.array(out)


N_FINGERS = 6              # overridden by --fingers
CONTACTS = make_contacts(N_FINGERS)
HOME = make_home(CONTACTS)


def set_finger_count(n, contacts=None):
    """Rebuild the contact layout and start poses for a different swarm size.

    `contacts` overrides the heuristic entirely, which is how an optimised set from
    min_fingers.py gets loaded in.
    """
    global N_FINGERS, CONTACTS, HOME, FINGER_COLORS
    if contacts is not None:
        CONTACTS = [(np.asarray(o, float), np.asarray(v, float)) for o, v in contacts]
        N_FINGERS = len(CONTACTS)
    else:
        N_FINGERS = n
        CONTACTS = (make_corner_contacts(n) if globals().get("LAYOUT") == "corner"
                    else make_contacts(n))
    HOME = make_home(CONTACTS)
    FINGER_COLORS = [BASE_COLORS[i % len(BASE_COLORS)] for i in range(N_FINGERS)]

# Fingertip contact compliance. Negative solref in MuJoCo means (-stiffness, -damping)
# in physical units rather than (timeconst, dampratio), which is what we want here: the
# gel pad IS the compliant element, so a commanded penetration maps to a finite, sane
# force. Leaving the contact rigid makes the squeeze ill-posed, because six
# infinite-mass fingertips commanded to overlap a rigid cube is a constraint system with
# no solution, and the solver freezes the object at absurd internal force.
TIP_STIFFNESS = 4000.0     # N/m
TIP_DAMPING = 60.0         # N s/m

# Fingertip body and its virtual 6-DOF mount.
#
# The fingers must be DYNAMIC bodies, not kinematic ones. A MuJoCo mocap body has no
# degrees of freedom, so the constraint solver reads its velocity as zero: it can push
# an object by resolving penetration, but it can never drag one through friction,
# because Coulomb friction opposes relative velocity and the solver believes there is
# none. A purely kinematic finger therefore cannot carry anything, however hard it grips.
#
# So each finger is a free body with gravity compensation, held by a stiff virtual
# spring-damper whose setpoint we command. From the outside it is still
# position-commanded: you hand it a pose. Internally the mount is what makes the contact
# forces finite and the friction real.
FINGER_MASS = 0.05         # kg
FINGER_INERTIA = 5e-5      # kg m^2, isotropic
MOUNT_KP = 3000.0          # N/m
MOUNT_KD = 32.0            # N s/m
MOUNT_KR = 0.60            # N m/rad
MOUNT_KW = 0.010           # N m s/rad
MOUNT_FMAX = 60.0          # N, saturation on the mount force
MOUNT_TMAX = 2.0           # N m, saturation on the mount torque

# Object-level controller (option B).
#
# The fingers are placed on the REFERENCE pose, clamped to stay within a bounded lag of
# the object's ACTUAL pose. Two things follow from that:
#
#   - While tracking is good the clamp is inactive, the fingers sit exactly on the
#     reference, and the object is carried by the resulting relative slip. A finger that
#     is merely parked at a static offset transmits nothing: Coulomb friction opposes
#     relative velocity, not accumulated displacement, so a stalled finger carries no
#     load however hard it grips.
#   - If the object falls badly behind (the grasp slips) the clamp holds the fingers
#     back so they wait for it instead of flying off and dropping it.
# Integral action matters more here than it looks. A frictional grasp under pure
# position control has a whole cone of static equilibria: once the object stops, static
# friction is content to hold it a few millimetres off the reference forever, because
# correcting requires relative motion and there is none. A slow integrator forces the
# creep that generates the slip that removes the offset.
MAX_LAG = 0.030            # m, how far ahead of the object the fingers may be commanded
MAX_LAG_ROT = 0.50         # rad, same for orientation (~29 deg); set per task
KD_POS = 0.012             # s, velocity-error damping on the commanded pose
KD_ROT = 0.004             # s, same for angular velocity error
KI_POS = 3.0               # 1/s, integral gain on position error
KI_ROT = 2.0               # 1/s, integral gain on orientation error
# The integral clamp must keep KI_POS * IMAX_POS strictly below the grip penetration,
# or the integrator can walk a finger clean off the object it is holding. At 3.0 * 0.0005
# the most it can ever command is 1.5 mm against a 3.0 mm squeeze.
IMAX_POS = 0.0005          # m, anti-windup clamp
IMAX_ROT = 0.05            # rad, anti-windup clamp
SLEW = 1.60                # m/s, rate limit on fingertip setpoint motion

BASE_COLORS = [
    "0.95 0.42 0.31 1", "0.99 0.72 0.24 1", "0.45 0.78 0.45 1",
    "0.35 0.66 0.92 1", "0.72 0.52 0.92 1", "0.95 0.55 0.75 1",
    "0.55 0.85 0.80 1", "0.90 0.90 0.55 1", "0.80 0.60 0.45 1",
    "0.65 0.72 0.95 1", "0.92 0.65 0.90 1", "0.60 0.90 0.62 1",
]
FINGER_COLORS = list(BASE_COLORS[:6])


# --------------------------------------------------------------------------------------
# Model
# --------------------------------------------------------------------------------------

def build_xml() -> str:
    """Assemble the MJCF. Fingertips are mocap bodies: kinematic, weightless, colliding."""
    a = CUBE_HALF
    sticker = a * 0.78
    inset = a + 0.0009

    # Visual-only face stickers so the object reads as a Rubik's cube on video.
    faces = [
        ("+x", f"{inset} 0 0", f"0.001 {sticker} {sticker}", "0.85 0.16 0.14 1"),
        ("-x", f"-{inset} 0 0", f"0.001 {sticker} {sticker}", "0.95 0.48 0.09 1"),
        ("+y", f"0 {inset} 0", f"{sticker} 0.001 {sticker}", "0.10 0.38 0.75 1"),
        ("-y", f"0 -{inset} 0", f"{sticker} 0.001 {sticker}", "0.11 0.55 0.25 1"),
        ("+z", f"0 0 {inset}", f"{sticker} {sticker} 0.001", "0.96 0.96 0.94 1"),
        ("-z", f"0 0 -{inset}", f"{sticker} {sticker} 0.001", "0.97 0.85 0.10 1"),
    ]
    sticker_xml = "\n".join(
        f'      <geom name="sticker_{n}" type="box" pos="{p}" size="{s}" rgba="{c}" '
        f'contype="0" conaffinity="0" group="1" mass="0"/>'
        for n, p, s, c in faces
    )

    fingers_xml = []
    for i, (_, normal) in enumerate(CONTACTS):
        # Shaft runs outward from the tip along the contact normal. Visual only, so
        # shaft-shaft intersections never destabilise the contact solve.
        # The shaft points along the body's +x axis, NOT along `normal`. finger_quat()
        # already rotates body +x onto the outward contact normal, so writing the shaft
        # along `normal` here rotates it twice: on the -x face that put the whole shaft
        # inside the cube with only the tip outside.
        shaft_to = np.array([SHAFT_LEN, 0.0, 0.0])
        fingers_xml.append(f"""
    <body name="finger{i}" pos="{HOME[i][0]} {HOME[i][1]} {HOME[i][2]}" gravcomp="1">
      <freejoint name="fj{i}"/>
      <inertial pos="0 0 0" mass="{FINGER_MASS}"
                diaginertia="{FINGER_INERTIA} {FINGER_INERTIA} {FINGER_INERTIA}"/>
      <geom name="tip{i}" type="sphere" size="{TIP_RADIUS}" rgba="{FINGER_COLORS[i]}"
            friction="{MU_TIP} 0.02 0.002" solref="-{TIP_STIFFNESS} -{TIP_DAMPING}"
            solimp="0.92 0.97 0.001" condim="4" priority="2" mass="0"/>
      <geom name="shaft{i}" type="capsule" fromto="0.004 0 0 {shaft_to[0]} 0 0"
            size="0.0052" rgba="0.30 0.32 0.36 1" friction="0.6 0.01 0.001"
            condim="3" mass="0"/>
      <geom name="cuff{i}" type="cylinder"
            fromto="{shaft_to[0]*0.62} 0 0 {shaft_to[0]*0.80} 0 0"
            size="0.0088" rgba="0.16 0.17 0.20 1" friction="0.6 0.01 0.001"
            condim="3" mass="0"/>
    </body>""")

    return f"""
<mujoco model="fingertip_swarm">
  <compiler angle="radian" autolimits="true"/>
  <option timestep="{TIMESTEP}" gravity="0 0 -{GRAVITY}" integrator="implicitfast"
          cone="elliptic" impratio="10" noslip_iterations="3"/>
  <size njmax="600" nconmax="300"/>

  <visual>
    <global offwidth="1920" offheight="1080" azimuth="130" elevation="-18"/>
    <quality shadowsize="4096" offsamples="4"/>
    <map znear="0.01" zfar="30"/>
    <headlight ambient="0.35 0.35 0.36" diffuse="0.45 0.45 0.45" specular="0.2 0.2 0.2"/>
  </visual>

  <asset>
    <texture name="sky" type="skybox" builtin="gradient" rgb1="0.16 0.18 0.24"
             rgb2="0.04 0.05 0.07" width="512" height="512"/>
    <texture name="tabletex" type="2d" builtin="checker" rgb1="0.30 0.31 0.34"
             rgb2="0.25 0.26 0.29" width="512" height="512"/>
    <material name="tablemat" texture="tabletex" texrepeat="9 9" specular="0.2"
              shininess="0.3" reflectance="0.06"/>
    <material name="cubemat" rgba="0.09 0.09 0.10 1" specular="0.35" shininess="0.5"/>
  </asset>

  <worldbody>
    <light name="key" pos="0.45 -0.40 1.05" dir="-0.4 0.38 -1" directional="true"
           diffuse="0.72 0.71 0.70" specular="0.25 0.25 0.25" castshadow="true"/>
    <light name="fill" pos="-0.55 0.45 0.80" dir="0.5 -0.42 -1" directional="true"
           diffuse="0.26 0.27 0.30" castshadow="false"/>

    <geom name="table" type="box" pos="0 0 -0.02" size="0.60 0.60 0.02"
          material="tablemat" friction="{MU_TABLE} 0.01 0.001" condim="4"/>

    <body name="cube" pos="0 0 {a}">
      <freejoint name="cube_free"/>
      <geom name="cube_geom" type="box" size="{a} {a} {a}" material="cubemat"
            mass="{CUBE_MASS}" friction="{MU_TABLE} 0.01 0.001" condim="4"
            solref="0.004 1" solimp="0.95 0.99 0.001"/>
{sticker_xml}
    </body>
{"".join(fingers_xml)}

    <camera name="hero" pos="0.42 -0.40 0.32" xyaxes="0.69 0.72 0 -0.21 0.20 0.96"/>
    <camera name="side" pos="0.02 -0.62 0.16" xyaxes="1 0 0 0 0.26 0.97"/>
  </worldbody>
</mujoco>
"""


# --------------------------------------------------------------------------------------
# Reference trajectory
# --------------------------------------------------------------------------------------

def smoothstep(u: float) -> float:
    """C2-continuous 0->1 ramp. Zero velocity and acceleration at both ends."""
    u = min(max(u, 0.0), 1.0)
    return u * u * u * (u * (6.0 * u - 15.0) + 10.0)


class Reference:
    """Keyframed SE(3) reference with smoothstep position blending and slerp rotation."""

    def __init__(self, keys):
        self.t = np.array([k[0] for k in keys], dtype=float)
        self.p = np.array([k[1] for k in keys], dtype=float)
        rots = Rot.from_quat([k[2] for k in keys])   # scipy uses xyzw
        self.slerp = Slerp(self.t, rots)
        self.duration = float(self.t[-1])

    def __call__(self, t: float):
        t = min(max(t, self.t[0]), self.t[-1])
        i = int(np.clip(np.searchsorted(self.t, t) - 1, 0, len(self.t) - 2))
        t0, t1 = self.t[i], self.t[i + 1]
        s = smoothstep((t - t0) / (t1 - t0)) if t1 > t0 else 0.0
        pos = self.p[i] + s * (self.p[i + 1] - self.p[i])
        # Re-time the slerp through the same smoothstep so translation and rotation
        # stay synchronised and both start and stop with zero rate.
        rot = self.slerp(t0 + s * (t1 - t0))
        return pos, rot


def q(axis, deg):
    return Rot.from_rotvec(np.array(axis, dtype=float) * np.deg2rad(deg)).as_quat()


IDENT = Rot.identity().as_quat()
REST_Z = CUBE_HALF

# --- corner balance -------------------------------------------------------------------
# Resting on a single vertex. The body diagonal must be vertical, so rotate the body
# diagonal (1,1,1)/sqrt(3) onto +z; the (-1,-1,-1) vertex is then the lowest point and
# the centre sits half a body diagonal above the table.
_diag = np.array([1.0, 1.0, 1.0]) / np.sqrt(3.0)
_axis = np.cross(_diag, np.array([0.0, 0.0, 1.0]))
_axis = _axis / np.linalg.norm(_axis)
CORNER_ROT = Rot.from_rotvec(_axis * np.arccos(1.0 / np.sqrt(3.0)))
CORNER_QUAT = (Rot.from_rotvec([0, 0, np.deg2rad(20.0)]) * CORNER_ROT).as_quat()
CORNER_Z = CUBE_HALF * np.sqrt(3.0)        # 51.96 mm for a 60 mm cube

# Balanced on a vertex the support region is a single point and the centre of mass sits
# directly above it, so gravity exerts no restoring torque at all: the equilibrium is
# neutral in force and unstable in attitude. Whatever holds the attitude has to be the
# fingers.


def corner_upper_faces():
    """The three body-frame face normals that point upward in the corner-balance pose.

    With the body diagonal vertical, each of the +x, +y and +z face normals tilts 35.3
    degrees above horizontal, and each of the -x, -y, -z normals tilts the same amount
    below. Fingers may only take the upper three: a finger on a downward face sits under
    the cube with its shaft driven into the table.
    """
    return [np.array([1.0, 0.0, 0.0]),
            np.array([0.0, 1.0, 0.0]),
            np.array([0.0, 0.0, 1.0])]


def make_corner_contacts(n):
    """n contacts spread over the three upward-facing faces of the corner-balance pose."""
    a = CUBE_HALF
    margin = a - TIP_RADIUS - 0.003
    faces = corner_upper_faces()
    # Two in-face axes per face, for spreading multiple contacts across it.
    out = []
    for k in range(n):
        nrm = faces[k % 3]
        ring = k // 3
        u = np.array([nrm[2], nrm[0], nrm[1]])      # a perpendicular axis
        v = np.cross(nrm, u)
        if ring == 0:
            off = nrm * a
        else:
            ang = 2.0 * np.pi * ((ring - 1) / max(1, (n - 3) / 3.0 + 1e-9)) + 0.4 * ring
            rad = min(margin, 0.016)
            off = nrm * a + u * (rad * np.cos(ang)) + v * (rad * np.sin(ang))
        out.append((off, nrm.copy()))
    return out


def build_reference() -> tuple[Reference, list[tuple[float, float, str]]]:
    """Four motions, each one the object could plausibly be asked to perform.

    Deliberately excludes balancing on a corner: that is a genuinely hard
    quasi-static problem and does not belong in a first demo.
    """
    k = []
    seg = []
    t = 0.0

    def hold(dur, pos, quat, label=None):
        nonlocal t
        k.append((t, pos, quat))
        if label:
            seg.append((t, t + dur, label))
        t += dur
        k.append((t, pos, quat))

    def move(dur, pos, quat, label):
        nonlocal t
        seg.append((t, t + dur, label))
        t += dur
        k.append((t, pos, quat))

    # Approach and grasp happen while the object sits still.
    hold(3.0, [0, 0, REST_Z], IDENT, "approach + grasp")

    # 1. Lift off the table, hold, set back down.
    move(2.0, [0, 0, REST_Z + 0.17], IDENT, "lift off table")
    move(1.2, [0, 0, REST_Z + 0.17], IDENT, "hold aloft")
    move(2.0, [0, 0, REST_Z], IDENT, "place")
    move(0.6, [0, 0, REST_Z], IDENT, "settle")

    # 2. Tip 90 degrees onto a side face and back. Lift clear first so the rotation
    #    is not fighting the table.
    move(1.2, [0, 0, REST_Z + 0.10], IDENT, "clear table")
    move(2.8, [0, 0, REST_Z + 0.10], q([1, 0, 0], 90), "flip 90 deg")
    move(0.8, [0, 0, REST_Z + 0.10], q([1, 0, 0], 90), "hold flipped")
    move(2.8, [0, 0, REST_Z + 0.10], IDENT, "flip back")
    move(1.2, [0, 0, REST_Z], IDENT, "place")
    move(0.5, [0, 0, REST_Z], IDENT, "settle")

    # 3. Drag across the table surface, staying in contact with it.
    move(2.5, [0.20, 0, REST_Z], IDENT, "drag across table")
    move(0.5, [0.20, 0, REST_Z], IDENT, "settle")

    # 4. Combined move: lift, yaw, translate, place. The one that looks hardest.
    move(1.5, [0.20, 0, REST_Z + 0.13], IDENT, "lift")
    move(2.5, [0.00, 0.14, REST_Z + 0.13], q([0, 0, 1], 75), "reorient + translate")
    move(1.8, [0.00, 0.14, REST_Z], q([0, 0, 1], 75), "place")
    move(1.0, [0.00, 0.14, REST_Z], q([0, 0, 1], 75), "release")

    return Reference(k), seg


def build_corner_reference():
    """Lift, rotate onto a vertex, balance there, then come back down flat.

    The balance phase is the point of this one. Standing on a vertex the support region
    is a single point with the centre of mass directly above it, so there is no
    restoring torque from gravity and the attitude is held entirely by the fingers.
    """
    k, seg = [], []
    t = 0.0

    def hold(dur, pos, quat, label=None):
        nonlocal t
        k.append((t, pos, quat))
        if label:
            seg.append((t, t + dur, label))
        t += dur
        k.append((t, pos, quat))

    def move(dur, pos, quat, label):
        nonlocal t
        seg.append((t, t + dur, label))
        t += dur
        k.append((t, pos, quat))

    hold(3.0, [0, 0, REST_Z], IDENT, "approach + grasp")
    move(1.6, [0, 0, REST_Z + 0.09], IDENT, "lift clear")
    move(3.0, [0, 0, CORNER_Z + 0.055], CORNER_QUAT, "rotate onto the diagonal")
    move(2.0, [0, 0, CORNER_Z], CORNER_QUAT, "lower onto the vertex")
    move(6.0, [0, 0, CORNER_Z], CORNER_QUAT, "BALANCE ON CORNER")
    # Then let go on purpose. Balanced on a vertex the equilibrium is unstable, so the
    # cube topples the instant the fingers leave. That topple is the evidence the balance
    # was real and not the object resting on something.
    move(3.0, [0, 0, CORNER_Z], CORNER_QUAT, "RELEASE - it topples")
    return Reference(k), seg


HANDOFF_PLACERS = 6        # equator fingers that lift and place; --fingers sets this


def make_handoff_contacts(n_place=None):
    """Six equator contacts that can lift and place, plus one on the top vertex.

    The vertex contact is the whole point. Both vertices lie on the body diagonal, so
    once the cube is standing on its bottom vertex, pressing down on the top one pins it
    between two points: any tilt has to drag the top vertex sideways against the
    fingertip's friction. Lever arm is the full body diagonal, oriented vertically, which
    is exactly the geometry that resists the two tipping axes.
    """
    a = CUBE_HALF
    diag = np.array([1.0, 1.0, 1.0]) / np.sqrt(3.0)
    n_place = HANDOFF_PLACERS if n_place is None else n_place
    return make_contacts(n_place) + [(np.array([a, a, a]), diag)]


# The handoff timeline. Kept here so the trajectory and the grasp schedule cannot drift
# apart: the vertex finger must be loaded BEFORE the equator fingers let go.
HO_SETTLED = 11.1          # cube standing on its vertex, still held by all six
HO_VERTEX_ON = (10.6, 11.6)
HO_EQUATOR_OFF = (11.9, 12.9)
HO_HOLD_END = 19.1
HO_VERTEX_OFF = (HO_HOLD_END, HO_HOLD_END + 0.7)


def build_handoff_reference():
    """Swarm places the cube on its vertex, then hands off to a single finger."""
    k, seg = [], []
    t = 0.0

    def hold(dur, pos, quat, label=None):
        nonlocal t
        k.append((t, pos, quat))
        if label:
            seg.append((t, t + dur, label))
        t += dur
        k.append((t, pos, quat))

    def move(dur, pos, quat, label):
        nonlocal t
        seg.append((t, t + dur, label))
        t += dur
        k.append((t, pos, quat))

    hold(3.0, [0, 0, REST_Z], IDENT, "approach + grasp")
    move(1.6, [0, 0, REST_Z + 0.09], IDENT, "lift clear")
    move(3.0, [0, 0, CORNER_Z + 0.055], CORNER_QUAT, "rotate onto the diagonal")
    move(2.0, [0, 0, CORNER_Z], CORNER_QUAT, "lower onto the vertex")
    move(1.5, [0, 0, CORNER_Z], CORNER_QUAT, "settle")
    move(2.0, [0, 0, CORNER_Z], CORNER_QUAT, "HANDOFF - six let go")
    move(6.0, [0, 0, CORNER_Z], CORNER_QUAT, "ONE FINGER HOLDING")
    move(2.5, [0, 0, CORNER_Z], CORNER_QUAT, "release - it topples")
    return Reference(k), seg


def setup_handoff(n_place=None):
    """Contacts, per-finger schedule and the nudges that prove the hold is active.

    `n_place` is the size of the placement swarm. The vertex finger is always added on
    top of it, so --fingers 4 means five bodies in the scene: four that lift and place,
    one that takes over at the end.
    """
    global FINGER_SCHEDULE, DISTURBANCES, HANDOFF_PLACERS
    if n_place is not None:
        HANDOFF_PLACERS = n_place
    n = HANDOFF_PLACERS
    set_finger_count(0, contacts=make_handoff_contacts(n))
    FINGER_SCHEDULE = (
        [(GRASP_START, GRASP_END, *HO_EQUATOR_OFF) for _ in range(n)]
        + [(HO_VERTEX_ON[0], HO_VERTEX_ON[1], *HO_VERTEX_OFF)]
    )
    # Tipping torques about horizontal axes, while only the vertex finger is on.
    # 0.15 N m is an order of magnitude above the 18.5 mN m that gravity itself applies
    # at a 14 degree tilt, and sits just under the ~0.21 N m single-axis threshold where
    # the vertex contact breaks. The tilt response is tiny because the two-point pin is
    # stiff; watch the grip-force bar rather than the cube.
    DISTURBANCES = [
        (14.2, 14.35, [0, 0, 0, 0.15, 0, 0]),
        (15.8, 15.95, [0, 0, 0, 0, -0.15, 0]),
        (17.4, 17.55, [0, 0, 0, 0.106, 0.106, 0]),
    ]


TASKS = {
    "demo": (build_reference, "equator"),
    "corner": (build_corner_reference, "corner"),
    "handoff": (build_handoff_reference, "equator"),
}
TASK = "demo"
LAYOUT = "equator"


# The rotational lag clamp has to exceed the largest rotation the task asks for in one
# move, otherwise it saturates and deadlocks: a clamped, stalled finger transmits no
# friction, so the object never starts turning and the clamp never releases. The corner
# task rotates 54.7 degrees onto the body diagonal, which the demo's 28.6 degree clamp
# cannot accommodate.
TASK_LAG_ROT = {"demo": 0.50, "corner": 1.20, "handoff": 1.20}


def set_task(name):
    global TASK, LAYOUT, MAX_LAG_ROT
    if name not in TASKS:
        raise SystemExit(f"unknown task {name!r}; choose from {sorted(TASKS)}")
    global FINGER_SCHEDULE, DISTURBANCES
    TASK = name
    LAYOUT = TASKS[name][1]
    MAX_LAG_ROT = TASK_LAG_ROT[name]
    FINGER_SCHEDULE = None
    DISTURBANCES = []
    if name == "handoff":
        setup_handoff()
    else:
        set_finger_count(N_FINGERS)


# --- hybrid position/force control -----------------------------------------------------
# Pure position control cannot realise an arbitrary contact-force distribution: the force
# at each contact is whatever the penetration geometry produces. That is fine for a
# redundant, roughly symmetric grasp, and it fails for a minimal one, where the required
# forces are specific and unbalanced. In hybrid mode the controller commands an object
# wrench, allocates it across the contacts inside their friction cones, and converts each
# contact force into a setpoint offset through the known mount stiffness.
CONTROL = "position"       # or "hybrid"
ALLOC_EDGES = 6            # friction-cone edges used by the allocator
ALLOC_EVERY = 10           # physics steps between allocation solves
PRELOAD = 2.5              # N, floor on each contact's normal force
WK_P, WK_D = 220.0, 14.0   # object force feedback: N/m and N s/m
WK_R, WK_W = 0.30, 0.012   # object torque feedback: N m/rad and N m s/rad


def cone_basis(normal, mu, n_edge=ALLOC_EDGES):
    """Edges of a polyhedral cone inscribed in the Coulomb cone about `normal`."""
    n = np.asarray(normal, float)
    n = n / np.linalg.norm(n)
    tmp = np.array([1.0, 0.0, 0.0]) if abs(n[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
    u = np.cross(n, tmp)
    u /= np.linalg.norm(u)
    v = np.cross(n, u)
    return np.array([n + mu * (np.cos(2 * np.pi * j / n_edge) * u
                               + np.sin(2 * np.pi * j / n_edge) * v)
                     for j in range(n_edge)])


# Per-finger engage/release schedule, for tasks where the grasp changes mid-run.
# Each entry is (engage_start, engage_end, release_start, release_end) in seconds.
# None means every finger follows the single global grasp schedule below.
FINGER_SCHEDULE = None

# Disturbances applied to the object, as (t_start, t_end, wrench6) in world frame. Used
# to show that a balance is actively held rather than merely undisturbed.
DISTURBANCES = []

GRASP_START, GRASP_END = 0.6, 2.8      # s, fingers fly in and close
RELEASE_START = None                   # filled in from the reference duration


def squeeze_depth(t: float, t_end: float) -> float:
    """Signed fingertip penetration into the nominal surface.

    Negative means hovering clear of the object. Positive means pressed in, which is
    what generates normal force through the contact model.
    """
    approach_gap = -0.035
    grip = 0.0030
    if t < GRASP_START:
        return approach_gap
    if t < GRASP_END:
        return approach_gap + (grip - approach_gap) * smoothstep(
            (t - GRASP_START) / (GRASP_END - GRASP_START))
    if t > t_end - 0.9:
        return grip + (approach_gap - grip) * smoothstep((t - (t_end - 0.9)) / 0.9)
    return grip


def squeeze_depth_i(t: float, t_end: float, i: int) -> float:
    """Per-finger penetration, honouring FINGER_SCHEDULE when one is set."""
    if FINGER_SCHEDULE is None:
        return squeeze_depth(t, t_end)
    es, ee, rs, re_ = FINGER_SCHEDULE[i]
    gap, grip = -0.035, 0.0030
    if t < es:
        return gap
    if t < ee:
        return gap + (grip - gap) * smoothstep((t - es) / (ee - es))
    if t < rs:
        return grip
    if t < re_:
        return grip + (gap - grip) * smoothstep((t - rs) / (re_ - rs))
    return gap


def finger_quat(normal_world: np.ndarray) -> np.ndarray:
    """Orient the finger so its shaft (+local normal) points away from the object."""
    ref = np.array([1.0, 0.0, 0.0])
    v = np.cross(ref, normal_world)
    c = float(np.dot(ref, normal_world))
    if np.linalg.norm(v) < 1e-9:
        rot = Rot.identity() if c > 0 else Rot.from_rotvec([0, 0, np.pi])
    else:
        rot = Rot.from_rotvec(v / np.linalg.norm(v) * np.arccos(np.clip(c, -1, 1)))
    xyzw = rot.as_quat()
    return np.array([xyzw[3], xyzw[0], xyzw[1], xyzw[2]])   # MuJoCo wants wxyz


# --------------------------------------------------------------------------------------
# Simulation
# --------------------------------------------------------------------------------------

class Sim:
    def __init__(self, mode: str = "dynamic"):
        self.mode = mode
        self.model = mujoco.MjModel.from_xml_string(build_xml())
        self.data = mujoco.MjData(self.model)
        self.ref, self.segments = TASKS[TASK][0]()
        self.cube_bid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "cube")
        self.finger_bids = [
            mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, f"finger{i}")
            for i in range(N_FINGERS)
        ]
        self.finger_qadr = [
            self.model.jnt_qposadr[
                mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, f"fj{i}")]
            for i in range(N_FINGERS)
        ]
        self.finger_vadr = [
            self.model.jnt_dofadr[
                mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, f"fj{i}")]
            for i in range(N_FINGERS)
        ]
        self.cube_qadr = self.model.jnt_qposadr[
            mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, "cube_free")]
        self.cube_vadr = self.model.jnt_dofadr[
            mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, "cube_free")]
        self.tip_gids = [
            mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, f"tip{i}")
            for i in range(N_FINGERS)
        ]
        self.cube_gid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, "cube_geom")
        self.table_gid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, "table")
        self.reset()

    def reset(self):
        mujoco.mj_resetData(self.model, self.data)
        a = self.cube_qadr
        self.data.qpos[a:a + 3] = [0, 0, REST_Z]
        self.data.qpos[a + 3:a + 7] = [1, 0, 0, 0]
        for i in range(N_FINGERS):
            q = self.finger_qadr[i]
            self.data.qpos[q:q + 3] = HOME[i]
            self.data.qpos[q + 3:q + 7] = finger_quat(CONTACTS[i][1])
        mujoco.mj_forward(self.model, self.data)
        self._prev_targets = HOME.copy()
        self._int_p = np.zeros(3)
        self._int_r = np.zeros(3)
        self._alloc_cache = None
        self._alloc_step = 0

    # -- state ---------------------------------------------------------------------

    def object_pose(self):
        p = self.data.xpos[self.cube_bid].copy()
        wxyz = self.data.xquat[self.cube_bid].copy()
        return p, Rot.from_quat([wxyz[1], wxyz[2], wxyz[3], wxyz[0]])

    def object_twist(self):
        # A freejoint stores linear velocity in the world frame but angular velocity in
        # the BODY frame. Everything downstream works in world coordinates, so rotate.
        a = self.cube_vadr
        _, R = self.object_pose()
        return self.data.qvel[a:a + 3].copy(), R.apply(self.data.qvel[a + 3:a + 6])

    def finger_pose(self, i):
        p = self.data.xpos[self.finger_bids[i]].copy()
        wxyz = self.data.xquat[self.finger_bids[i]].copy()
        return p, Rot.from_quat([wxyz[1], wxyz[2], wxyz[3], wxyz[0]])

    def finger_twist(self, i):
        a = self.finger_vadr[i]
        _, R = self.finger_pose(i)
        return self.data.qvel[a:a + 3].copy(), R.apply(self.data.qvel[a + 3:a + 6])

    def pose_error(self, t):
        p_d, R_d = self.ref(t)
        p, R = self.object_pose()
        e_p = p_d - p
        e_r = (R_d * R.inv()).as_rotvec()
        return e_p, e_r

    # -- control -------------------------------------------------------------------

    def touching_table(self):
        for ci in range(self.data.ncon):
            c = self.data.contact[ci]
            if self.cube_gid in (c.geom1, c.geom2) and self.table_gid in (c.geom1, c.geom2):
                return True
        return False

    def ref_twist(self, t, h=2e-3):
        """Finite-difference the reference for feedforward velocity."""
        p1, R1 = self.ref(min(t + h, self.ref.duration))
        p0, R0 = self.ref(max(t - h, 0.0))
        dt = min(t + h, self.ref.duration) - max(t - h, 0.0)
        if dt <= 0:
            return np.zeros(3), np.zeros(3)
        return (p1 - p0) / dt, (R1 * R0.inv()).as_rotvec() / dt

    def commanded_pose(self, t):
        """Object pose the fingers are placed against: the reference, lag-clamped."""
        p, R = self.object_pose()
        if self.mode != "dynamic" or t <= GRASP_START:
            return p, R

        e_p, e_r = self.pose_error(t)
        v, w = self.object_twist()
        v_d, w_d = self.ref_twist(t)

        # Conditional integration. When the object is resting on the table, the table is
        # what sets its height, and no amount of pushing will raise it to the reference.
        # Integrating against that is a standing wind-up source: it walks the fingers
        # upward until they let go of the object entirely.
        if not self.touching_table():
            self._int_p = np.clip(self._int_p + e_p * TIMESTEP, -IMAX_POS, IMAX_POS)
        self._int_r = np.clip(self._int_r + e_r * TIMESTEP, -IMAX_ROT, IMAX_ROT)

        lin = np.clip(e_p + KI_POS * self._int_p - KD_POS * (v - v_d),
                      -MAX_LAG, MAX_LAG)
        rv = e_r + KI_ROT * self._int_r - KD_ROT * (w - w_d)
        n = np.linalg.norm(rv)
        if n > MAX_LAG_ROT:
            rv = rv * (MAX_LAG_ROT / n)
        return p + lin, Rot.from_rotvec(rv) * R

    def allocate(self, t, R_c):
        """Contact forces (on the object, world frame) realising the commanded wrench.

        Each contact force is written on the edges of its friction cone with nonnegative
        coefficients, so unilaterality and the friction limit hold by construction. A
        floor on each contact's normal force keeps the grasp loaded; the solver finds the
        internal squeeze that satisfies it without disturbing the net wrench.

        A fixed preload along each contact normal does NOT work for a minimal set. Three
        contacts on mutually orthogonal faces pushing purely inward produce a net force
        toward the shared corner that nothing can cancel, and the object gets shoved off.
        The squeeze has to be found, not imposed: those same three contacts CAN hold a
        zero-net-force internal squeeze, but only by tilting each force inside its cone.
        """
        from scipy.optimize import linprog, nnls

        e_p, e_r = self.pose_error(t)
        v, w = self.object_twist()
        v_d, w_d = self.ref_twist(t)

        h = 2e-3
        p0, _ = self.ref(max(t - h, 0.0))
        p1, _ = self.ref(t)
        p2, _ = self.ref(min(t + h, self.ref.duration))
        acc = (p2 - 2 * p1 + p0) / (h * h)
        ff_force = CUBE_MASS * (acc - np.array([0.0, 0.0, -GRAVITY]))

        w_cmd = np.concatenate([
            ff_force + WK_P * e_p + WK_D * (v_d - v),
            WK_R * e_r + WK_W * (w_d - w),
        ])

        arms = [R_c.apply(off) for off, _ in CONTACTS]
        cols, owner, ncomp = [], [], []
        for i, (_, nrm) in enumerate(CONTACTS):
            n_w = R_c.apply(nrm)
            inward = -n_w                      # force ON the object points inward
            for e in cone_basis(inward, MU_TIP):
                cols.append(np.concatenate([e, np.cross(arms[i], e)]))
                owner.append(i)
                ncomp.append(float(np.dot(e, inward)))
        A = np.array(cols).T
        nv = A.shape[1]

        # normal force at contact i must be at least PRELOAD
        A_ub = np.zeros((N_FINGERS, nv))
        for k, i in enumerate(owner):
            A_ub[i, k] = -ncomp[k]
        b_ub = -np.full(N_FINGERS, PRELOAD)

        res = linprog(np.ones(nv), A_ub=A_ub, b_ub=b_ub, A_eq=A, b_eq=w_cmd,
                      bounds=[(0.0, 40.0)] * nv, method="highs")
        if res.success:
            lam = res.x
        else:
            # Commanded wrench is outside what these contacts can produce. Fall back to
            # the closest achievable one, keeping the squeeze floor as a soft term.
            A_aug = np.vstack([A, np.sqrt(3.0) * A_ub])
            b_aug = np.concatenate([w_cmd, np.sqrt(3.0) * b_ub])
            lam, _ = nnls(A_aug, b_aug)

        out = [np.zeros(3) for _ in CONTACTS]
        for k, i in enumerate(owner):
            if lam[k] > 0:
                out[i] = out[i] + lam[k] * A[:3, k]
        return out

    def finger_targets(self, t):
        """Where each fingertip centre should be commanded this step."""
        p_c, R_c = self.commanded_pose(t)
        delta = squeeze_depth(t, self.ref.duration)

        if CONTROL == "hybrid" and t > GRASP_END and self.mode == "dynamic":
            if (self._alloc_cache is None
                    or self._alloc_step % ALLOC_EVERY == 0):
                self._alloc_cache = self.allocate(t, R_c)
            self._alloc_step += 1
            forces = self._alloc_cache
            targets, quats = [], []
            for (offset, normal), f in zip(CONTACTS, forces):
                n_w = R_c.apply(normal)
                # Tip just touching, then displaced by the force the mount must supply.
                touch = p_c + R_c.apply(offset) + n_w * TIP_RADIUS
                targets.append(touch + f / MOUNT_KP)
                quats.append(finger_quat(n_w))
            return np.array(targets), quats

        targets, quats = [], []
        for i, (offset, normal) in enumerate(CONTACTS):
            d_i = squeeze_depth_i(t, self.ref.duration, i)
            n_w = R_c.apply(normal)
            targets.append(p_c + R_c.apply(offset) + n_w * (TIP_RADIUS - d_i))
            quats.append(finger_quat(n_w))
        return np.array(targets), quats

    def fly_in(self, t, targets):
        """Blend from the scattered home poses to the grasp poses over the approach."""
        if t < GRASP_START:
            s = smoothstep(t / GRASP_START)
            targets = HOME + s * (targets - HOME)
        return self.fly_out(t, targets)

    def fly_out(self, t, targets):
        """Send a released finger back to its home pose so it is clear of the object.

        Without this a finger that has let go still hovers 35 mm off the surface, which
        is close enough for a toppling object to land on it.
        """
        if FINGER_SCHEDULE is None:
            return targets
        out = np.array(targets, dtype=float)
        for i, (_, _, _, rel_end) in enumerate(FINGER_SCHEDULE):
            if t > rel_end:
                s = smoothstep((t - rel_end) / 1.2)
                out[i] = out[i] + s * (HOME[i] - out[i])
        return out

    def apply_mount_wrenches(self, targets, quats):
        """The virtual 6-DOF spring-damper holding each finger at its setpoint."""
        self.data.xfrc_applied[:] = 0.0
        for i in range(N_FINGERS):
            p, R = self.finger_pose(i)
            v, w = self.finger_twist(i)
            wxyz = quats[i]
            R_d = Rot.from_quat([wxyz[1], wxyz[2], wxyz[3], wxyz[0]])

            f = MOUNT_KP * (targets[i] - p) - MOUNT_KD * v
            nf = np.linalg.norm(f)
            if nf > MOUNT_FMAX:
                f *= MOUNT_FMAX / nf

            tau = MOUNT_KR * (R_d * R.inv()).as_rotvec() - MOUNT_KW * w
            nt = np.linalg.norm(tau)
            if nt > MOUNT_TMAX:
                tau *= MOUNT_TMAX / nt

            self.data.xfrc_applied[self.finger_bids[i], :3] = f
            self.data.xfrc_applied[self.finger_bids[i], 3:] = tau

    def step(self):
        t = self.data.time
        targets, quats = self.finger_targets(t)
        targets = self.fly_in(t, targets)

        # Rate-limit the setpoints so a large jump cannot slam the mount to saturation.
        max_step = SLEW * TIMESTEP
        delta = targets - self._prev_targets
        norms = np.linalg.norm(delta, axis=1, keepdims=True)
        scale = np.minimum(1.0, max_step / np.maximum(norms, 1e-12))
        targets = self._prev_targets + delta * scale
        self._prev_targets = targets

        if self.mode == "kinematic":
            # Option A: write both the object and the fingers straight from the
            # reference. No dynamics, cannot fail, proves nothing.
            p_d, R_d = self.ref(t)
            xyzw = R_d.as_quat()
            a = self.cube_qadr
            self.data.qpos[a:a + 3] = p_d
            self.data.qpos[a + 3:a + 7] = [xyzw[3], xyzw[0], xyzw[1], xyzw[2]]
            for i in range(N_FINGERS):
                q = self.finger_qadr[i]
                self.data.qpos[q:q + 3] = targets[i]
                self.data.qpos[q + 3:q + 7] = quats[i]
            self.data.qvel[:] = 0.0
            self.data.time += TIMESTEP
            mujoco.mj_forward(self.model, self.data)
        else:
            self.apply_mount_wrenches(targets, quats)
            for t0, t1, w in DISTURBANCES:
                if t0 <= t < t1:
                    self.data.xfrc_applied[self.cube_bid] += np.asarray(w, float)
            mujoco.mj_step(self.model, self.data)

    # -- diagnostics ---------------------------------------------------------------

    def grip_forces(self):
        """Normal force magnitude at each fingertip/object contact, in newtons."""
        out = np.zeros(N_FINGERS)
        buf = np.zeros(6)
        for ci in range(self.data.ncon):
            con = self.data.contact[ci]
            g1, g2 = con.geom1, con.geom2
            if self.cube_gid not in (g1, g2):
                continue
            other = g2 if g1 == self.cube_gid else g1
            if other in self.tip_gids:
                mujoco.mj_contactForce(self.model, self.data, ci, buf)
                out[self.tip_gids.index(other)] += abs(buf[0])
        return out


# --------------------------------------------------------------------------------------
# Rendering
# --------------------------------------------------------------------------------------

def draw_hud(frame, t, e_pos_mm, e_rot_deg, label, grip_n, mode):
    try:
        from PIL import Image, ImageDraw, ImageFont
    except ImportError:
        return frame

    img = Image.fromarray(frame)
    d = ImageDraw.Draw(img, "RGBA")
    W, H = img.size

    def font(sz):
        for path in ("/usr/share/fonts/truetype/dejavu/DejaVuSansMono-Bold.ttf",
                     "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"):
            if os.path.exists(path):
                return ImageFont.truetype(path, sz)
        return ImageFont.load_default()

    f_big, f_mid, f_small = font(int(H * 0.030)), font(int(H * 0.024)), font(int(H * 0.019))
    pad = int(H * 0.028)

    d.rounded_rectangle([pad, pad, pad + int(W * 0.33), pad + int(H * 0.250)],
                        radius=10, fill=(12, 14, 18, 205))
    x, y = pad + int(W * 0.016), pad + int(H * 0.020)
    d.text((x, y), "FINGERTIP SWARM", font=f_big, fill=(235, 238, 245, 255))
    y += int(H * 0.042)
    tag = "closed loop on dynamic object" if mode == "dynamic" else "kinematic playback"
    d.text((x, y), tag, font=f_small, fill=(150, 158, 175, 255))
    y += int(H * 0.040)
    d.text((x, y), f"t   {t:5.2f} s", font=f_mid, fill=(200, 206, 218, 255))
    y += int(H * 0.034)
    col = (120, 220, 150, 255) if e_pos_mm < 5 else (245, 190, 90, 255)
    d.text((x, y), f"pos err  {e_pos_mm:5.2f} mm", font=f_mid, fill=col)
    y += int(H * 0.034)
    col = (120, 220, 150, 255) if e_rot_deg < 3 else (245, 190, 90, 255)
    d.text((x, y), f"rot err  {e_rot_deg:5.2f} deg", font=f_mid, fill=col)
    n_touch = int((np.asarray(grip_n) > 0.05).sum())
    y += int(H * 0.034)
    col = (245, 190, 90, 255) if n_touch <= 1 else (200, 206, 218, 255)
    d.text((x, y), f"contacts {n_touch:2d} / {len(grip_n)}", font=f_mid, fill=col)

    # Segment label, bottom centre.
    tw = d.textlength(label, font=f_mid)
    bx0 = (W - tw) / 2 - int(W * 0.015)
    d.rounded_rectangle([bx0, H - pad - int(H * 0.055), (W + tw) / 2 + int(W * 0.015),
                         H - pad], radius=8, fill=(12, 14, 18, 205))
    d.text(((W - tw) / 2, H - pad - int(H * 0.044)), label, font=f_mid,
           fill=(228, 232, 240, 255))

    # Per-finger grip force bars, bottom right.
    bw, bh, gap = int(W * 0.020), int(H * 0.105), int(W * 0.008)
    x0 = W - pad - N_FINGERS * (bw + gap)
    y0 = H - pad - bh - int(H * 0.030)
    d.text((x0, y0 - int(H * 0.030)), "grip force (N)", font=f_small,
           fill=(150, 158, 175, 255))
    fmax = 12.0
    for i, fN in enumerate(grip_n):
        bx = x0 + i * (bw + gap)
        d.rectangle([bx, y0, bx + bw, y0 + bh], fill=(30, 33, 40, 210))
        h = int(bh * min(fN / fmax, 1.0))
        if h > 0:
            d.rectangle([bx, y0 + bh - h, bx + bw, y0 + bh], fill=(96, 176, 230, 240))
        d.text((bx, y0 + bh + 2), f"{fN:.0f}", font=f_small, fill=(150, 158, 175, 255))

    return np.array(img.convert("RGB"))


def seg_label(segments, t):
    for t0, t1, name in segments:
        if t0 <= t < t1:
            return name
    return "done"


# Camera framing. The object travels a fair way, so a fixed camera leaves it small and
# wandering. A smoothed tracking camera keeps it centred and the swarm filling the frame.
CAM_DISTANCE = 0.38
CAM_ELEVATION = -16.0
CAM_AZ_START = 124.0
CAM_AZ_DRIFT = 26.0        # deg of slow orbit across the whole clip
CAM_SMOOTH = 0.045         # lookat low-pass coefficient per frame


def run(mode, out_path, width, height, fps, camera, hud=True):
    sim = Sim(mode=mode)
    renderer = mujoco.Renderer(sim.model, height, width)
    scene_opt = mujoco.MjvOption()
    scene_opt.flags[mujoco.mjtVisFlag.mjVIS_CONTACTPOINT] = False

    cam = mujoco.MjvCamera()
    cam.type = mujoco.mjtCamera.mjCAMERA_FREE
    cam.distance = CAM_DISTANCE
    cam.elevation = CAM_ELEVATION
    cam.azimuth = CAM_AZ_START
    cam.lookat[:] = [0.0, 0.0, REST_Z + 0.02]

    import imageio.v2 as imageio
    pathlib.Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    writer = imageio.get_writer(out_path, fps=fps, codec="libx264", quality=8,
                                macro_block_size=1,
                                ffmpeg_params=["-pix_fmt", "yuv420p"])

    steps_per_frame = max(1, int(round((1.0 / fps) / TIMESTEP)))
    total_frames = int(sim.ref.duration * fps)
    log = []

    for n in range(total_frames):
        for _ in range(steps_per_frame):
            sim.step()
        t = sim.data.time
        e_p, e_r = sim.pose_error(t)
        e_pos_mm = float(np.linalg.norm(e_p) * 1000.0)
        e_rot_deg = float(np.rad2deg(np.linalg.norm(e_r)))
        log.append((t, e_pos_mm, e_rot_deg, seg_label(sim.segments, t)))

        # Track the object, low-passed so the camera glides instead of jittering.
        p_obj, _ = sim.object_pose()
        aim = np.array([p_obj[0], p_obj[1], p_obj[2] + 0.015])
        cam.lookat[:] = (1 - CAM_SMOOTH) * np.array(cam.lookat) + CAM_SMOOTH * aim
        cam.azimuth = CAM_AZ_START + CAM_AZ_DRIFT * (n / max(total_frames - 1, 1))

        renderer.update_scene(sim.data, camera=cam, scene_option=scene_opt)
        frame = renderer.render()
        if hud:
            frame = draw_hud(frame, t, e_pos_mm, e_rot_deg,
                             seg_label(sim.segments, t), sim.grip_forces(), mode)
        writer.append_data(frame)

        if n % 60 == 0:
            print(f"  frame {n:4d}/{total_frames}  t={t:5.2f}s  "
                  f"pos={e_pos_mm:6.2f}mm  rot={e_rot_deg:5.2f}deg", flush=True)

    writer.close()
    renderer.close()
    return log


def report(log):
    print("\n  segment                    RMS pos    max pos    RMS rot    max rot")
    print("  " + "-" * 68)
    order, buckets = [], {}
    for t, ep, er, name in log:
        if name in ("approach + grasp", "done", "RELEASE - it topples"):
            continue
        if name not in buckets:
            buckets[name] = []
            order.append(name)
        buckets[name].append((ep, er))
    allv = []
    for name in order:
        v = np.array(buckets[name])
        allv.append(v)
        print(f"  {name:24s} {np.sqrt((v[:,0]**2).mean()):7.2f}mm {v[:,0].max():8.2f}mm"
              f" {np.sqrt((v[:,1]**2).mean()):8.2f}d {v[:,1].max():8.2f}d")
    if allv:
        v = np.vstack(allv)
        print("  " + "-" * 68)
        print(f"  {'OVERALL':24s} {np.sqrt((v[:,0]**2).mean()):7.2f}mm {v[:,0].max():8.2f}mm"
              f" {np.sqrt((v[:,1]**2).mean()):8.2f}d {v[:,1].max():8.2f}d")
    return np.vstack(allv) if allv else np.zeros((1, 2))


# --------------------------------------------------------------------------------------
# Physics sanity check
# --------------------------------------------------------------------------------------

def selftest():
    """Two-finger antipodal lift against the analytic friction bound.

    A cube held by two opposing fingertips can only be carried upward if the friction
    the normal force buys exceeds its weight:  2 * mu * f_n >= m * g.
    Sweep mu with the squeeze held fixed and check that the simulator's slip threshold
    lands where that inequality says it should. If this disagrees, the contact forces in
    the main demo are solver artifacts and nothing downstream is trustworthy.
    """
    print("\n=== physics check: two-finger antipodal lift ===")
    mg = CUBE_MASS * GRAVITY
    print(f"  object weight              {mg:.4f} N")
    print(f"  prediction: the pair holds the cube iff 2*mu*f_n >= {mg:.4f} N\n")

    # Build an explicit two-finger antipodal pair at the cube's equator. Cleaner than
    # parking unused fingers, and it isolates the quantity under test.
    mod = sys.modules[__name__]
    mod.N_FINGERS = 2
    mod.CONTACTS = [(np.array([+CUBE_HALF, 0.0, 0.0]), np.array([+1.0, 0.0, 0.0])),
                    (np.array([-CUBE_HALF, 0.0, 0.0]), np.array([-1.0, 0.0, 0.0]))]
    mod.HOME = np.array([[+0.25, 0.0, REST_Z], [-0.25, 0.0, REST_Z]])
    mod.FINGER_COLORS = list(BASE_COLORS[:2])

    active = [0, 1]
    squeeze = 0.0030
    lift_h = 0.08
    T_close, T_lift, T_hold = 1.2, 2.0, 0.6

    rows = []
    for mu in [0.05, 0.10, 0.12, 0.13, 0.14, 0.16, 0.20, 0.50, 1.20]:
        xml = build_xml().replace(f'friction="{MU_TIP} 0.02 0.002"',
                                  f'friction="{mu} 0.02 0.002"')
        model = mujoco.MjModel.from_xml_string(xml)
        data = mujoco.MjData(model)

        cube_bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "cube")
        cube_gid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "cube_geom")
        fb = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, f"finger{i}")
              for i in range(N_FINGERS)]
        qa = [model.jnt_qposadr[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, f"fj{i}")]
              for i in range(N_FINGERS)]
        va = [model.jnt_dofadr[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, f"fj{i}")]
              for i in range(N_FINGERS)]
        tip_gids = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, f"tip{i}")
                    for i in range(N_FINGERS)]

        mujoco.mj_resetData(model, data)
        data.qpos[0:3] = [0, 0, REST_Z]
        data.qpos[3:7] = [1, 0, 0, 0]
        park = {}
        for i in range(N_FINGERS):
            data.qpos[qa[i]:qa[i] + 3] = HOME[i]
            data.qpos[qa[i] + 3:qa[i] + 7] = finger_quat(CONTACTS[i][1])
            park[i] = HOME[i]
        mujoco.mj_forward(model, data)

        nsteps = int((T_close + T_lift + T_hold) / TIMESTEP)
        for n in range(nsteps):
            t = n * TIMESTEP
            if t < T_close:
                pen = -0.02 + (squeeze + 0.02) * smoothstep(t / T_close)
                z = REST_Z
            else:
                pen = squeeze
                z = REST_Z + lift_h * smoothstep(min((t - T_close) / T_lift, 1.0))

            data.xfrc_applied[:] = 0.0
            for i in range(N_FINGERS):
                if i in active:
                    off, nrm = CONTACTS[i]
                    tgt = np.array([0, 0, z]) + off + nrm * (TIP_RADIUS - pen)
                else:
                    tgt = park[i]
                p = data.xpos[fb[i]]
                v = data.qvel[va[i]:va[i] + 3]
                f = MOUNT_KP * (tgt - p) - MOUNT_KD * v
                nf = np.linalg.norm(f)
                if nf > MOUNT_FMAX:
                    f = f * (MOUNT_FMAX / nf)
                data.xfrc_applied[fb[i], :3] = f
                # Full orientation PD, as in the main loop. Damping alone is not enough:
                # torsional contact friction acting on a 5e-5 kg m^2 body spins it up to
                # divergence unless there is a restoring term.
                wq = data.xquat[fb[i]]
                R_now = Rot.from_quat([wq[1], wq[2], wq[3], wq[0]])
                wq_d = finger_quat(CONTACTS[i][1])
                R_des = Rot.from_quat([wq_d[1], wq_d[2], wq_d[3], wq_d[0]])
                w_world = R_now.apply(data.qvel[va[i] + 3:va[i] + 6])
                tau = MOUNT_KR * (R_des * R_now.inv()).as_rotvec() - MOUNT_KW * w_world
                nt = np.linalg.norm(tau)
                if nt > MOUNT_TMAX:
                    tau = tau * (MOUNT_TMAX / nt)
                data.xfrc_applied[fb[i], 3:] = tau
            mujoco.mj_step(model, data)

        buf = np.zeros(6)
        fn = 0.0
        for ci in range(data.ncon):
            con = data.contact[ci]
            if cube_gid in (con.geom1, con.geom2):
                other = con.geom2 if con.geom1 == cube_gid else con.geom1
                if other in tip_gids:
                    mujoco.mj_contactForce(model, data, ci, buf)
                    fn += abs(buf[0])
        fn_per = fn / len(active) if fn > 0 else 0.0
        commanded_z = REST_Z + lift_h
        slip = commanded_z - data.xpos[cube_bid][2]
        held = slip < 0.010
        capacity = 2.0 * mu * fn_per
        predicted = capacity >= mg
        ok = (held == predicted)
        rows.append((mu, fn_per, capacity, slip, held, predicted, ok))
        print(f"  mu={mu:5.2f}  f_n={fn_per:6.2f} N  2*mu*f_n={capacity:7.4f} N  "
              f"slip={slip*1000:7.1f} mm  held={str(held):5s}  "
              f"predicted={str(predicted):5s}  {'ok' if ok else 'MISMATCH'}")

    # A case that drops the cube reports f_n = 0 because contact is already lost, so
    # comparing each row against its own measured f_n is partly circular. The real test
    # is where the slip threshold sits. Take f_n from the cases that held (it is set by
    # the mount/contact series stiffness and is mu-independent) and check that the
    # measured threshold brackets the predicted one.
    set_finger_count(6)        # restore module state for any later call

    held_fn = [r[1] for r in rows if r[4]]
    if not held_fn:
        print("\n  no case held the cube; the test is inconclusive.")
        return False
    fn = float(np.mean(held_fn))
    mu_star = mg / (2.0 * fn)

    mus = [r[0] for r in rows]
    last_drop = max((m for m, r in zip(mus, rows) if not r[4]), default=None)
    first_hold = min((m for m, r in zip(mus, rows) if r[4]), default=None)

    print(f"\n  normal force while held      f_n = {fn:.3f} N per finger")
    print(f"  predicted slip threshold     mu* = mg/(2*f_n) = {mu_star:.4f}")
    print(f"  measured slip threshold      between mu={last_drop} and mu={first_hold}")

    monotonic = all(
        rows[i][4] <= rows[i + 1][4] for i in range(len(rows) - 1))
    bracketed = (last_drop is not None and first_hold is not None
                 and last_drop <= mu_star <= first_hold)

    print(f"  threshold bracketed          {bracketed}")
    print(f"  hold/drop monotonic in mu    {monotonic}")
    ok = bracketed and monotonic
    print("\n  " + ("PASS: the simulated slip threshold matches Coulomb's law, so the "
                    "demo's\n        grasp forces are real contact physics, not solver "
                    "artifacts."
                    if ok else
                    "FAIL: the simulated threshold does not match the analytic bound. "
                    "Do not\n        trust the grasp forces in the demo."))
    return ok


# --------------------------------------------------------------------------------------

def sweep():
    """Tracking versus swarm size, at the fixed contact-placement heuristic.

    This is NOT the minimisation you eventually want. It runs the same trajectory for
    each n and reports what the tracking looks like, so you can see where the placement
    heuristic stops producing a workable grasp. A real answer needs an optimiser over
    contact positions and a wrench-feasibility test, not a sweep over a fixed layout.
    """
    handoff = (TASK == "handoff")
    if handoff:
        sizes = [3, 4, 5, 6, 8, 10]
        print("\n=== handoff: placement swarm size (+1 vertex finger) ===")
        print("  place  total   place RMS   place max   hold max tilt   outcome")
        print("  " + "-" * 66)
    else:
        sizes = [3, 6, 9, 12] if LAYOUT == "corner" else [3, 4, 5, 6, 8, 10]
        print("\n=== tracking versus swarm size (fixed placement heuristic) ===")
        print("  n   RMS pos    max pos    RMS rot    max rot   outcome")
        print("  " + "-" * 62)

    diag = np.array([1.0, 1.0, 1.0]) / np.sqrt(3.0)
    results = []
    for n in sizes:
        if handoff:
            setup_handoff(n)
        else:
            set_finger_count(n)
        sim = Sim("dynamic")
        errs, hold_tilt, diverged = [], [], False
        while sim.data.time < sim.ref.duration:
            sim.step()
            t = sim.data.time
            lbl = seg_label(sim.segments, t)
            if not np.all(np.isfinite(sim.data.qpos)):
                diverged = True
                break
            # The release phase is a deliberate topple, not a tracking failure.
            if "topple" in lbl or lbl in ("approach + grasp", "done"):
                continue
            e_p, e_r = sim.pose_error(t)
            errs.append((np.linalg.norm(e_p) * 1000, np.rad2deg(np.linalg.norm(e_r))))
            if lbl == "ONE FINGER HOLDING":
                _, R = sim.object_pose()
                hold_tilt.append(
                    np.rad2deg(np.arccos(np.clip(R.apply(diag)[2], -1.0, 1.0))))

        v = np.array(errs) if errs else np.zeros((1, 2))
        lost = diverged or v[:, 0].max() > 25.0
        if handoff:
            ht = max(hold_tilt) if hold_tilt else float("nan")
            lost = lost or not (ht < 3.0)
            print(f"  {n:5d}  {n+1:5d}   {np.sqrt((v[:,0]**2).mean()):8.2f}mm "
                  f"{v[:,0].max():9.2f}mm   {ht:10.2f}deg   "
                  f"{'LOST IT' if lost else 'held'}")
        else:
            print(f"  {n:2d}  {np.sqrt((v[:,0]**2).mean()):7.2f}mm {v[:,0].max():8.2f}mm"
                  f" {np.sqrt((v[:,1]**2).mean()):8.2f}d {v[:,1].max():8.2f}d   "
                  f"{'LOST THE OBJECT' if lost else 'held'}")
        results.append((n, lost))

    if handoff:
        setup_handoff(6)
    else:
        set_finger_count(6)
    held = [n for n, lost in results if not lost]
    print(f"\n  smallest swarm that worked with this layout: n = {min(held)}"
          if held else "\n  no swarm size worked.")
    print("  Caveat: that is the smallest n the HEURISTIC happens to work at, not the\n"
          "  minimum number of fingers the task requires.")
    return bool(held)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["dynamic", "kinematic"], default="dynamic")
    ap.add_argument("--fingers", type=int, default=6,
                    help="swarm size (3-12), spread around the cube's equator. For "
                         "--task handoff this is the PLACEMENT swarm; the vertex finger "
                         "is always added on top. Does NOT optimise placement.")
    ap.add_argument("--sweep", action="store_true",
                    help="run the finger-count sweep instead of rendering")
    ap.add_argument("--task", choices=sorted(TASKS), default="demo",
                    help="which reference trajectory to track")
    ap.add_argument("--control", choices=["position", "hybrid"], default="position",
                    help="hybrid adds friction-cone force allocation; needed for "
                         "minimal contact sets")
    ap.add_argument("--contacts", default=None,
                    help="JSON contact set from min_fingers.py; overrides --fingers")
    ap.add_argument("--out", default="out/swarm_demo.mp4")
    ap.add_argument("--width", type=int, default=1280)
    ap.add_argument("--height", type=int, default=720)
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--camera", default="hero")
    ap.add_argument("--no-hud", action="store_true")
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()

    if args.selftest:
        sys.exit(0 if selftest() else 1)

    if args.task == "handoff":
        globals()["HANDOFF_PLACERS"] = args.fingers
    set_task(args.task)
    globals()["CONTROL"] = args.control

    if args.sweep:
        sys.exit(0 if sweep() else 1)

    if args.task == "handoff":
        pass            # setup_handoff already ran, sized by --fingers
    elif args.contacts:
        import json
        with open(args.contacts) as fh:
            spec = json.load(fh)
        set_finger_count(0, contacts=[(c["offset"], c["normal"])
                                      for c in spec["contacts"]])
        print(f"loaded {N_FINGERS} optimised contacts from {args.contacts}"
              f"  (task={spec.get('task', '?')})")
    else:
        set_finger_count(args.fingers)

    print(f"task={args.task}  mode={args.mode}  control={args.control}  "
          f"fingers={N_FINGERS}  out={args.out}  {args.width}x{args.height}@{args.fps}")
    log = run(args.mode, args.out, args.width, args.height, args.fps,
              args.camera, hud=not args.no_hud)
    report(log)
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
