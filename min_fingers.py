"""
Minimise the number of fingertips needed to drive the object along a trajectory.

This is the analysis layer, and it runs without a simulator. Given the object's mass
properties and a prescribed trajectory, Newton-Euler gives the exact wrench the contacts
must supply at every instant. The question is then: what is the smallest set of contact
points that can produce that wrench at every instant, with every contact pushing (never
pulling) and every contact force inside its friction cone?

Formulation
-----------
Each candidate contact i contributes a force f_i = sum_j lambda_ij * e_ij, where the
e_ij are the edges of a polyhedral cone inscribed in the Coulomb cone and lambda_ij >= 0.
Writing it on the cone edges makes unilaterality and the friction limit automatic, and
keeps the whole thing linear.

At each sampled time t the contacts must balance the required wrench:

    sum_i sum_j lambda_ij(t) * [ e_ij ; r_i(t) x e_ij ]  =  w(t)

One extra variable s_i per contact upper-bounds that contact's normal force across all
sampled times. Minimising sum_i s_i is the convex (L1) surrogate for "use few contacts";
iteratively reweighting by 1/(s_i + eps) drives the surrogate toward true cardinality.
Contacts whose s_i collapses to ~0 are not needed and get dropped.

The cone is a polyhedron inscribed in the true cone, so a set this says is feasible is
genuinely feasible. It is conservative, never optimistic.

KNOWN TO BE WRONG FOR THE CORNER BALANCE
----------------------------------------
The disturbance margin below is an isotropic ball of wrenches at the centre of mass:
plus and minus a fixed force on each axis and a fixed torque about each axis. That is a
reasonable model for a grasp being jostled in free space. It is the WRONG model for an
object balanced on a support point, and it fails in both directions:

  * With the table credited in the disturbance conditions, the bar is far too low. It
    returned two face contacts for the corner balance; the simulator jammed the cube
    against the table with one finger at a 14 degree lean and never balanced anything.
  * With the table excluded, the bar is too high. It demands the fingers reject a pull
    upward, which nothing in the task applies, and returns 3 to 4 contacts.

Meanwhile a single fingertip on the vertex OPPOSITE the support vertex holds the cube
at 0.00 degrees of tilt against tipping torques up to 20 mN m, verified in simulation,
while the same cube with no finger topples at 5 mN m. One finger is the right answer and
this tool reports three.

The criterion an unstable equilibrium needs is narrow and specific: reject tipping
torque about the SUPPORT POINT, with the disturbance set chosen from the physics of the
task. That is a modelling judgement, not something the LP supplies. For balance tasks,
trust the simulator over this file. This tool is sound for grasps where the disturbance
really is roughly isotropic and the object is not resting on anything.

What this does NOT do
---------------------
It is a quasi-static feasibility argument at sampled instants. It says a contact set
*can* produce the required wrench; it does not say a position-controlled finger will
*achieve* it, and it ignores how fingers get to their contacts, whether they collide, and
whether the grasp is robust to disturbance. That is what running it back through the
simulator is for, and `--validate` does exactly that.

Usage:
    python3 min_fingers.py --task corner --phase "BALANCE ON CORNER" --validate
    python3 min_fingers.py --task demo --out contacts_demo.json --validate
"""

from __future__ import annotations

import argparse
import json

import numpy as np
from scipy.optimize import linprog
from scipy.spatial.transform import Rotation as Rot

import swarm_sim as S

CONE_EDGES = 8          # polyhedral approximation of the friction cone
ZERO_TOL = 1e-4         # N, below this a contact is treated as unused


# --------------------------------------------------------------------------------------
# Candidate contacts
# --------------------------------------------------------------------------------------

def candidate_contacts(faces, per_face=9, include_edges=True, include_vertices=True):
    """Candidate contact points: face grids, plus edge midpoints and vertices.

    Vertices and edges matter enormously and were missing from the first version, which
    only gridded face planes. On a cube balanced on one vertex, the single best contact
    in the whole set is the OPPOSITE vertex: it sits on the body diagonal, so its lever
    arm about the support point is the full diagonal (104 mm for a 60 mm cube) and it is
    oriented purely vertically, meaning a lateral force there produces torque purely
    about the horizontal axes that actually tip the object. A face-centre contact has a
    73 mm arm pointing the wrong way. Searching only face planes excluded the answer, so
    the "minimum" it returned was the minimum over the wrong set.

    A vertex or edge contact takes the outward bisector of its adjoining faces as its
    normal, which is the direction a fingertip would approach from.
    """
    a = S.CUBE_HALF
    margin = a - S.TIP_RADIUS - 0.003
    k = int(round(np.sqrt(per_face)))
    out = []

    face_keys = [tuple(np.round(np.asarray(n, float), 6)) for n in faces]

    for nrm in faces:
        nrm = np.asarray(nrm, float)
        u = np.array([nrm[2], nrm[0], nrm[1]])
        u = u - nrm * np.dot(u, nrm)
        u /= np.linalg.norm(u)
        v = np.cross(nrm, u)
        for iu in range(k):
            for iv in range(k):
                su = -margin + 2 * margin * (iu / max(k - 1, 1))
                sv = -margin + 2 * margin * (iv / max(k - 1, 1))
                out.append((nrm * a + u * su + v * sv, nrm.copy()))

    def allowed(axes):
        """Only offer a vertex/edge whose every adjoining face is reachable."""
        return all(tuple(np.round(ax, 6)) in face_keys for ax in axes)

    if include_vertices:
        for sx in (-1, 1):
            for sy in (-1, 1):
                for sz in (-1, 1):
                    axes = [np.array([sx, 0.0, 0.0]), np.array([0.0, sy, 0.0]),
                            np.array([0.0, 0.0, sz])]
                    if not allowed(axes):
                        continue
                    n = np.array([sx, sy, sz], float) / np.sqrt(3.0)
                    out.append((np.array([sx, sy, sz], float) * a, n))

    if include_edges:
        for ax in range(3):
            for s1 in (-1, 1):
                for s2 in (-1, 1):
                    off = np.zeros(3)
                    nrm = np.zeros(3)
                    others = [i for i in range(3) if i != ax]
                    off[others[0]] = s1 * a
                    off[others[1]] = s2 * a
                    e1 = np.zeros(3); e1[others[0]] = s1
                    e2 = np.zeros(3); e2[others[1]] = s2
                    if not allowed([e1, e2]):
                        continue
                    nrm = (e1 + e2) / np.sqrt(2.0)
                    out.append((off.copy(), nrm))
    return out


ALL_FACES = [np.array(v, float) for v in
             ([1, 0, 0], [-1, 0, 0], [0, 1, 0], [0, -1, 0], [0, 0, 1], [0, 0, -1])]


def cone_edges(normal, mu, n_edge=CONE_EDGES):
    """Edges of a polyhedral cone inscribed in the Coulomb cone about `normal`."""
    n = np.asarray(normal, float)
    n = n / np.linalg.norm(n)
    tmp = np.array([1.0, 0.0, 0.0]) if abs(n[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
    u = np.cross(n, tmp)
    u /= np.linalg.norm(u)
    v = np.cross(n, u)
    return [n + mu * (np.cos(2 * np.pi * j / n_edge) * u
                      + np.sin(2 * np.pi * j / n_edge) * v)
            for j in range(n_edge)]


# --------------------------------------------------------------------------------------
# Required wrench from the reference trajectory
# --------------------------------------------------------------------------------------

def cube_inertia():
    """Inertia tensor of a uniform cube about its centre, body frame."""
    e = 2.0 * S.CUBE_HALF
    return np.eye(3) * (S.CUBE_MASS * e * e / 6.0)


def required_wrench(ref, t, h=5e-3):
    """Net force and torque the contacts must supply at time t, world frame.

    Newton-Euler on the reference: the contacts must account for the acceleration the
    trajectory demands plus the weight the trajectory does not carry by itself.
    """
    p0, R0 = ref(max(t - h, 0.0))
    p1, R1 = ref(t)
    p2, R2 = ref(min(t + h, ref.duration))
    dt = h

    acc = (p2 - 2 * p1 + p0) / (dt * dt)
    w1 = (R2 * R1.inv()).as_rotvec() / dt
    w0 = (R1 * R0.inv()).as_rotvec() / dt
    alpha = (w1 - w0) / dt
    w = 0.5 * (w0 + w1)

    g = np.array([0.0, 0.0, -S.GRAVITY])
    force = S.CUBE_MASS * (acc - g)

    I_world = R1.as_matrix() @ cube_inertia() @ R1.as_matrix().T
    torque = I_world @ alpha + np.cross(w, I_world @ w)
    return np.concatenate([force, torque]), p1, R1


def table_contact(p, R):
    """The vertex/face contact with the table, if the object is resting on it.

    The table carries load the fingers then do not have to. Leaving it out would make the
    analysis ask the fingers to support the full weight and overstate the answer.
    """
    corners = np.array([[sx, sy, sz] for sx in (-1, 1) for sy in (-1, 1)
                        for sz in (-1, 1)]) * S.CUBE_HALF
    world = np.array([p + R.apply(c) for c in corners])
    zmin = world[:, 2].min()
    if zmin > 1.5e-3:
        return []                      # airborne, no table contact
    touching = world[world[:, 2] < zmin + 1e-3]
    return [(pt - p, np.array([0.0, 0.0, 1.0])) for pt in touching]


# --------------------------------------------------------------------------------------
# The LP
# --------------------------------------------------------------------------------------

def build_conditions(ref, times, margin_f, margin_m):
    """Every wrench the contact set must be able to supply.

    For each sampled time this is the nominal Newton-Euler wrench, plus that wrench
    perturbed along each of the six wrench axes in both directions. Requiring the
    perturbed cases is what makes the answer meaningful at an unstable equilibrium: a
    cube balanced on its vertex has its centre of mass exactly over the contact point, so
    the nominal required wrench is zero and pure feasibility would say no fingers are
    needed. What the fingers are actually for is rejecting the disturbance that tips it.
    """
    disturbances = [np.zeros(6)]
    for ax in range(6):
        mag = margin_f if ax < 3 else margin_m
        for sgn in (+1.0, -1.0):
            d = np.zeros(6)
            d[ax] = sgn * mag
            disturbances.append(d)

    conds = []
    for t in times:
        w_req, p, R = required_wrench(ref, t)
        tab = table_contact(p, R)
        for j, d in enumerate(disturbances):
            # The table carries the NOMINAL load, so it belongs in that condition. It
            # must NOT be credited with rejecting the disturbance: its force acts through
            # the support point, and a force through the support point cannot restore
            # attitude about it. In a centre-of-mass wrench balance that force does show
            # a nonzero moment, which is how the first version talked itself into
            # believing two face contacts could hold a corner balance.
            conds.append((w_req + d, p, R, tab if j == 0 else []))
    return conds


def solve(cands, conds, mu_tip, mu_table, weights, fmax=25.0):
    """One reweighted-L1 pass over all conditions. Returns per-contact force caps."""
    nc = len(cands)
    ne = CONE_EDGES
    edges = [cone_edges(n, mu_tip) for _, n in cands]

    nK = len(conds)
    n_lam = nc * ne * nK
    n_s = nc
    n_fixed = n_lam + n_s

    # Table lambdas get their own block per condition. Size them all up front so the
    # column layout is known before any row is written.
    tab_edges = [[cone_edges(n, mu_table) for _, n in tab] for _, _, _, tab in conds]
    tab_count = [sum(len(e) for e in te) for te in tab_edges]
    tab_start, acc = [], n_fixed
    for c in tab_count:
        tab_start.append(acc)
        acc += c
    width = acc

    A_eq = np.zeros((6 * nK, width))
    b_eq = np.zeros(6 * nK)
    A_ub = np.zeros((nc * nK, width))

    for ki, (w_req, p, R, tab) in enumerate(conds):
        r0 = 6 * ki
        b_eq[r0:r0 + 6] = w_req
        for i, (off, nrm) in enumerate(cands):
            r = R.apply(off)
            base = ki * nc * ne + i * ne
            for j, e in enumerate(edges[i]):
                A_eq[r0:r0 + 3, base + j] = e
                A_eq[r0 + 3:r0 + 6, base + j] = np.cross(r, e)
                # normal force at this contact and condition, capped by s_i
                A_ub[ki * nc + i, base + j] = float(np.dot(e, nrm))
            A_ub[ki * nc + i, n_lam + i] = -1.0

        col = tab_start[ki]
        for (off, _), ee in zip(tab, tab_edges[ki]):
            r = R.apply(off)
            for e in ee:
                A_eq[r0:r0 + 3, col] = e
                A_eq[r0 + 3:r0 + 6, col] = np.cross(r, e)
                col += 1

    c = np.zeros(width)
    c[n_lam:n_lam + nc] = weights

    bounds = [(0.0, None)] * width
    for i in range(nc):
        bounds[n_lam + i] = (0.0, fmax)

    res = linprog(c, A_ub=A_ub, b_ub=np.zeros(nc * nK),
                  A_eq=A_eq, b_eq=b_eq, bounds=bounds, method="highs")
    if not res.success:
        return None
    return res.x[n_lam:n_lam + nc]


def minimise(cands, conds, mu_tip, mu_table, iters=8):
    """Reweighted L1 until the active set stops shrinking."""
    nc = len(cands)
    weights = np.ones(nc)
    active = list(range(nc))
    for it in range(iters):
        s_sub = solve([cands[i] for i in active], conds, mu_tip, mu_table,
                      weights[active])
        if s_sub is None:
            return None
        s = np.zeros(nc)
        s[active] = s_sub
        keep = [i for i in active if s[i] > ZERO_TOL]
        if not keep:
            return None
        shrunk = len(keep) < len(active)
        active = keep
        weights = np.ones(nc)
        weights[active] = 1.0 / (s[active] + 1e-3)
        if not shrunk and it > 1:
            break
    return active


def verify(cands, idx, conds, mu_tip, mu_table):
    """Is this exact subset feasible under every condition, on its own?"""
    sub = [cands[i] for i in idx]
    return solve(sub, conds, mu_tip, mu_table, np.ones(len(sub))) is not None


def prune(cands, active, conds, mu_tip, mu_table):
    """Backward elimination: drop any contact the set can still do without."""
    active = list(active)
    changed = True
    while changed and len(active) > 1:
        changed = False
        for i in sorted(active, key=lambda k: -k):
            trial = [a for a in active if a != i]
            if verify(cands, trial, conds, mu_tip, mu_table):
                active = trial
                changed = True
                break
    return active


# --------------------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", choices=sorted(S.TASKS), default="corner")
    ap.add_argument("--phase", default=None,
                    help="restrict to one trajectory segment, by its label")
    ap.add_argument("--samples", type=int, default=40)
    ap.add_argument("--per-face", type=int, default=9)
    ap.add_argument("--no-edges", action="store_true")
    ap.add_argument("--no-vertices", action="store_true")
    ap.add_argument("--mu", type=float, default=S.MU_TIP)
    ap.add_argument("--margin-force", type=float, default=0.5,
                    help="N of disturbance force the grasp must reject in any direction")
    ap.add_argument("--margin-torque", type=float, default=0.010,
                    help="N m of disturbance torque the grasp must reject, any axis")
    ap.add_argument("--out", default=None)
    ap.add_argument("--validate", action="store_true",
                    help="run the resulting set through the simulator")
    args = ap.parse_args()

    S.set_task(args.task)
    ref, segments = S.TASKS[args.task][0]()

    if args.phase:
        match = [s for s in segments if s[2] == args.phase]
        if not match:
            raise SystemExit(f"no segment {args.phase!r}; have "
                             f"{[s[2] for s in segments]}")
        t0, t1 = match[0][0], match[0][1]
    else:
        t0, t1 = S.GRASP_END, ref.duration - 1.0
    times = np.linspace(t0 + 1e-3, t1 - 1e-3, args.samples)

    # For the corner task only the three upward faces are reachable: a finger on a
    # downward face sits under the cube with its shaft in the table.
    faces = (S.corner_upper_faces() if args.task == "corner" else ALL_FACES)
    cands = candidate_contacts(faces, args.per_face,
                               include_edges=not args.no_edges,
                               include_vertices=not args.no_vertices)

    print(f"task={args.task}  window=[{t0:.2f}, {t1:.2f}] s  "
          f"samples={len(times)}  candidates={len(cands)}  mu={args.mu}")
    print(f"faces available: {len(faces)}")

    wmax = max(np.linalg.norm(required_wrench(ref, t)[0][:3]) for t in times)
    tmax = max(np.linalg.norm(required_wrench(ref, t)[0][3:]) for t in times)
    print(f"peak required force {wmax:.3f} N, peak required torque {tmax:.5f} N m\n")

    conds = build_conditions(ref, times, args.margin_force, args.margin_torque)
    print(f"disturbance margin: {args.margin_force} N force, "
          f"{args.margin_torque} N m torque")
    print(f"wrench conditions to satisfy: {len(conds)}\n")

    active = minimise(cands, conds, args.mu, S.MU_TABLE)
    if active is None:
        raise SystemExit("infeasible: no subset of the candidate set can produce the "
                         "required wrench. Widen the candidate set or raise mu.")
    print(f"reweighted L1 kept {len(active)} contacts")

    active = prune(cands, active, conds, args.mu, S.MU_TABLE)
    print(f"after backward elimination: {len(active)} contacts\n")

    for rank, i in enumerate(active):
        off, nrm = cands[i]
        print(f"  {rank}: offset ({off[0]*1000:+6.1f}, {off[1]*1000:+6.1f}, "
              f"{off[2]*1000:+6.1f}) mm   normal ({nrm[0]:+.0f}, {nrm[1]:+.0f}, "
              f"{nrm[2]:+.0f})")

    spec = {
        "task": args.task,
        "phase": args.phase,
        "mu": args.mu,
        "margin_force": args.margin_force,
        "margin_torque": args.margin_torque,
        "n_fingers": len(active),
        "contacts": [{"offset": list(map(float, cands[i][0])),
                      "normal": list(map(float, cands[i][1]))} for i in active],
    }
    out = args.out or f"contacts_{args.task}{'_' + args.phase.replace(' ', '_') if args.phase else ''}.json"
    with open(out, "w") as fh:
        json.dump(spec, fh, indent=2)
    print(f"\nwrote {out}")

    print("\nThis is a feasibility result: the set CAN produce the required wrench at "
          "every\nsampled instant. Whether a position-controlled finger actually "
          "achieves it is a\ndifferent question, which is what --validate checks.")

    if args.validate:
        print("\n=== simulator validation ===")
        S.set_finger_count(0, contacts=[(c["offset"], c["normal"])
                                        for c in spec["contacts"]])
        sim = S.Sim("dynamic")
        log = []
        while sim.data.time < sim.ref.duration:
            sim.step()
            t = sim.data.time
            e_p, e_r = sim.pose_error(t)
            log.append((t, np.linalg.norm(e_p) * 1000,
                        np.rad2deg(np.linalg.norm(e_r)), S.seg_label(sim.segments, t)))
        S.report(log)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
