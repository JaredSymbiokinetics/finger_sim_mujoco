"""
Objects the swarm can manipulate, and how to find contact points on them.

Adding a shape means supplying three things:

  xml(prefix, cx, cy)   the MJCF body, with a freejoint named <prefix>_objfree, a body
                        named <prefix>_obj and a collision geom named <prefix>_objgeom
  surface_point(dir)    ray-cast from the object's centre along `dir`, returning
                        (offset, outward normal) in the body frame
  rest_z                height of the centre when the object sits on the table

`surface_point` is the only interesting one. It is the map the placement optimiser
searches over, so it has to be continuous and cover the whole boundary INCLUDING edges
and vertices. On a cube, the single most useful contact for balancing on a corner is the
opposite vertex, and a parameterisation built from face patches cannot express it. Every
shape here returns the true outward normal, which at an edge or vertex is the bisector of
the adjoining faces.

For a mesh the normal is estimated numerically from nearby ray hits, because MuJoCo's
`mj_ray` returns a distance and a geom id but not a surface normal. That estimate is
taken on the CONVEX HULL, which is what MuJoCo collides against by default anyway, so it
is consistent with the physics rather than with the original CAD.

Usage from the CLI:
    --object cube | box | sphere | cylinder | capsule | ellipsoid | tee
    --mesh path/to/part.stl --mesh-scale 0.001      (CAD in mm -> metres)
"""

from __future__ import annotations

import numpy as np

CUBE_HALF = 0.030
DEFAULT_MASS = 0.150


# --------------------------------------------------------------------------------------
# Analytic surface maps
# --------------------------------------------------------------------------------------

def _unit(d):
    d = np.asarray(d, float)
    n = np.linalg.norm(d)
    return np.array([1.0, 0.0, 0.0]) if n < 1e-9 else d / n


def box_surface(half):
    half = np.asarray(half, float)

    def f(direction):
        d = _unit(direction)
        t = np.min(half / np.maximum(np.abs(d), 1e-12))
        p = t * d
        nrm = np.zeros(3)
        for i in range(3):
            if abs(abs(p[i]) - half[i]) < 1e-6:
                nrm[i] = np.sign(p[i])
        if np.linalg.norm(nrm) < 1e-9:
            j = int(np.argmax(np.abs(p) / half))
            nrm[j] = np.sign(p[j]) or 1.0
        return p, nrm / np.linalg.norm(nrm)
    return f


def sphere_surface(r):
    def f(direction):
        d = _unit(direction)
        return r * d, d.copy()
    return f


def ellipsoid_surface(radii):
    a = np.asarray(radii, float)

    def f(direction):
        d = _unit(direction)
        t = 1.0 / np.sqrt(np.sum((d / a) ** 2))
        p = t * d
        n = p / (a ** 2)                       # gradient of (x/a)^2 sum
        return p, n / np.linalg.norm(n)
    return f


def cylinder_surface(r, hz):
    """Z-aligned cylinder of radius r and half-height hz, including the rim edges."""
    def f(direction):
        d = _unit(direction)
        rad = np.hypot(d[0], d[1])
        t_side = r / rad if rad > 1e-12 else np.inf
        t_cap = hz / abs(d[2]) if abs(d[2]) > 1e-12 else np.inf
        t = min(t_side, t_cap)
        p = t * d
        on_side = abs(np.hypot(p[0], p[1]) - r) < 1e-6
        on_cap = abs(abs(p[2]) - hz) < 1e-6
        n = np.zeros(3)
        if on_side:
            n += np.array([p[0], p[1], 0.0]) / max(np.hypot(p[0], p[1]), 1e-12)
        if on_cap:
            n += np.array([0.0, 0.0, np.sign(p[2])])
        if np.linalg.norm(n) < 1e-9:
            n = np.array([0.0, 0.0, 1.0])
        return p, n / np.linalg.norm(n)
    return f


def mesh_surface(model, geom_id, body_id):
    """Ray-cast onto a mesh geom in a live MuJoCo model; normal by finite difference.

    mj_ray gives a hit distance, not a normal, so the normal is estimated from the local
    shape of the hit-distance field using two tangential probes. This runs against the
    CONVEX HULL, which is what MuJoCo collides with by default, so the placement the
    optimiser finds is consistent with the contact the simulator will actually produce.
    """
    import mujoco

    def f(direction):
        d = _unit(direction)
        geomgroup = np.ones(6, dtype=np.uint8)
        gid = np.zeros(1, dtype=np.int32)

        def hit(vec):
            v = _unit(vec)
            origin = np.zeros(3)
            dist = mujoco.mj_ray(model, _DATA[0], origin, v, geomgroup, 1,
                                 body_id, gid)
            return None if dist < 0 else dist * v

        p = hit(d)
        if p is None:
            return 0.03 * d, d.copy()
        tmp = np.array([0.0, 0.0, 1.0]) if abs(d[2]) < 0.9 else np.array([1.0, 0.0, 0.0])
        u = np.cross(d, tmp)
        u /= np.linalg.norm(u)
        v = np.cross(d, u)
        eps = 0.06
        pa, pb = hit(d + eps * u), hit(d + eps * v)
        if pa is None or pb is None:
            return p, d.copy()
        n = np.cross(pa - p, pb - p)
        if np.linalg.norm(n) < 1e-12:
            return p, d.copy()
        n = n / np.linalg.norm(n)
        if np.dot(n, d) < 0:
            n = -n
        return p, n
    return f


_DATA = [None]          # set by build_mesh_probe


# --------------------------------------------------------------------------------------
# Object specs
# --------------------------------------------------------------------------------------

def _stickers(prefix, half):
    a = float(np.min(half))
    s = a * 0.78
    ins = [h + 0.0009 for h in half]
    faces = [("px", f"{ins[0]} 0 0", f"0.001 {s} {s}", "0.85 0.16 0.14 1"),
             ("nx", f"-{ins[0]} 0 0", f"0.001 {s} {s}", "0.95 0.48 0.09 1"),
             ("py", f"0 {ins[1]} 0", f"{s} 0.001 {s}", "0.10 0.38 0.75 1"),
             ("ny", f"0 -{ins[1]} 0", f"{s} 0.001 {s}", "0.11 0.55 0.25 1"),
             ("pz", f"0 0 {ins[2]}", f"{s} {s} 0.001", "0.96 0.96 0.94 1"),
             ("nz", f"0 0 -{ins[2]}", f"{s} {s} 0.001", "0.97 0.85 0.10 1")]
    return "\n".join(
        f'        <geom name="{prefix}_s{nm}" type="box" pos="{p}" size="{sz}" '
        f'rgba="{c}" contype="0" conaffinity="0" group="1" mass="0"/>'
        for nm, p, sz, c in faces)


def _default_contacts_from(surface, n, ring_z=0.009):
    """Evenly spaced directions around the equator: a starting point, not a solution."""
    out = []
    for k in range(n):
        ang = 2.0 * np.pi * k / n
        if abs(abs(np.cos(ang)) - abs(np.sin(ang))) < 1e-6:
            ang += 0.09
        d = np.array([np.cos(ang), np.sin(ang),
                      (0.35 if k % 2 == 0 else -0.35)])
        out.append(surface(d))
    return out


def _spec(name, xml_fn, surface, rest_z, assets="", mass=DEFAULT_MASS):
    return {"name": name, "xml": xml_fn, "surface_point": surface,
            "rest_z": rest_z, "assets": assets, "mass": mass,
            "default_contacts": lambda n: _default_contacts_from(surface, n)}


def make_cube(half=CUBE_HALF, mass=DEFAULT_MASS, **_):
    h = np.array([half, half, half])
    surf = box_surface(h)

    def xml(prefix, cx, cy):
        return f"""    <body name="{prefix}_obj" pos="{cx} {cy} {half}">
      <freejoint name="{prefix}_objfree"/>
      <geom name="{prefix}_objgeom" type="box" size="{half} {half} {half}"
            material="cubemat" mass="{mass}" friction="0.45 0.01 0.001" condim="4"
            solref="-4000 -60" solimp="0.92 0.97 0.001"/>
{_stickers(prefix, h)}
    </body>"""
    return _spec("cube 60mm", xml, surf, half, mass=mass)


def make_box(half=(0.040, 0.025, 0.030), mass=DEFAULT_MASS, **_):
    h = np.array(half, float)
    surf = box_surface(h)

    def xml(prefix, cx, cy):
        return f"""    <body name="{prefix}_obj" pos="{cx} {cy} {h[2]}">
      <freejoint name="{prefix}_objfree"/>
      <geom name="{prefix}_objgeom" type="box" size="{h[0]} {h[1]} {h[2]}"
            rgba="0.80 0.35 0.25 1" mass="{mass}" friction="0.45 0.01 0.001"
            condim="4" solref="-4000 -60" solimp="0.92 0.97 0.001"/>
    </body>"""
    return _spec(f"box {h*2000} mm", xml, surf, float(h[2]), mass=mass)


def make_sphere(r=0.032, mass=DEFAULT_MASS, **_):
    def xml(prefix, cx, cy):
        return f"""    <body name="{prefix}_obj" pos="{cx} {cy} {r}">
      <freejoint name="{prefix}_objfree"/>
      <geom name="{prefix}_objgeom" type="sphere" size="{r}" rgba="0.35 0.55 0.85 1"
            mass="{mass}" friction="0.45 0.01 0.001" condim="4"
            solref="-4000 -60" solimp="0.92 0.97 0.001"/>
    </body>"""
    return _spec("sphere 64mm", xml, sphere_surface(r), r, mass=mass)


def make_cylinder(r=0.028, hz=0.038, mass=DEFAULT_MASS, **_):
    def xml(prefix, cx, cy):
        return f"""    <body name="{prefix}_obj" pos="{cx} {cy} {hz}">
      <freejoint name="{prefix}_objfree"/>
      <geom name="{prefix}_objgeom" type="cylinder" size="{r} {hz}"
            rgba="0.85 0.70 0.30 1" mass="{mass}" friction="0.45 0.01 0.001"
            condim="4" solref="-4000 -60" solimp="0.92 0.97 0.001"/>
    </body>"""
    return _spec("cylinder", xml, cylinder_surface(r, hz), hz, mass=mass)


def make_capsule(r=0.024, hz=0.030, mass=DEFAULT_MASS, **_):
    # A capsule's surface is a cylinder with hemispherical caps; treat the search map as
    # the enclosing cylinder, which is close enough to seed an optimiser.
    def xml(prefix, cx, cy):
        return f"""    <body name="{prefix}_obj" pos="{cx} {cy} {r}">
      <freejoint name="{prefix}_objfree"/>
      <geom name="{prefix}_objgeom" type="capsule" size="{r} {hz}"
            euler="0 1.5708 0" rgba="0.60 0.80 0.55 1" mass="{mass}"
            friction="0.45 0.01 0.001" condim="4" solref="-4000 -60"
            solimp="0.92 0.97 0.001"/>
    </body>"""
    return _spec("capsule", xml, cylinder_surface(r, hz + r), r, mass=mass)


def make_ellipsoid(radii=(0.038, 0.028, 0.026), mass=DEFAULT_MASS, **_):
    a = np.array(radii, float)

    def xml(prefix, cx, cy):
        return f"""    <body name="{prefix}_obj" pos="{cx} {cy} {a[2]}">
      <freejoint name="{prefix}_objfree"/>
      <geom name="{prefix}_objgeom" type="ellipsoid" size="{a[0]} {a[1]} {a[2]}"
            rgba="0.75 0.45 0.80 1" mass="{mass}" friction="0.45 0.01 0.001"
            condim="4" solref="-4000 -60" solimp="0.92 0.97 0.001"/>
    </body>"""
    return _spec("ellipsoid", xml, ellipsoid_surface(a), float(a[2]), mass=mass)


def make_mesh(mesh=None, scale=1.0, mass=DEFAULT_MASS, **_):
    """Load an STL/OBJ. Collision is MuJoCo's convex hull of it, which is the default.

    If your part is meaningfully non-convex, the hull is what the physics sees and the
    contacts the optimiser finds will sit on the hull, not on the real surface. For a
    concave part, split it into convex pieces or keep primitives for collision and use
    the mesh for visual only.
    """
    if mesh is None:
        raise SystemExit("--object mesh needs --mesh path/to/file.stl")
    import os
    path = os.path.abspath(mesh)
    if not os.path.exists(path):
        raise SystemExit(f"mesh not found: {path}")
    assets = (f'    <mesh name="objmesh" file="{path}" '
              f'scale="{scale} {scale} {scale}"/>\n')

    # Probe the hull once, in a throwaway model, to get its extent and a surface map.
    import mujoco
    probe_xml = f"""<mujoco><asset>{assets}</asset><worldbody>
      <body name="probe"><geom name="pg" type="mesh" mesh="objmesh"/></body>
      </worldbody></mujoco>"""
    pm = mujoco.MjModel.from_xml_string(probe_xml)
    pd = mujoco.MjData(pm)
    mujoco.mj_forward(pm, pd)
    _DATA[0] = pd
    gid = mujoco.mj_name2id(pm, mujoco.mjtObj.mjOBJ_GEOM, "pg")
    bid = mujoco.mj_name2id(pm, mujoco.mjtObj.mjOBJ_BODY, "probe")
    surf = mesh_surface(pm, gid, bid)
    lo = pm.mesh_vert[:, 2].min() * scale
    rest = float(-lo)
    print(f"  mesh: {os.path.basename(path)}  hull verts={pm.nmeshvert}  "
          f"rest_z={rest*1000:.1f} mm")

    def xml(prefix, cx, cy):
        return f"""    <body name="{prefix}_obj" pos="{cx} {cy} {rest}">
      <freejoint name="{prefix}_objfree"/>
      <geom name="{prefix}_objgeom" type="mesh" mesh="objmesh"
            rgba="0.70 0.72 0.78 1" mass="{mass}" friction="0.45 0.01 0.001"
            condim="4" solref="-4000 -60" solimp="0.92 0.97 0.001"/>
    </body>"""
    return _spec(os.path.basename(path), xml, surf, rest, assets=assets, mass=mass)


OBJECTS = {
    "cube": make_cube,
    "box": make_box,
    "sphere": make_sphere,
    "cylinder": make_cylinder,
    "capsule": make_capsule,
    "ellipsoid": make_ellipsoid,
    "mesh": make_mesh,
}


def object_xml(obj, prefix, cx, cy):
    return obj["xml"](prefix, cx, cy)
