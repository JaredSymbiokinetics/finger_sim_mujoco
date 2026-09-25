"""ONE finger on the top vertex vs. genuine tipping torques."""
import sys, numpy as np, mujoco
sys.path.insert(0, '/mnt/user-data/outputs/finger_simulation')
import swarm_sim as S
from scipy.spatial.transform import Rotation as Rot

a = S.CUBE_HALF
diag = np.array([1.0, 1.0, 1.0]) / np.sqrt(3.0)
R_corner = Rot.from_quat(S.CORNER_QUAT)
S.set_task('corner')

def run(with_finger, torque_Nm, settle_z):
    S.set_finger_count(0, contacts=[(np.array([a, a, a]), diag)])
    class Hold:
        duration = 9.0
        def __call__(self, t):
            return np.array([0.0, 0.0, settle_z]), R_corner
    S.TASKS['corner'] = (lambda: (Hold(), [(0.0, 9.0, 'BALANCE')]), 'corner')
    S.CONTROL = 'position'
    S.GRASP_START, S.GRASP_END = 0.0, 0.05
    S.squeeze_depth = lambda t, tend: 0.004
    sim = S.Sim('dynamic')
    q = sim.cube_qadr
    sim.data.qpos[q:q+3] = [0, 0, settle_z]
    xyzw = R_corner.as_quat()
    sim.data.qpos[q+3:q+7] = [xyzw[3], xyzw[0], xyzw[1], xyzw[2]]
    fq = sim.finger_qadr[0]
    n_w = R_corner.apply(diag)
    far = np.array([0, 0, 10.0])
    tip = (np.array([0,0,settle_z]) + R_corner.apply(np.array([a,a,a]))
           + n_w * (S.TIP_RADIUS - 0.004))
    sim.data.qpos[fq:fq+3] = tip if with_finger else far
    sim.data.qpos[fq+3:fq+7] = S.finger_quat(n_w)
    mujoco.mj_forward(sim.model, sim.data)
    sim._prev_targets = sim.data.qpos[fq:fq+3].copy().reshape(1,3)
    if not with_finger:
        sim.finger_targets = lambda t: (np.array([far]), [S.finger_quat(n_w)])

    def tilt():
        _, R = sim.object_pose()
        return np.rad2deg(np.arccos(np.clip(R.apply(diag)[2], -1, 1)))

    peak, grip_min, hist = 0.0, 9.9, []
    while sim.data.time < 9.0:
        t = sim.data.time
        # tipping torque bursts about a horizontal axis
        if any(abs(t-k) < 0.08 for k in (1.5, 3.5, 5.5, 7.0)):
            ax = {0:[1,0,0],1:[0,1,0],2:[0.7,0.7,0],3:[-1,0,0]}[
                [k for k in (1.5,3.5,5.5,7.0) if abs(t-k)<0.08][0].__hash__() % 4]
            sim.data.xfrc_applied[sim.cube_bid, 3:] += np.array(ax)*torque_Nm
        sim.step()
        if t > 1.0:
            peak = max(peak, tilt())
            if with_finger:
                grip_min = min(grip_min, sim.grip_forces()[0])
        if abs(round(t,3)*2 - round(round(t,3)*2)) < 1e-9:
            hist.append((round(t,2), tilt()))
    return peak, tilt(), grip_min, hist

# settled height of the cube resting on its vertex
S.set_finger_count(0, contacts=[(np.array([a,a,a]), diag)])
class H0:
    duration=1.0
    def __call__(self,t): return np.array([0,0,S.CORNER_Z]), R_corner
S.TASKS['corner']=(lambda:(H0(),[(0,1,'x')]),'corner')
S.CONTROL='position'; S.GRASP_START,S.GRASP_END=99,99
s0=S.Sim('dynamic')
q=s0.cube_qadr; s0.data.qpos[q:q+3]=[0,0,S.CORNER_Z]
x=R_corner.as_quat(); s0.data.qpos[q+3:q+7]=[x[3],x[0],x[1],x[2]]
s0.data.qpos[s0.finger_qadr[0]:s0.finger_qadr[0]+3]=[0,0,10]
mujoco.mj_forward(s0.model,s0.data)
for _ in range(300): s0.step()
settle_z = s0.object_pose()[0][2]
print(f"cube settles on its vertex at z = {settle_z*1000:.2f} mm "
      f"(ideal {S.CORNER_Z*1000:.2f} mm)\n")

for tq in (0.005, 0.010, 0.020):
    pk_no, fin_no, _, _ = run(False, tq, settle_z)
    pk_yes, fin_yes, gmin, hist = run(True, tq, settle_z)
    print(f"tipping torque {tq*1000:5.1f} mN m")
    print(f"   no finger : peak tilt {pk_no:6.2f} deg   final {fin_no:6.2f} deg")
    print(f"   1 finger  : peak tilt {pk_yes:6.2f} deg   final {fin_yes:6.2f} deg"
          f"   min grip {gmin:.2f} N")
