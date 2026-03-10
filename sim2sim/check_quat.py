import numpy as np
from isaacgym.torch_utils import quat_rotate_inverse
import torch

a = np.radians(10)
q = torch.tensor([[0, np.sin(a/2), 0, np.cos(a/2)]], dtype=torch.float32)
g = torch.tensor([[0.0, 0.0, -1.0]], dtype=torch.float32)
r = quat_rotate_inverse(q, g)
print("IsaacGym result:", r.numpy()[0])

# numpy version
def qri(q, v):
    w = q[3]; u = q[:3]
    return v * (2*w**2 - 1) - np.cross(u, v) * w * 2 + u * np.dot(u, v) * 2

q_np = np.array([0, np.sin(a/2), 0, np.cos(a/2)])
r2 = qri(q_np, np.array([0, 0, -1.0]))
print("Numpy result:", r2)
print("Match:", np.allclose(r.numpy()[0], r2, atol=1e-5))

# MuJoCo converted
qm = np.array([np.cos(a/2), 0, np.sin(a/2), 0])  # wxyz
qi = np.array([qm[1], qm[2], qm[3], qm[0]])  # xyzw
r3 = qri(qi, np.array([0, 0, -1.0]))
print("MuJoCo conv:", r3)
print("Match MJ:", np.allclose(r.numpy()[0], r3, atol=1e-5))
