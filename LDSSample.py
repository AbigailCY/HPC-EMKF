import numpy as np

L = 1000
dimz = 2
dimx = 4

# Initialize
zs = np.zeros([L, dimz])
xs = np.zeros([L, dimx])
u0 = np.zeros(dimz)
S0 = np.zeros([dimz]*2)
A = np.zeros([dimz]*2)
Q = np.zeros([dimz]*2)
C = np.zeros([dimx, dimz])
R = np.zeros([dimx]*2)

# Assign values

u0 = np.random.random(size=dimz)
S0 = np.random.random(size=[dimz]*2)
A = np.array([[0.5,0],[0,0.25]])#np.random.random(size=[dimz]*2)
Q = np.random.random(size=[dimz]*2)
Q = Q @ Q
C = np.random.random(size=[dimx, dimz])
R = np.random.random(size=[dimx]*2)
R = R @ R

zs[0] = u0
for i in range(1,L):
  zs[i] = A@zs[i-1] + Q @ np.random.random(size=dimz)
  xs[i] = C@zs[i] + R @ np.random.random(size=dimx)
