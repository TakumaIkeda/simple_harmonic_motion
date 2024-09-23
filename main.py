import numpy as np
import matplotlib.pyplot as plt
import struct

def float32(n):
  return struct.unpack('f', struct.pack('f', n))[0]

def verlet(r, r_prev, dt, m, r0):
  F = -float32(r - r0)
  tmp = float32(r)
  # print(float32(r), float32(r - r_prev))
  return 2 * r - r_prev + F * dt ** 2 / m, tmp

def velocity_verlet(r, v, dt, m, r0):
  F = -float32((r - r0))
  v = float32(v + F * dt / (2 * m))
  r = float32(r + v * dt)
  F = -float32(r - r0)
  v = float32(v + F * dt / (2 * m))
  return r, v

def analytical(r_init, v_init, t, m):
  return np.float32(r_init * np.cos(t * np.sqrt(1 / m)) + v_init / np.sqrt(1 / m) * np.sin(t * np.sqrt(1 / m)))

def main():
  # define parameters
  nsteps = 50000
  dt = 1e-10
  m = 1e-13
  r0 = 0
  r_init = 1
  v_init = 0

  # initialize
  r = r_init
  r_prev = analytical(r_init, v_init, -dt, m)
  history_verlet = np.array([r], dtype=np.float32)
  history_vv = np.array([r], dtype=np.float32)
  history_analytical = np.array([r_init], dtype=np.float32)

  for _ in range(nsteps):
    r, r_prev = verlet(r, r_prev, dt, m, r0)
    # cast r, r_prev to float32 to save memory
    history_verlet = np.append(history_verlet, r)

  r = r_init
  v = v_init
  for _ in range(nsteps):
    r, v = velocity_verlet(r, v, dt, m, r0)
    history_vv = np.append(history_vv, r)

  r = 2
  v = 0
  t = np.arange(0, nsteps * dt, dt)
  history_analytical = analytical(r_init, v_init, t, m)

  plt.figure()
  plt.plot(history_verlet, label='verlet')
  plt.plot(history_vv, label='velocity verlet')
  plt.plot(history_analytical, label='analytical')
  plt.legend()
  plt.savefig('all.png')

  plt.figure()
  plt.plot(history_verlet, label='verlet')
  plt.plot(history_vv, label='velocity verlet')
  plt.legend()
  plt.savefig('v-vv.png')

  plt.figure()
  plt.plot(history_verlet, label='verlet')
  plt.plot(history_analytical, label='analytical')
  plt.legend()
  plt.savefig('v-analytical.png')

if __name__ == '__main__':
  main()