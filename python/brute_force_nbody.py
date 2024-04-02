import numpy as np
from numba import njit, prange
import pickle
import matplotlib.pyplot as plt 
from tqdm import tqdm 

G = 1.e-6
dt = 1.e-2
N_bodies = 100
N_steps = 500

# Fix Seed for Initialization
np.random.seed(123)

# Initial Conditions
Masses = np.random.random(N_bodies)*10
X = np.random.random(N_bodies)
Y = np.random.random(N_bodies)
PX = np.random.random(N_bodies) - 0.5
PY = np.random.random(N_bodies) - 0.5
pos = np.array((X,Y))
mom = np.array((PX, PY))

@njit(parallel=True)
def force_array(pos_arr, m_array):
    n = pos_arr.shape[1]
    force_arr = np.zeros((2 ,n, n))
    for i in prange(n):
        for j in range(i):
            force = G * m_array[i] * m_array[j] * (pos_arr[:,i] - pos_arr[:, j]) / np.abs((pos_arr[:,i] - pos_arr[:, j]))**3
            force_arr[:, i, j] = force
            force_arr[:, j, i] = -force
    return force_arr

@njit(parallel=True)
def update_mom(step, mom_arr, force_arr):
    n = mom_arr.shape[1]
    del_mom = np.zeros_like(mom_arr)
    for i in prange(n):
        for j in range(n):
            del_mom[:, i] += step * force_arr[:, i, j]
    return mom_arr + del_mom

@njit(parallel=True)
def update_pos(step, pos_arr, new_mom, m_arr):
    return pos_arr + step * new_mom / m_arr

@njit(parallel=True)
def main_loop(n, pos_arr, mom_arr):
    for i in prange(n):
        force = force_array(pos_arr, Masses)
        mom_arr = update_mom(dt, mom_arr, force)
        pos_arr = update_pos(dt, pos_arr, mom_arr, Masses)
    return pos_arr

padding_zero=len(str(N_steps))
for i in tqdm(range(N_steps)):
    force = force_array(pos, Masses)
    mom = update_mom(dt, mom, force)
    pos = update_pos(dt, pos, mom, Masses)
    plt.plot(pos[0],pos[1],'o', markersize=1)
    plt.savefig(f"img/{str(i).zfill(padding_zero)}.jpg")

with open("pos.p", "wb") as f: # "wb" because we want to write in binary mode
    pickle.dump(pos, f)
