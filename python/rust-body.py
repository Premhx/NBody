import numpy as np
import pickle
import math
import random
from tqdm import tqdm
from numba import jit, prange

def rand_disc():
    theta = random.random() * 2 * math.pi
    # pairs=np.array([math.cos(theta), math.sin(theta)]) * random.random()
    pairs=np.random.rand(2)
    return pairs



def rand_body():
    pos = rand_disc()
    vel = rand_disc()
    return np.concatenate((pos, vel, np.random.rand(1)))

@jit(nopython=True, parallel=True)  
def update_bodies(bodies, dt):
    n = len(bodies)
    acc = np.zeros((n, 2))
    d_min = 0.0001
    for i in prange(n):
        p1 = bodies[i, :2]
        m1 = bodies[i, 4]
        for j in range(i + 1, n):
            p2 = bodies[j, :2]
            m2 = bodies[j, 4]

            r = p2 - p1
            mag_sq = max(np.sum(r ** 2), d_min)
            mag = math.sqrt(mag_sq)
            tmp = r / (mag_sq * mag)

            acc[i] += m2 * tmp
            acc[j] -= m1 * tmp

    bodies[:, 2:4] += acc * dt
    bodies[:, :2] += bodies[:, 2:4] * dt
    
def compile_simulation(num_bodies=10, num_iterations=10):
    # Generate a small set of random bodies
    bodies = np.array([rand_body() for _ in range(num_bodies)], dtype='float64')
    
    # Run a few iterations to compile the function
    for _ in range(num_iterations):
        update_bodies(bodies, 0.0001)
        
        
def simulate_and_save(seed, steps, filename):
    random.seed(seed)
    np.random.seed(seed)
    N = 2000
    bodies = np.array([rand_body() for _ in range(N)], dtype='float64')
    coordinates = []

    for _ in tqdm(range(steps)):
        step_coords = np.copy(bodies[:, :2])
        coordinates.append(step_coords)
        update_bodies(bodies, 0.0001)

    with open(filename, 'wb') as file:
        pickle.dump(coordinates, file)
        
        
compile_simulation()
# Example usage:
simulate_and_save(3, 1000, 'simulation_coords.pkl')
