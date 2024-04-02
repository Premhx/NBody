import numpy as np  
from tqdm import tqdm 
import pickle  
from numba import jit  , prange 

  
# Function to generate initial conditions  
def initialize_bodies(N):  
    np.random.seed(0)  # For reproducibility  
    bounds=50
    positions = np.random.uniform(-bounds, bounds, (N, 2)) 
    # positions = np.random.rand(N, 2) 
    
    velocities = np.zeros((N, 2), dtype=np.float64)  # Initial velocities set to zero  
    masses = np.random.rand(N) + 0.00005  # Random masses (ensure no mass is too small)  
    # masses[-1]=2000
    # positions[-1]=[0,0]
    return positions, velocities, masses  
  
  
# Function to calculate pairwise forces   
@jit(nopython=True,parallel=True)  
def compute_forces(pos, masses):  
    num_bodies = len(masses)  
    softening_dist=0.0001
    forces = np.zeros((num_bodies, 2))  
    for i in prange(num_bodies):  
        for j in range(i + 1, num_bodies):  
            r_ij = pos[j] - pos[i]  
            distance = np.sqrt(r_ij[0]**2 + r_ij[1]**2)  
            if distance > 0:  # Avoid division by zero and very close encounters  
                force_magnitude = (G * masses[i] * masses[j]) / distance**2  
                force_direction = r_ij / (distance *softening_dist)
                forces[i] += force_magnitude * force_direction  
                forces[j] -= force_magnitude * force_direction  
    return forces  

# Time stepping parameters  
dt = 0.0001  # Time step  
num_steps = 2000  # Number of time steps to simulate  
# Constants  
# G = 6.67430e-11  # Gravitational constant in m^3 kg^-1 s^-2  
G = 1e0  # Gravitational constant in m^3 kg^-1 s^-2  
# Number of bodies  
N = 4000  # Change this to simulate different number of bodies  
  
# Initial conditions  
positions, velocities, masses = initialize_bodies(N)  



# Store the positions at each time step  
trajectories = []  
# Time integration loop  
for step in tqdm(range(num_steps)):  
    # Compute forces on each body  
    forces = compute_forces(positions, masses)  
      
    # Update velocities  
    velocities += dt * forces / masses[:, np.newaxis]  
    
    # Update positions  
    positions += dt * velocities  
    # Store the positions for later plotting  
    trajectories.append(positions.copy())     
    
# Save the trajectories to disk using pickle  
with open('trajectories.pkl', 'wb') as f:  
    pickle.dump(trajectories, f)  
  
  
