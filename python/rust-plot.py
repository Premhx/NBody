import matplotlib.pyplot as plt
import pickle
from tqdm import tqdm

def plot_simulation(filename):
    with open(filename, 'rb') as file:
        coordinates = pickle.load(file)

    for step_coords in tqdm(coordinates):
        for coord in step_coords:
            plt.scatter(coord[0], coord[1], color='blue', s=5)

            plt.xlabel('X Position')
            plt.ylabel('Y Position')
            plt.title('N-Body Simulation')
            plt.show()

# Example usage:
plot_simulation('simulation_coords.pkl')