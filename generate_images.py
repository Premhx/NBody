import os  
import numpy as np  
import matplotlib.pyplot as plt  
from multiprocessing import Pool  
import pickle  
from tqdm import tqdm 
from glob import glob 

# Load the trajectories from disk  
with open('trajectories.pkl', 'rb') as f:  
    trajectories = pickle.load(f)  
  
# Create a directory for the frames  
frames_dir = "frames"  
# Check if the directory already exists
if os.path.exists(frames_dir):
    # If it does, remove all PNG files within it
    png_files = glob(os.path.join(frames_dir, '*.png'))
    for png_file in png_files:
        os.remove(png_file)
else:
    # If the directory doesn't exist, create it
    os.makedirs(frames_dir)
    
# Function to plot and save a single frame  
def save_frame(step,dpi=300):  
    
    # The target resolution is 1920x1080 pixels  
    target_width_px = 1920  
    target_height_px = 1080  
      
    # Calculate the figure size in inches (width, height)  
    fig_width_in = target_width_px / dpi  
    fig_height_in = target_height_px / dpi  
      
    # Create a new figure with the specified size and DPI  
    fig, ax = plt.subplots(figsize=(fig_width_in, fig_height_in), dpi=dpi)     
    # fig, ax = plt.subplots()
    limits=100
    ax.set_xlim(-limits, limits)  # Set appropriate limits  
    ax.set_ylim(-limits, limits)  # Set appropriate limits  
    for j in range(len(trajectories[step])):  
        ax.plot(trajectories[step][j, 0], trajectories[step][j, 1], 'o', markersize=1)  
    # Format the filename with leading zeros  
    frame_filename = os.path.join(frames_dir, f"frame_{step:06d}.png")  
    fig.savefig(frame_filename, dpi=dpi, format='png')  
    plt.close(fig)  
  
# Determine the number of processes to use  
num_processes = os.cpu_count()-2
  
# Generate and save frames using multiprocessing  
if __name__ == '__main__':  
    with Pool(num_processes) as pool:  
        max_ = len(trajectories)  
        with tqdm(total=max_, desc="Generating frames", unit="frame") as pbar:  
            for _ in pool.imap_unordered(save_frame, range(max_)):  
                pbar.update()  
  
