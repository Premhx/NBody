import os
import numpy as np
import cv2
import pickle
from tqdm import tqdm


def read_csv_file(filename):
    coordinates = []
    with open(filename, 'r') as file:
        step_coords = []
        for line in file:
            if line.strip():  # Non-empty line
                x, y = map(float, line.strip().split(','))
                step_coords.append([x, y])
            else:  # Empty line indicates the end of a step
                coordinates.append(np.array(step_coords))
                step_coords = []
        # Append the last set of coordinates
        if step_coords:
            coordinates.append(np.array(step_coords))
    return coordinates

# Example usage:


# Load the trajectories from disk
# with open('trajectories.pkl', 'rb') as f:
#     trajectories = pickle.load(f)


# with open('simulation_coords.pkl', 'rb') as f:
#     trajectories = pickle.load(f)


trajectories = read_csv_file('C/simulation_coords.csv')


target_width_px = 1920
target_height_px = 1080

# # Function to generate and save a single frame
# def generate_frame(step):
#     target_width_px = 1920
#     target_height_px = 1080
#     limits = 100

#     frame = np.zeros((target_height_px, target_width_px, 3), dtype=np.uint8) 
#     for j in range(len(trajectories[step])):
#         x = int(trajectories[step][j, 0]) + target_width_px // 2
#         y = int(trajectories[step][j, 1]) + target_height_px // 2
#         # print("x:", x, "y:", y)  # Add debug print
#         cv2.circle(frame, (x, y), 1, (255, 255, 255), -1)

#     return frame




# Find the bounding box that encompasses all particles
max_distance_from_origin = np.max(np.linalg.norm(trajectories, axis=2))
bounding_box_half_size = max_distance_from_origin * 0.3  # Add some margin

# Calculate scaling factors
scale_x = target_width_px / (2 * bounding_box_half_size)
scale_y = target_height_px / (2 * bounding_box_half_size)

# Update scaling in the generate_frame function
def generate_frame(step):
    frame = np.zeros((target_height_px, target_width_px, 3), dtype=np.uint8) 
    for j in range(len(trajectories[step])):
        scaled_x = int(trajectories[step][j, 0] * scale_x)
        scaled_y = int(trajectories[step][j, 1] * scale_y)
        x = scaled_x + target_width_px // 2
        y = scaled_y + target_height_px // 2
        cv2.circle(frame, (x, y), 1, (255, 255, 255), -1)
    return frame




# # Find the minimum and maximum x and y coordinates across all time steps
# all_positions = np.concatenate(trajectories, axis=0)
# min_x = np.min(all_positions[:, 0])
# max_x = np.max(all_positions[:, 0])
# min_y = np.min(all_positions[:, 1])
# max_y = np.max(all_positions[:, 1])

# # Calculate the centroid of all particles
# centroid_x = np.mean(all_positions[:, 0])
# centroid_y = np.mean(all_positions[:, 1])

# # Determine the bounding box size to include approximately 80% of particles
# bounding_box_width = max_x - min_x
# bounding_box_height = max_y - min_y

# # Adjust the bounding box size to have 80% of particles within it
# scale_factor = 0.3
# bounding_box_width *= scale_factor
# bounding_box_height *= scale_factor

# # Calculate scaling factors based on the adjusted bounding box size and target resolution
# scale_x = target_width_px / bounding_box_width
# scale_y = target_height_px / bounding_box_height

# # Update scaling in the generate_frame function
# def generate_frame(step):
#     # frame = np.ones((target_height_px, target_width_px, 3), dtype=np.uint8) * 255
#     frame = np.zeros((target_height_px, target_width_px, 3), dtype=np.uint8) 
    
#     for j in range(len(trajectories[step])):
#         scaled_x = int((trajectories[step][j, 0] - centroid_x) * scale_x) + target_width_px // 2
#         scaled_y = int((trajectories[step][j, 1] - centroid_y) * scale_y) + target_height_px // 2
#         if 0 <= scaled_x < target_width_px and 0 <= scaled_y < target_height_px:
#             # cv2.circle(frame, (scaled_x, scaled_y), 1, (0, 0, 0), -1)  # Draw only if within frame bounds
#             cv2.circle(frame, (scaled_x, scaled_y), 1, (255, 255, 255), -1)  # Draw only if within frame bounds
            
#     return frame




# Generate frames sequentially
if __name__ == '__main__':
    max_frames = len(trajectories)

    # Video writer initialization
    target_width_px = 1920
    target_height_px = 1080
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter('output_video.mp4', fourcc, 60, (target_width_px, target_height_px))

    with tqdm(total=max_frames, desc="Generating frames", unit="frame") as pbar:
        for step in range(max_frames):
            frame = generate_frame(step)
            out.write(frame)
            pbar.update(1)

    # Release video writer
    out.release()