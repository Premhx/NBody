from typing import List, Tuple
import numpy as np
from numba import jit
from tqdm import tqdm
import cv2
from numba import  float64, int64, types
from numba.experimental import jitclass

# Constants
G = 6.67430e-11  # gravitational constant
dt = 0.1  # time step
num_frames = 100


spec = [
    ('center', float64[:]),
    ('size', float64),
    ('mass', float64),
    ('com', float64[:]),
    ('indices', types.UniTuple(int64, 4)),
    ('children', types.Tuple((types.Optional('QuadTreeNode'),) * 4))
]

@jitclass(spec)
class QuadTreeNode:
    """Class representing a node in the Barnes-Hut quadtree."""
    def __init__(self, center, size):
        self.center = center
        self.size = size
        self.mass = 0.0
        self.com = np.zeros(2, dtype=np.float64)
        self.indices = ()
        self.children = (None, None, None, None)
        
class Body:
    """Class representing a celestial body."""
    def __init__(self, mass: float, position: np.ndarray, velocity: np.ndarray):
        self.mass = mass
        self.position = position
        self.velocity = velocity
        self.acceleration = np.zeros(2, dtype=np.float64)

# class QuadTreeNode:
#     """Class representing a node in the Barnes-Hut quadtree."""
#     def __init__(self, center: np.ndarray, size: float):
#         self.center = center
#         self.size = size
#         self.mass = 0.0
#         self.com = np.zeros(2, dtype=np.float64)
#         self.indices: Tuple[int, int, int, int] = ()
#         self.children = [None, None, None, None]

@jit(nopython=True)
def build_quadtree(bodies: np.ndarray, center: np.ndarray, size: float) -> QuadTreeNode:
    quadtree = QuadTreeNode(center, size)
    for i in range(len(bodies)):
        body = bodies[i]
        if in_quad(body.position, quadtree):
            if len(quadtree.indices) == 0:
                quadtree.mass = body.mass
                quadtree.com = body.position
            else:
                quadtree.mass += body.mass
                quadtree.com = (quadtree.com * (quadtree.mass - body.mass) + body.mass * body.position) / quadtree.mass
            quadtree.indices += (i,)
        else:
            continue

    if len(quadtree.indices) > 1:
        if quadtree.size > 1e-3:
            quadtree.children = [
                QuadTreeNode(np.array([quadtree.center[0] - quadtree.size / 4, quadtree.center[1] - quadtree.size / 4]), quadtree.size / 2),
                QuadTreeNode(np.array([quadtree.center[0] + quadtree.size / 4, quadtree.center[1] - quadtree.size / 4]), quadtree.size / 2),
                QuadTreeNode(np.array([quadtree.center[0] - quadtree.size / 4, quadtree.center[1] + quadtree.size / 4]), quadtree.size / 2),
                QuadTreeNode(np.array([quadtree.center[0] + quadtree.size / 4, quadtree.center[1] + quadtree.size / 4]), quadtree.size / 2)
            ]
            for child in quadtree.children:
                build_quadtree(bodies, child.center, child.size)

    return quadtree

@jit(nopython=True)
def in_quad(pos: np.ndarray, quadtree: QuadTreeNode) -> bool:
    return (pos[0] >= quadtree.center[0] - quadtree.size / 2) and (pos[0] <= quadtree.center[0] + quadtree.size / 2) and (pos[1] >= quadtree.center[1] - quadtree.size / 2) and (pos[1] <= quadtree.center[1] + quadtree.size / 2)

@jit(nopython=True)
def calculate_force(body: Body, bodies: np.ndarray, quadtree: QuadTreeNode, G: float, theta: float) -> np.ndarray:
    if quadtree.mass == 0:
        return np.zeros(2, dtype=np.float64)

    if len(quadtree.indices) == 1 and quadtree.indices[0] == body:
        return np.zeros(2, dtype=np.float64)

    distance = np.linalg.norm(quadtree.com - body.position)
    if distance == 0:
        return np.zeros(2, dtype=np.float64)

    if quadtree.size / distance < theta or len(quadtree.indices) == 1:
        direction = quadtree.com - body.position
        force = G * quadtree.mass * body.mass / distance ** 2 * direction / distance
        return force

    total_force = np.zeros(2, dtype=np.float64)
    for child in quadtree.children:
        total_force += calculate_force(body, bodies, child, G, theta)

    return total_force

@jit(nopython=True)
def update_acceleration(bodies: np.ndarray, quadtree: QuadTreeNode, G: float, theta: float) -> None:
    for i in range(len(bodies)):
        acceleration = np.zeros(2, dtype=np.float64)
        for j in range(len(bodies)):
            if j != i:
                force = calculate_force(bodies[i], bodies, quadtree, G, theta)
                acceleration += force / bodies[i].mass
        bodies[i].acceleration = acceleration

@jit(nopython=True)
def update_velocity_position(bodies: np.ndarray, dt: float) -> None:
    for body in bodies:
        body.velocity += body.acceleration * dt
        body.position += body.velocity * dt

# Generate N random bodies
def generate_random_bodies(N: int) -> np.ndarray:
    np.random.seed(0)
    return np.array([Body(np.random.uniform(1, 10), np.random.uniform(-10, 10, size=(2,)), np.zeros(2)) for _ in range(N)])

# Initialize video writer
frame_width, frame_height = 800, 800
fourcc = cv2.VideoWriter_fourcc(*'MP4V')
video_writer = cv2.VideoWriter('simulation.mp4', fourcc, 20.0, (frame_width, frame_height))

# Main simulation loop
bodies = generate_random_bodies(10)
for frame in tqdm(range(num_frames)):
    # Build quadtree
    quadtree = build_quadtree(bodies, np.array([0.0, 0.0]), 20.0)

    # Update forces
    update_acceleration(bodies, quadtree, G, theta=0.5)

    # Update velocities and positions
    update_velocity_position(bodies, dt)

    # Create visualization
    frame_image = np.ones((frame_height, frame_width, 3), dtype=np.uint8) * 255  # White background
    for body in bodies:
        x, y = int(body.position[0] / 20 * frame_width / 2 + frame_width / 2), int(-body.position[1] / 20 * frame_height / 2 + frame_height / 2)
        cv2.circle(frame_image, (x, y), 5, (0, 0, 0), -1)  # Draw black circle for each body

    # Write frame to video
    video_writer.write(frame_image)

# Release video writer
video_writer.release()
