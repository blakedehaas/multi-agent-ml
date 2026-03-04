import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.widgets import Slider, Button
from matplotlib.patches import Polygon
from matplotlib.collections import PatchCollection, LineCollection
from collections import deque

#
# 
# Initialize arena, set parameters
#
#

NUM_AGENTS = 50
ARENA_RADIUS = 15.0
AGENT_LENGTH = 0.5
PLAYBACK_FPS = 30

TRAIL_LENGTH = 20
TARGET_AGENT_IDX = 0  

INITIAL_NOISE = 0.5                         # Initial noise
V_0 = 0.5                                   # Constant speed for all agents
INTERACTION_RADIUS = 2.0                    # Radius r for neighborhood

#
# 
# position & velocity update model (Vicsek)
#
#

def compute_update(pos, vel, noise):
    ### Calculate all distances between all agents
    diff = pos[np.newaxis, :, :] - pos[:, np.newaxis, :]
    dist = np.linalg.norm(diff, axis=2)
    # dist is a NxN matrix where N = NUM_AGENTS. Each row, col here represents the distance between the two nodes.
    # The main diagonal of dist is 0
    
    # TODO: Implement the Vicsek update rules
    # 
    # Conceptually, for each point:
    #   1. Find the nodes within range (INTERACTION_RADIUS)
    #   2. Sum the velocities of nearby nodes for y and x
    #   3. Use np.arctan2(y,x) to find average heading theta_R
    #   4. Add random noise to theta_R
    #   5. Update agent’s velocity using the new theta and constant speed V_0

    np.fill_diagonal(dist, 1.0) # prevents division by 0
    unit_vectors = diff / dist[:, :, np.newaxis]
    for i in range(N):
        unit_vectors[i, i] = [0.0, 0.0]

    ### Alignment
    # Find agents within the alignment radius
    interaction_mask = dist < INTERACTION_RADIUS
    np.fill_diagonal(interaction_mask, False) # ignore self

    # Count neighbors to find averages, default to 1 to prevent division by 0
    neighbor_counts_att = np.sum(interaction_mask, axis=1)
    counts_att = np.where(neighbor_counts_att > 0, neighbor_counts_att, 1)

    # sum velocities of valid neighbors
    sum_vel = interaction_mask.astype(float) @ vel
    avg_vel = np.arctan2(sum_vel[:, 1], sum_vel[:, 0]) + np.random.uniform(-noise, noise, N)

    # Calculate steering force towards local heading
    F_interaction = np.zeros_like(vel)
    has_neighbors_att = neighbor_counts_att > 0
    F_interaction[has_neighbors_att] = avg_vel[has_neighbors_att] - vel[has_neighbors_att]

    # define new_vel, a (N,2), where each row is the update velocity for an agent, and [:,0] is x update, [:,1] is y update

    ### Update positions
    new_pos = pos + new_vel
    
    ### If agent leaves boundry, wrap around
    dist_from_center = np.linalg.norm(new_pos, axis=1)
    out_of_bounds = dist_from_center > ARENA_RADIUS
    if np.any(out_of_bounds):
        direction = new_pos[out_of_bounds] / dist_from_center[out_of_bounds, np.newaxis]
        new_pos[out_of_bounds] = new_pos[out_of_bounds] - (2 * ARENA_RADIUS * direction)
        
    return new_pos, new_vel

#
# 
# Agent setup
#
#

def generate_initial_state():
    angles = np.random.uniform(0, 2 * np.pi, NUM_AGENTS)
    radii = np.random.uniform(0, ARENA_RADIUS, NUM_AGENTS)
    pos = np.column_stack((radii * np.cos(angles), radii * np.sin(angles)))
    
    # Initialize with random headings and constant speed V_0
    vel_angles = np.random.uniform(-np.pi, np.pi, NUM_AGENTS)
    vel = V_0 * np.column_stack((np.cos(vel_angles), np.sin(vel_angles)))
    return pos, vel

positions, velocities = generate_initial_state()

agent_trail = deque(maxlen=TRAIL_LENGTH)
agent_trail.append(positions[TARGET_AGENT_IDX].copy())

#
# 
# Plots & gui
#
#

fig, ax = plt.subplots(figsize=(8, 8))
plt.subplots_adjust(bottom=0.3) 

ax.set_xlim(-ARENA_RADIUS * 1.5, ARENA_RADIUS * 1.5)
ax.set_ylim(-ARENA_RADIUS * 1.5, ARENA_RADIUS * 1.5)
ax.set_aspect('equal')
ax.set_title("Vicsek Model Simulation")

arena = plt.Circle((0, 0), ARENA_RADIUS, color='lightgray', fill=False, linestyle='--')
ax.add_patch(arena)

trail_collection = LineCollection([], linewidths=2)
ax.add_collection(trail_collection)

def get_agent_polygons(pos, vel):
    polys = []
    for p, v in zip(pos, vel):
        angle = np.arctan2(v[1], v[0])
        p1 = np.array([AGENT_LENGTH, 0])
        p2 = np.array([-AGENT_LENGTH/2, AGENT_LENGTH/2])
        p3 = np.array([-AGENT_LENGTH/2, -AGENT_LENGTH/2])
        rot = np.array([[np.cos(angle), -np.sin(angle)],
                        [np.sin(angle),  np.cos(angle)]])
        poly = np.dot(np.array([p1, p2, p3]), rot.T) + p
        polys.append(poly)
    return polys

agent_colors = np.array(['blue'] * NUM_AGENTS, dtype=object)
agent_colors[TARGET_AGENT_IDX] = 'red'

collection = PatchCollection([Polygon(p) for p in get_agent_polygons(positions, velocities)], 
                             facecolor=agent_colors, edgecolor='black')
ax.add_collection(collection)

#
# 
# GUI controls
#
#

ax_noise = plt.axes([0.15, 0.15, 0.65, 0.03])
slider_noise = Slider(ax_noise, 'Noise ($\eta$)', 0.0, 2 * np.pi, valinit=INITIAL_NOISE)

is_playing = True

def update_trail_graphics():
    if len(agent_trail) < 2: 
        trail_collection.set_segments([])
        return

    trail_points = np.array(agent_trail)
    segments = np.stack((trail_points[:-1], trail_points[1:]), axis=1)
    
    num_segments = len(segments)
    alphas = np.linspace(0.1, 1.0, num_segments)
    
    segment_colors = np.zeros((num_segments, 4))
    segment_colors[:, 0] = 1.0 
    segment_colors[:, 3] = alphas 
    
    segment_lengths = np.linalg.norm(segments[:, 0, :] - segments[:, 1, :], axis=1)
    jumps = segment_lengths > ARENA_RADIUS
    segment_colors[jumps, 3] = 0.0

    trail_collection.set_segments(segments)
    trail_collection.set_color(segment_colors)

def toggle_play(event):
    global is_playing
    is_playing = not is_playing
    btn_play.label.set_text('Play' if not is_playing else 'Pause')
    fig.canvas.draw_idle()

def step_frame(event):
    global is_playing, positions, velocities
    if not is_playing:
        positions, velocities = compute_update(positions, velocities, slider_noise.val)
        agent_trail.append(positions[TARGET_AGENT_IDX].copy())
        collection.set_paths([Polygon(p) for p in get_agent_polygons(positions, velocities)])
        update_trail_graphics()
        fig.canvas.draw_idle()

def reset_sim(event):
    global positions, velocities, agent_trail
    positions, velocities = generate_initial_state()
    agent_trail.clear()
    agent_trail.append(positions[TARGET_AGENT_IDX].copy())
    
    collection.set_paths([Polygon(p) for p in get_agent_polygons(positions, velocities)])
    update_trail_graphics()
    fig.canvas.draw_idle()

# Button positions
ax_play = plt.axes([0.25, 0.02, 0.15, 0.05])
btn_play = Button(ax_play, 'Pause')
btn_play.on_clicked(toggle_play)

ax_step = plt.axes([0.45, 0.02, 0.15, 0.05])
btn_step = Button(ax_step, 'Step Frame')
btn_step.on_clicked(step_frame)

ax_reset = plt.axes([0.65, 0.02, 0.15, 0.05])
btn_reset = Button(ax_reset, 'Reset')
btn_reset.on_clicked(reset_sim)

#
# 
# main loop
#
#

def update(frame):
    global positions, velocities
    if is_playing:
        positions, velocities = compute_update(positions, velocities, slider_noise.val)
        agent_trail.append(positions[TARGET_AGENT_IDX].copy())
        
        collection.set_paths([Polygon(p) for p in get_agent_polygons(positions, velocities)])
        update_trail_graphics()
        
    return collection, trail_collection

frame_interval = int(1000 / PLAYBACK_FPS)
ani = FuncAnimation(fig, update, interval=frame_interval, blit=True, cache_frame_data=False)

plt.show()