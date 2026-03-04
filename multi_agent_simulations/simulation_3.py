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

NUM_AGENTS = 10
ARENA_RADIUS = 10.0
AGENT_LENGTH = 0.5
PLAYBACK_FPS = 10

TRAIL_LENGTH = 20
TARGET_AGENT_IDX = 0  

INITIAL_W1 = 0.05
INITIAL_W2 = 0.05
INITIAL_W3 = 0.05

V_MIN = 0.25                                 # min velocity
V_MAX = 2.5                                   # max velocity

REPULSION_RADIUS = 1                        # max distance for repulsion
ALIGNMENT_RADIUS = 3                        # max distance for alignment
ATTRACTION_RADIUS = 5                       # max distance for attraction

def generate_initial_state():
    angles = np.random.uniform(0, 2 * np.pi, NUM_AGENTS)
    radii = np.random.uniform(0, ARENA_RADIUS, NUM_AGENTS)
    pos = np.column_stack((radii * np.cos(angles), radii * np.sin(angles)))
    vel = np.random.uniform(-0.5, 0.5, (NUM_AGENTS, 2))
    return pos, vel

positions, velocities = generate_initial_state()

agent_trail = deque(maxlen=TRAIL_LENGTH)
agent_trail.append(positions[TARGET_AGENT_IDX].copy())

#
# 
# position & velocity update model
#
#

def compute_update(pos, vel, w1, w2, w3):
    N = pos.shape[0]

    ### Calculate all distances between all agents
    diff = pos[np.newaxis, :, :] - pos[:, np.newaxis, :]
    dist = np.linalg.norm(diff, axis=2)
    np.fill_diagonal(dist, 1.0) # prevents division by 0
    
    unit_vectors = diff / dist[:, :, np.newaxis]
    for i in range(N):
        unit_vectors[i, i] = [0.0, 0.0]
        
    ### Attraction
    # Find agents within the attraction radius
    attraction_mask = dist < ATTRACTION_RADIUS
    np.fill_diagonal(attraction_mask, False) # ignore self

    # Count neighbors to find averages, default to 1 to prevent division by 0
    neighbor_counts_att = np.sum(attraction_mask, axis=1)
    counts_att = np.where(neighbor_counts_att > 0, neighbor_counts_att, 1)

    # sum positions of valid neighbors
    sum_pos = attraction_mask.astype(float) @ pos
    avg_pos = sum_pos / counts_att[:, np.newaxis]

    # Calculate steering force towards local center of mass
    F_attraction = np.zeros_like(pos)
    has_neighbors_att = neighbor_counts_att > 0
    F_attraction[has_neighbors_att] = avg_pos[has_neighbors_att] - pos[has_neighbors_att]
    
    ### Repulsion
    repulsion_mask = dist < REPULSION_RADIUS
    np.fill_diagonal(repulsion_mask, False) # ignore self
    # Scale by 1 / distance^2 (adding a tiny epsilon to prevent division by zero)
    dist_sq = dist**2 + 1e-8
    repulsion_vectors = -unit_vectors * (repulsion_mask[:, :, np.newaxis] / dist_sq[:, :, np.newaxis])
    F_repulsion = np.sum(repulsion_vectors, axis=1)
    
    ### Alignment
    # Find agents within the alignment radius
    alignment_mask = dist < ALIGNMENT_RADIUS
    np.fill_diagonal(alignment_mask, False) # ignore self

    # Count neighbors to find averages, default to 1 to prevent division by 0
    neighbor_counts = np.sum(alignment_mask, axis=1)
    counts = np.where(neighbor_counts > 0, neighbor_counts, 1)

    # sum velocities of valid neighbors
    sum_vel = alignment_mask.astype(float) @ vel
    avg_vel = sum_vel / counts[:, np.newaxis]
    
    # Calculate steering force only for agents that actually have neighbors
    F_alignment = np.zeros_like(vel)
    has_neighbors = neighbor_counts > 0
    F_alignment[has_neighbors] = avg_vel[has_neighbors] - vel[has_neighbors]
    
    ### Update velocity
    new_vel = vel + (w1 * F_attraction) + (w2 * F_repulsion) + (w3 * F_alignment)

    ### Limit to maximum and minimum velocity
    speeds = np.linalg.norm(new_vel, axis=1)
    
    # Prevent division by zero if an agent perfectly stops
    safe_speeds = np.where(speeds == 0, 1e-8, speeds)
    
    exceed_max = speeds > V_MAX
    drop_below_min = speeds < V_MIN
    
    if np.any(exceed_max):
        new_vel[exceed_max] = (new_vel[exceed_max] / safe_speeds[exceed_max, np.newaxis]) * V_MAX
        
    if np.any(drop_below_min):
        new_vel[drop_below_min] = (new_vel[drop_below_min] / safe_speeds[drop_below_min, np.newaxis]) * V_MIN
        
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
# Plots & gui
#
#

fig, ax = plt.subplots(figsize=(8, 8))
plt.subplots_adjust(bottom=0.3) 

ax.set_xlim(-ARENA_RADIUS * 1.5, ARENA_RADIUS * 1.5)
ax.set_ylim(-ARENA_RADIUS * 1.5, ARENA_RADIUS * 1.5)
ax.set_aspect('equal')
ax.set_title("Swarm Simulation")

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

# Shifted y-coordinates slightly to make room for the third slider
ax_w1 = plt.axes([0.15, 0.20, 0.65, 0.03])
ax_w2 = plt.axes([0.15, 0.15, 0.65, 0.03])
ax_w3 = plt.axes([0.15, 0.10, 0.65, 0.03])

slider_w1 = Slider(ax_w1, 'w1 (Attract)', 0.0, 0.5, valinit=INITIAL_W1)
slider_w2 = Slider(ax_w2, 'w2 (Repel)', 0.0, 0.5, valinit=INITIAL_W2)
slider_w3 = Slider(ax_w3, 'w3 (Align)', 0.0, 0.5, valinit=INITIAL_W3)

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
        # Pass slider_w3.val to the update function
        positions, velocities = compute_update(positions, velocities, slider_w1.val, slider_w2.val, slider_w3.val)
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
        # Pass slider_w3.val to the update function
        positions, velocities = compute_update(positions, velocities, slider_w1.val, slider_w2.val, slider_w3.val)
        agent_trail.append(positions[TARGET_AGENT_IDX].copy())
        
        collection.set_paths([Polygon(p) for p in get_agent_polygons(positions, velocities)])
        update_trail_graphics()
        
    return collection, trail_collection

frame_interval = int(1000 / PLAYBACK_FPS)
ani = FuncAnimation(fig, update, interval=frame_interval, blit=True, cache_frame_data=False)

plt.show()