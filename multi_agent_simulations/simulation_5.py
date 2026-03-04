import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.widgets import Slider, Button

#
# 
# Initialize arena, set parameters
#
#

ARENA_RADIUS = 10.0
PLAYBACK_FPS = 30
DT = 0.1  # time step

# Firefly sim parameters
INITIAL_AGENTS = 150
INITIAL_K = 2.0            # Kuramoto coupling strength
INITIAL_RADIUS = 3.0       # radius of influence (meters)
INITIAL_SPEED = 1.0        # flash phase speed multiplier
INITIAL_INFLUENCE = False  # turn on/off coupling

#
# 
# position & phase update model
#
#

def compute_update(pos, phase, omega, headings, K, R, speed_mult, influence_on):
    N = pos.shape[0]
    
    ### Base clock speed for this frame
    d_phase = omega * speed_mult
    
    ### If influence is on, apply the Kuramoto nudge
    if influence_on:
        diff_pos = pos[np.newaxis, :, :] - pos[:, np.newaxis, :]
        dist = np.linalg.norm(diff_pos, axis=2)
        
        # TODO: Implement the continuous Kuramoto phase nudge
        # Formula: nudge = (K / N_neighbors) * sum(sin(theta_j - theta_i))
        #
        # Conceptually, for each firefly:
        #   1. Find all neighbors within the interaction radius R
        #   2. Calculate the phase difference (theta_j - theta_i) for all pairs
        #   3. Apply np.sin() to these phase differences within radius R
        #   4. Count neighbors in radius (avoid dividing by zero)
        #   5. Multiply the sum by (K / neighbor_counts) and add it to d_phase

        d_phase += #result of 5 for each firefly, this should have a shape of (N,)
    
    new_phase = (phase + d_phase * DT) % (2 * np.pi)
    
    ### Correlated random walk
    headings += np.random.uniform(-0.2, 0.2, N)
    drift = 0.05 * np.column_stack((np.cos(headings), np.sin(headings)))
    new_pos = pos + drift
    
    ### If agent leaves boundry, wrap around
    dist_from_center = np.linalg.norm(new_pos, axis=1)
    out_of_bounds = dist_from_center > ARENA_RADIUS
    if np.any(out_of_bounds):
        direction = new_pos[out_of_bounds] / dist_from_center[out_of_bounds, np.newaxis]
        new_pos[out_of_bounds] = new_pos[out_of_bounds] - (2 * ARENA_RADIUS * direction)
        
    return new_pos, new_phase, headings

#
# 
# Set up agents and arena
#
#

def generate_initial_state(num_agents):
    # Random positions within the arena
    angles = np.random.uniform(0, 2 * np.pi, num_agents)
    radii = np.random.uniform(0, ARENA_RADIUS, num_agents)
    pos = np.column_stack((radii * np.cos(angles), radii * np.sin(angles)))
    
    # Random initial phases [0, 2pi)
    phases = np.random.uniform(0, 2 * np.pi, num_agents)
    omegas = np.random.normal(1.0, 0.2, num_agents)

    # Initial heading
    headings = np.random.uniform(0, 2 * np.pi, num_agents)
    
    return pos, phases, omegas, headings

# Global state variables
num_current_agents = INITIAL_AGENTS
positions, phases, omegas, headings = generate_initial_state(num_current_agents)

#
# 
# Plots & gui
#
#

plt.style.use('dark_background')

fig, ax = plt.subplots(figsize=(10, 11)) 
plt.subplots_adjust(bottom=0.35) 

ax.set_xlim(-ARENA_RADIUS * 1.05, ARENA_RADIUS * 1.05)
ax.set_ylim(-ARENA_RADIUS * 1.05, ARENA_RADIUS * 1.05)
ax.set_aspect('equal')
ax.axis('off')

arena = plt.Circle((0, 0), ARENA_RADIUS, color='#111111', fill=True, zorder=0)
ax.add_patch(arena)

scatter = ax.scatter(positions[:, 0], positions[:, 1], s=50, c='yellow', edgecolors='none', zorder=1)

def get_colors_and_sizes(phase, n_agents):
    brightness = (np.cos(phase) + 1) / 2
    brightness = brightness ** 8 
    
    colors = np.zeros((n_agents, 4))
    colors[:, 0] = 1.0  
    colors[:, 1] = 1.0  
    colors[:, 2] = 0.2  
    colors[:, 3] = brightness * 0.9 + 0.1 
    
    sizes = 20 + (brightness * 80)
    return colors, sizes

#
# 
# GUI controls
#
#

DARK_FACE = '#333333'
DARK_HOVER = '#555555'

ax_num   = plt.axes([0.15, 0.28, 0.65, 0.03], facecolor=DARK_FACE)
ax_speed = plt.axes([0.15, 0.22, 0.65, 0.03], facecolor=DARK_FACE)
ax_nudge = plt.axes([0.15, 0.16, 0.65, 0.03], facecolor=DARK_FACE)
ax_rad   = plt.axes([0.15, 0.10, 0.65, 0.03], facecolor=DARK_FACE)

slider_num   = Slider(ax_num,   '# Fireflies', 10, 500, valinit=INITIAL_AGENTS, valstep=1, color='yellow')
slider_speed = Slider(ax_speed, 'Clock Speed', 0.1, 3.0, valinit=INITIAL_SPEED, color='yellow')
slider_nudge = Slider(ax_nudge, 'Nudge (K)', 0.0, 5.0, valinit=INITIAL_K, color='yellow')
slider_rad   = Slider(ax_rad,   'Radius', 0.0, 20.0, valinit=INITIAL_RADIUS, color='yellow')

is_playing = True
influence_on = INITIAL_INFLUENCE

def toggle_play(event):
    global is_playing
    is_playing = not is_playing
    btn_play.label.set_text('Play' if not is_playing else 'Pause')
    fig.canvas.draw_idle()

def toggle_influence(event):
    global influence_on
    influence_on = not influence_on
    btn_inf.label.set_text(f"Influence: {'ON' if influence_on else 'OFF'}")
    btn_inf.color = '#555500' if influence_on else DARK_FACE
    fig.canvas.draw_idle()

def reset_sim(event):
    global positions, phases, omegas, headings, num_current_agents
    num_current_agents = int(slider_num.val)
    positions, phases, omegas, headings = generate_initial_state(num_current_agents)
    colors, sizes = get_colors_and_sizes(phases, num_current_agents)
    scatter.set_offsets(positions)
    scatter.set_facecolors(colors)
    scatter.set_sizes(sizes)
    fig.canvas.draw_idle()

ax_play  = plt.axes([0.20, 0.05, 0.15, 0.05])
ax_inf   = plt.axes([0.40, 0.05, 0.20, 0.05])
ax_reset = plt.axes([0.65, 0.05, 0.15, 0.05])

btn_play  = Button(ax_play, 'Pause', color=DARK_FACE, hovercolor=DARK_HOVER)

btn_inf_text = f"Influence: {'ON' if INITIAL_INFLUENCE else 'OFF'}"
btn_inf_color = '#555500' if INITIAL_INFLUENCE else DARK_FACE
btn_inf   = Button(ax_inf, btn_inf_text, color=btn_inf_color, hovercolor=DARK_HOVER)

btn_reset = Button(ax_reset, 'Reset', color=DARK_FACE, hovercolor=DARK_HOVER)

btn_play.label.set_color('white')
btn_inf.label.set_color('white')
btn_reset.label.set_color('white')

btn_play.on_clicked(toggle_play)
btn_inf.on_clicked(toggle_influence)
btn_reset.on_clicked(reset_sim)

def update_agent_count(val):
    reset_sim(None)
slider_num.on_changed(update_agent_count)

#
# 
# main loop
#
#

def update(frame):
    global positions, phases, headings, num_current_agents
    if is_playing:
        positions, phases, headings = compute_update(
            positions, phases, omegas, headings,
            slider_nudge.val, slider_rad.val, slider_speed.val, influence_on
        )
        
        colors, sizes = get_colors_and_sizes(phases, num_current_agents)
        scatter.set_offsets(positions)
        scatter.set_facecolors(colors)
        scatter.set_sizes(sizes)
        
    return scatter,

frame_interval = int(1000 / PLAYBACK_FPS)
ani = FuncAnimation(fig, update, interval=frame_interval, blit=False, cache_frame_data=False)

plt.show()