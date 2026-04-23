import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np
from torch.nn.utils import parameters_to_vector, vector_to_parameters
from scipy.optimize import linear_sum_assignment
from tqdm import tqdm
import copy
from sklearn.decomposition import PCA


# ==========================================
# CONFIGURATION
# ==========================================
torch.manual_seed(42)

# Number of input dimensions (>= 5 recommended)
INPUT_DIM = 50        # You can set this to 5, 10, 20, 50, 100, etc.
TRAIN_SAMPLES = 5000
TEST_SAMPLES = 2000

# Noise level for training labels
TRAIN_NOISE_STD = 0.1


# ======================================================
# 1. Dataset Creation (Friedman #1) and Model Definition
# ======================================================

def friedman1_function(X):
    """
    X: tensor of shape (N, D)
    Only the first 5 dimensions matter for the true function.
    Extra dimensions are ignored (pure noise features).
    """
    x1, x2, x3, x4, x5 = X[:,0], X[:,1], X[:,2], X[:,3], X[:,4]
    y = (
        10 * torch.sin(np.pi * x1 * x2)
        + 20 * (x3 - 0.5)**2
        + 10 * x4
        + 5 * x5
    )
    return y.unsqueeze(1)


# Training data
X_train = torch.rand(TRAIN_SAMPLES, INPUT_DIM)
y_train = friedman1_function(X_train) + TRAIN_NOISE_STD * torch.randn(TRAIN_SAMPLES, 1)

# Test data (clean)
X_test = torch.rand(TEST_SAMPLES, INPUT_DIM)
y_test = friedman1_function(X_test)


def get_loss(model, X, y):
    pred = model(X)
    return nn.MSELoss()(pred, y)


class TinyNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(INPUT_DIM, 64)
        self.act = nn.Tanh()
        self.fc2 = nn.Linear(64, 1)

    def forward(self, x):
        return self.fc2(self.act(self.fc1(x)))


# ==========================================
# 2. Alignment Logic 
# ==========================================
def align_networks(net_reference, net_to_align):
    W1_target = net_reference.fc1.weight.data
    W1_align = net_to_align.fc1.weight.data
    
    cost_matrix = torch.cdist(W1_target, W1_align, p=2).cpu().numpy()
    row_ind, col_ind = linear_sum_assignment(cost_matrix)
    
    net_to_align.fc1.weight.data = net_to_align.fc1.weight.data[col_ind, :]
    net_to_align.fc1.bias.data = net_to_align.fc1.bias.data[col_ind]
    net_to_align.fc2.weight.data = net_to_align.fc2.weight.data[:, col_ind]

# ==========================================
# 3. Training Logic (A/B Test)
# ==========================================
num_agents = 3
epochs_explore = 150
epochs_phase2 = 150

# --- Phase 1: Shared Initial Exploration ---
print("Phase 1: Shared Independent Exploration...")
shared_agents = [TinyNet() for _ in range(num_agents)]
for i, agent in enumerate(shared_agents):
    with torch.no_grad():
        for param in agent.parameters():
            param.data = torch.randn_like(param) * (2.0 + i) # Scatter

optimizers = [optim.Adam(agent.parameters(), lr=0.03) for agent in shared_agents]
shared_trajectories = [[] for _ in range(num_agents)]

for epoch in tqdm(range(epochs_explore), desc="Exploring"):
    for i, agent in enumerate(shared_agents):
        optimizers[i].zero_grad()
        loss = get_loss(agent, X_train, y_train)
        loss.backward()
        optimizers[i].step()
        shared_trajectories[i].append(parameters_to_vector(agent.parameters()).detach().clone())

# Align them once at the end of Phase 1 so both groups start with same permutation frame
for i in range(1, num_agents):
    align_networks(shared_agents[0], shared_agents[i])

# Split into A/B groups
baseline_agents = copy.deepcopy(shared_agents)
swarm_agents = copy.deepcopy(shared_agents)

baseline_trajectories = copy.deepcopy(shared_trajectories)
swarm_trajectories = copy.deepcopy(shared_trajectories)

# Re-init optimizers to reset momentum states for a fair Phase 2 start
opt_baseline = [optim.Adam(agent.parameters(), lr=0.03) for agent in baseline_agents]
opt_swarm = [optim.Adam(agent.parameters(), lr=0.03) for agent in swarm_agents]

# --- Phase 2: BASELINE (Independent) ---
print("\nPhase 2: Training BASELINE (No Swarm Forces)...")
for epoch in tqdm(range(epochs_phase2), desc="Baseline Phase 2"):
    for i, agent in enumerate(baseline_agents):
        opt_baseline[i].zero_grad()
        loss = get_loss(agent, X_train, y_train)
        loss.backward()
        opt_baseline[i].step()
        baseline_trajectories[i].append(parameters_to_vector(agent.parameters()).detach().clone())

# --- Phase 2: SWARM (Attraction to Center of Mass) ---
print("\nPhase 2: Training SWARM (Attraction to CoM)...")
attraction_strength = 0.05
for epoch in tqdm(range(epochs_phase2), desc="Swarm Phase 2"):
    with torch.no_grad():
        all_params = torch.stack([parameters_to_vector(a.parameters()) for a in swarm_agents])
        center_of_mass = torch.mean(all_params, dim=0)
        
    for i, agent in enumerate(swarm_agents):
        opt_swarm[i].zero_grad()
        loss = get_loss(agent, X_train, y_train)
        loss.backward()
        opt_swarm[i].step()
        
        with torch.no_grad():
            agent_params = parameters_to_vector(agent.parameters())
            force = attraction_strength * (center_of_mass - agent_params)
            vector_to_parameters(agent_params + force, agent.parameters())
            
        swarm_trajectories[i].append(parameters_to_vector(agent.parameters()).detach().clone())

# ==========================================
# 4. Evaluation Report Calculation
# ==========================================
def evaluate_group(agents, name):
    with torch.no_grad():
        # 1. Best Individual
        losses = [get_loss(a, X_test, y_test).item() for a in agents]
        best_indiv = min(losses)
        mean_indiv = sum(losses) / len(losses)
        
        # 2. Ensemble Output (Average of predictions)
        preds = torch.stack([a(X_test) for a in agents])
        ensemble_pred = preds.mean(dim=0)
        ensemble_loss = nn.MSELoss()(ensemble_pred, y_test).item()
        
        # 3. Center of Mass (Average of weights)
        com_params = torch.mean(torch.stack([parameters_to_vector(a.parameters()) for a in agents]), dim=0)
        com_net = TinyNet()
        vector_to_parameters(com_params, com_net.parameters())
        com_loss = get_loss(com_net, X_test, y_test).item()
        
        # 4. Swarm Cohesion (Spread)
        spreads = [torch.norm(parameters_to_vector(a.parameters()) - com_params).item() for a in agents]
        avg_spread = sum(spreads)/len(spreads)
        
    return best_indiv, mean_indiv, ensemble_loss, com_loss, avg_spread

b_best, b_mean, b_ens, b_com, b_spread = evaluate_group(baseline_agents, "Baseline")
s_best, s_mean, s_ens, s_com, s_spread = evaluate_group(swarm_agents, "Swarm")

print("\n" + "="*50)
print("             EVALUATION REPORT (TEST MSE)             ")
print("="*50)
print(f"Metric                     | Baseline    | Swarm      ")
print(f"---------------------------|-------------|------------")
print(f"Mean Individual Loss       | {b_mean:.4f}      | {s_mean:.4f}")
print(f"Best Individual Loss       | {b_best:.4f}      | {s_best:.4f}")
print(f"Ensemble Loss (Avg Output) | {b_ens:.4f}      | {s_ens:.4f}")
print(f"Center of Mass Loss        | {b_com:.4f}      | {s_com:.4f}")
print(f"Cohesion (Avg L2 Spread)   | {b_spread:.4f}      | {s_spread:.4f}")
print("="*50)

# ==========================================
# Write Evaluation Report to File
# ==========================================
report_lines = [
    "==================================================\n",
    "             EVALUATION REPORT (TEST MSE)\n",
    "==================================================\n",
    f"Mean Individual Loss (Baseline): {b_mean:.6f}\n",
    f"Mean Individual Loss (Swarm):    {s_mean:.6f}\n",
    f"Best Individual Loss (Baseline): {b_best:.6f}\n",
    f"Best Individual Loss (Swarm):    {s_best:.6f}\n",
    f"Ensemble Loss (Baseline):        {b_ens:.6f}\n",
    f"Ensemble Loss (Swarm):           {s_ens:.6f}\n",
    f"Center of Mass Loss (Baseline):  {b_com:.6f}\n",
    f"Center of Mass Loss (Swarm):     {s_com:.6f}\n",
    f"Cohesion Spread (Baseline):      {b_spread:.6f}\n",
    f"Cohesion Spread (Swarm):         {s_spread:.6f}\n",
    "==================================================\n"
]

with open("evaluation_report.txt", "w") as f:
    f.writelines(report_lines)

print("\nEvaluation report written to evaluation_report.txt")

# ==========================================
# 5. PCA-Based Loss Landscape Projection
# ==========================================
# Collect all parameter vectors from both groups
all_params = []
for traj_group in [baseline_trajectories, swarm_trajectories]:
    for i in range(num_agents):
        for p in traj_group[i]:
            all_params.append(p.cpu().numpy())

all_params = np.stack(all_params)

# Fit PCA on the full trajectory set
pca = PCA(n_components=2)
pca.fit(all_params)

# PCA basis vectors in parameter space
u = torch.tensor(pca.components_[0], dtype=torch.float32)
v = torch.tensor(pca.components_[1], dtype=torch.float32)

# Center of projection = final swarm center of mass
final_com_swarm = torch.mean(torch.stack([swarm_trajectories[i][-1] for i in range(num_agents)]), dim=0)

def project(vec):
    dv = vec - final_com_swarm
    return torch.dot(dv, u).item(), torch.dot(dv, v).item()

# Determine grid boundaries
all_x, all_y = [], []
for traj_group in [baseline_trajectories, swarm_trajectories]:
    for i in range(num_agents):
        for p in traj_group[i]:
            px, py = project(p)
            all_x.append(px)
            all_y.append(py)

mx = (max(all_x) - min(all_x)) * 0.1
my = (max(all_y) - min(all_y)) * 0.1

grid_size = 20
X_grid, Y_grid = np.meshgrid(
    np.linspace(min(all_x)-mx, max(all_x)+mx, grid_size),
    np.linspace(min(all_y)-my, max(all_y)+my, grid_size)
)

Z_loss = np.zeros((grid_size, grid_size))
dummy_net = TinyNet()

print("\nRendering PCA-Based Loss Landscape...")
for i in tqdm(range(grid_size), desc="Calculating Grid"):
    for j in range(grid_size):
        w = final_com_swarm + X_grid[i,j] * u + Y_grid[i,j] * v
        vector_to_parameters(w, dummy_net.parameters())
        with torch.no_grad():
            Z_loss[i,j] = get_loss(dummy_net, X_train, y_train).item()


# Create a shared projection function
def project_shared(vec):
    return torch.dot(vec - final_com_swarm, u).item(), torch.dot(vec - final_com_swarm, v).item()

# 2. Determine Grid Boundaries using ALL trajectories to ensure everything fits
all_x, all_y = [], []
for traj_group in [baseline_trajectories, swarm_trajectories]:
    for i in range(3):
        for point in traj_group[i][epochs_explore:]:
            px, py = project_shared(point)
            all_x.append(px)
            all_y.append(py)

mx, my = (max(all_x) - min(all_x)) * 0.1, (max(all_y) - min(all_y)) * 0.1
grid_size = 10 # Increased slightly for smoother shared background
X_grid, Y_grid = np.meshgrid(np.linspace(min(all_x)-mx, max(all_x)+mx, grid_size), 
                             np.linspace(min(all_y)-my, max(all_y)+my, grid_size))
Z_loss = np.zeros((grid_size, grid_size))
dummy_net = TinyNet()

print("\nRendering Shared Loss Landscape...")
for i in tqdm(range(grid_size), desc="Calculating Shared Grid"):
    for j in range(grid_size):
        w_grid = final_com_swarm + X_grid[i,j] * u + Y_grid[i,j] * v
        vector_to_parameters(w_grid, dummy_net.parameters())
        with torch.no_grad():
            Z_loss[i,j] = get_loss(dummy_net, X_train, y_train).item()

# 3. Plotting
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 8))
colors = ['red', 'cyan', 'magenta']
labels = ['Agent 0', 'Agent 1', 'Agent 2']

def draw_plot(ax, trajectories, title, is_swarm):
    vmax = np.percentile(Z_loss, 95) 
    levels = np.logspace(np.log10(Z_loss.min()), np.log10(vmax), 40)
    # Both axes now use the exact same X_grid, Y_grid, and Z_loss!
    contour = ax.contourf(X_grid, Y_grid, np.clip(Z_loss, a_min=None, a_max=vmax), 
                           levels=levels, cmap='viridis', alpha=0.9, extend='max')
    
    for i in range(3):
        traj = trajectories[i][epochs_explore:] 
        x_coords, y_coords = zip(*[project_shared(p) for p in traj])
        ax.plot(x_coords, y_coords, color=colors[i], label=labels[i], linewidth=2, marker='.', markersize=3)
        ax.scatter(x_coords[0], y_coords[0], color='white', edgecolors='black', s=80, zorder=5, marker='s')
        ax.scatter(x_coords[-1], y_coords[-1], color=colors[i], edgecolors='black', s=150, zorder=5, marker='*')
        
    # Calculate group-specific Center of Mass for the yellow X
    group_com = torch.mean(torch.stack([trajectories[i][-1] for i in range(3)]), dim=0)
    com_x, com_y = project_shared(group_com)
    ax.scatter(com_x, com_y, color='yellow', edgecolors='black', s=100, zorder=6, marker='X', label="Center of Mass")
    
    ax.set_title(title)
    ax.set_xlabel("Subspace Vector U")
    ax.set_ylabel("Subspace Vector V")
    return contour

draw_plot(ax1, baseline_trajectories, "BASELINE (No Flocking Forces)\nAgents train independently", False)
contour2 = draw_plot(ax2, swarm_trajectories, "Flocking DYNAMICS\nAgents pulled toward Center of Mass", True)

fig.colorbar(contour2, ax=[ax1, ax2], label='MSE Loss (Log Scale)', fraction=0.02, pad=0.04)
ax1.legend(loc='upper left')
plt.suptitle("A/B Test: Multi-Agent Mode Connectivity on a Shared Subspace Projection", fontsize=16, fontweight='bold')
plt.savefig("multi_agent_loss_landscape.png", dpi=300)
plt.show()