import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from torch.nn.utils import parameters_to_vector, vector_to_parameters
from tqdm import tqdm
import math

# ==========================================
# CONFIGURATION
# ==========================================
CONFIGURATION = {
    "random_seed": 42,
    "number_of_agents": 10,
    "number_of_epochs": 1500,
    "learning_rate": 0.05,
    
    # Dataset Parameters
    "number_of_data_points": 500,
    "spiral_noise": 0.15,
    
    # Swarm / Flocking Parameters
    # Radii are large because high-dimensional weight spaces have large L2 norms
    "attraction_radius": 20.0,
    "repulsion_radius": 5.0,
    "attraction_weight": 0.01,
    "repulsion_weight": 0.05,
    
    # Network Architecture
    "hidden_layer_size": 16
}

# Set random seeds for reproducibility
torch.manual_seed(CONFIGURATION["random_seed"])
np.random.seed(CONFIGURATION["random_seed"])

# ==========================================
# DATASET GENERATOR
# ==========================================
class SpiralDataset:
    """Generates a two-class interleaved spiral dataset."""
    def __init__(self, number_of_points: int, noise: float):
        self.number_of_points = number_of_points
        self.noise = noise
        self.features, self.labels = self._generate_data()
        
    def _generate_data(self):
        points_per_class = self.number_of_points // 2
        features = np.zeros((self.number_of_points, 2))
        labels = np.zeros(self.number_of_points, dtype=np.int64)
        
        for class_index in range(2):
            index_range = range(points_per_class * class_index, points_per_class * (class_index + 1))
            radius = np.linspace(0.0, 1.0, points_per_class)
            # Offset the angle for the second class to interleave them
            angle = np.linspace(class_index * math.pi, (class_index + 2) * math.pi, points_per_class) + np.random.randn(points_per_class) * self.noise
            
            features[index_range] = np.c_[radius * np.sin(angle), radius * np.cos(angle)]
            labels[index_range] = class_index
            
        return torch.FloatTensor(features), torch.LongTensor(labels)

# ==========================================
# NEURAL NETWORK ARCHITECTURE
# ==========================================
class SmallMultiLayerPerceptron(nn.Module):
    """A small neural network that will act as an individual agent."""
    def __init__(self, hidden_size: int):
        super().__init__()
        # Small capacity ensures a single model struggles with the complex spirals
        self.network = nn.Sequential(
            nn.Linear(2, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 2) # 2 output classes
        )

    def forward(self, input_features):
        return self.network(input_features)

# ==========================================
# TRAINING CONTROLLERS
# ==========================================
class IndependentEnsembleTrainer:
    """Trains a collection of models entirely independently."""
    def __init__(self, number_of_agents: int, hidden_size: int, learning_rate: float):
        self.agents = [SmallMultiLayerPerceptron(hidden_size) for _ in range(number_of_agents)]
        self.optimizers = [optim.Adam(agent.parameters(), lr=learning_rate) for agent in self.agents]
        self.loss_function = nn.CrossEntropyLoss()

    def train(self, features: torch.Tensor, labels: torch.Tensor, epochs: int):
        for epoch in tqdm(range(epochs), desc="Training Independent Ensemble"):
            for index, agent in enumerate(self.agents):
                self.optimizers[index].zero_grad()
                predictions = agent(features)
                loss = self.loss_function(predictions, labels)
                loss.backward()
                self.optimizers[index].step()

    def get_ensemble_predictions(self, features: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            probabilities = torch.stack([torch.softmax(agent(features), dim=1) for agent in self.agents])
            # Average the probabilities across all independent agents
            mean_probabilities = probabilities.mean(dim=0)
            return torch.argmax(mean_probabilities, dim=1)


class SwarmEnsembleTrainer:
    """Trains a collection of models that influence each other via spatial flocking forces."""
    def __init__(self, number_of_agents: int, hidden_size: int, learning_rate: float, config: dict):
        self.agents = [SmallMultiLayerPerceptron(hidden_size) for _ in range(number_of_agents)]
        self.optimizers = [optim.Adam(agent.parameters(), lr=learning_rate) for agent in self.agents]
        self.loss_function = nn.CrossEntropyLoss()
        self.config = config

    def _apply_flocking_forces(self):
        """Calculates and applies dimension-wise attraction and repulsion forces."""
        with torch.no_grad():
            # Extract flattened parameter vectors for all agents. Shape: (Number of Agents, Number of Parameters)
            agent_positions = torch.stack([parameters_to_vector(agent.parameters()) for agent in self.agents])
            number_of_agents = agent_positions.shape[0]
            
            # Vectorized pairwise differences: target - source
            # Shape: (Agents, Agents, Parameters)
            differences = agent_positions.unsqueeze(1) - agent_positions.unsqueeze(0)
            
            # Euclidean distance between models in parameter space. Shape: (Agents, Agents)
            distances = torch.norm(differences, dim=2)
            
            # Prevent division by zero for self-distance
            safe_distances = distances.clone()
            safe_distances[safe_distances == 0] = 1.0 
            
            # Unit vectors pointing from agent i to agent j. Shape: (Agents, Agents, Parameters)
            unit_vectors = differences / safe_distances.unsqueeze(2)
            
            # --- ATTRACTION ---
            # Mask out agents beyond the radius or identical agents (distance == 0)
            attraction_mask = (distances < self.config["attraction_radius"]) & (distances > 0)
            neighbor_counts = attraction_mask.sum(dim=1).unsqueeze(1).clamp(min=1)
            
            # Sum the vector differences of valid neighbors and divide by count
            attraction_forces = (differences * attraction_mask.unsqueeze(2)).sum(dim=1) / neighbor_counts
            
            # --- REPULSION ---
            repulsion_mask = (distances < self.config["repulsion_radius"]) & (distances > 0)
            # Repulsion scales via Inverse Square Law: 1 / distance^2
            inverse_square_weights = repulsion_mask.float() / (distances ** 2 + 1e-8)
            
            # Apply weights to unit vectors and negate (push away)
            repulsion_forces = - (unit_vectors * inverse_square_weights.unsqueeze(2)).sum(dim=1)
            
            # --- APPLY COMBINED FORCES ---
            total_forces = (self.config["attraction_weight"] * attraction_forces) + \
                           (self.config["repulsion_weight"] * repulsion_forces)
            
            for index, agent in enumerate(self.agents):
                current_parameters = parameters_to_vector(agent.parameters())
                updated_parameters = current_parameters + total_forces[index]
                vector_to_parameters(updated_parameters, agent.parameters())

    def train(self, features: torch.Tensor, labels: torch.Tensor, epochs: int):
        for epoch in tqdm(range(epochs), desc="Training Swarm Ensemble"):
            # 1. Standard Gradient Descent Step
            for index, agent in enumerate(self.agents):
                self.optimizers[index].zero_grad()
                predictions = agent(features)
                loss = self.loss_function(predictions, labels)
                loss.backward()
                self.optimizers[index].step()
            
            # 2. Apply Weight-Space Flocking Physics
            self._apply_flocking_forces()

    def get_ensemble_predictions(self, features: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            probabilities = torch.stack([torch.softmax(agent(features), dim=1) for agent in self.agents])
            mean_probabilities = probabilities.mean(dim=0)
            return torch.argmax(mean_probabilities, dim=1)

# ==========================================
# VISUALIZATION AND EXECUTION
# ==========================================
def plot_decision_boundaries(dataset, independent_predictions, swarm_predictions):
    """Plots the generated dataset against the decision surfaces of both ensembles."""
    x_min, x_max = dataset.features[:, 0].min() - 0.5, dataset.features[:, 0].max() + 0.5
    y_min, y_max = dataset.features[:, 1].min() - 0.5, dataset.features[:, 1].max() + 0.5
    
    # Create a dense grid to map the decision boundaries
    grid_resolution = 0.02
    xx, yy = np.meshgrid(np.arange(x_min, x_max, grid_resolution),
                         np.arange(y_min, y_max, grid_resolution))
    grid_tensor = torch.FloatTensor(np.c_[xx.ravel(), yy.ravel()])
    
    # Get grid predictions
    z_independent = independent_trainer.get_ensemble_predictions(grid_tensor).reshape(xx.shape).numpy()
    z_swarm = swarm_trainer.get_ensemble_predictions(grid_tensor).reshape(xx.shape).numpy()

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))
    
    # Plot Independent Ensemble
    ax1.contourf(xx, yy, z_independent, alpha=0.3, cmap=plt.cm.RdBu)
    ax1.scatter(dataset.features[:, 0], dataset.features[:, 1], c=dataset.labels, edgecolors='k', cmap=plt.cm.RdBu)
    ax1.set_title("Independent Ensemble Boundary\n(Lack of Coordination)")
    
    # Plot Swarm Ensemble
    ax2.contourf(xx, yy, z_swarm, alpha=0.3, cmap=plt.cm.RdBu)
    ax2.scatter(dataset.features[:, 0], dataset.features[:, 1], c=dataset.labels, edgecolors='k', cmap=plt.cm.RdBu)
    ax2.set_title("Swarm Ensemble Boundary\n(Collaborative Problem Partitioning)")
    
    plt.suptitle("Interleaved Spirals: Independent vs. Flocking Ensembles", fontsize=16)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    print("Generating Interleaved Spirals Dataset...")
    spirals_data = SpiralDataset(CONFIGURATION["number_of_data_points"], CONFIGURATION["spiral_noise"])
    
    # Initialize Trainers
    independent_trainer = IndependentEnsembleTrainer(
        CONFIGURATION["number_of_agents"], 
        CONFIGURATION["hidden_layer_size"], 
        CONFIGURATION["learning_rate"]
    )
    
    swarm_trainer = SwarmEnsembleTrainer(
        CONFIGURATION["number_of_agents"], 
        CONFIGURATION["hidden_layer_size"], 
        CONFIGURATION["learning_rate"],
        CONFIGURATION
    )
    
    # Execute Training
    independent_trainer.train(spirals_data.features, spirals_data.labels, CONFIGURATION["number_of_epochs"])
    swarm_trainer.train(spirals_data.features, spirals_data.labels, CONFIGURATION["number_of_epochs"])
    
    # Render Results
    print("Rendering Decision Boundaries...")
    plot_decision_boundaries(spirals_data, independent_trainer, swarm_trainer)