## DTN_GNN: A Digital Twin Network and Graph Neural Network-Based Edge Computing System

### Project Overview

DTN_GNN combines Digital Twin Network (DTN) and Graph Neural Network (GNN) technologies to build an intelligent edge
computing decision-making system. The system mainly addresses the computation task allocation problem in vehicular
networks by utilizing GNN and reinforcement learning technologies to achieve efficient computation resource scheduling.

### Technical Background

* **Digital Twin Network (DTN)**

Digital twins are virtual representations of physical entities in the digital world. In this project, DTN is used to
simulate and predict the behavior of vehicular networks, providing an accurate environmental model for decision-making.

* **Graph Neural Network (GNN)**

GNN is a deep learning model specialized in processing graph-structured data. It processes network topology information,
learns the complex relationships between nodes, and supports resource allocation decisions.

* **Edge Computing**
  Edge computing moves computational resources from centralized clouds to the network edge, reducing latency and
  improving efficiency. This project simulates the task offloading decision-making process between edge servers and
  vehicles.

### Core Module Introduction

1. **Digital Twin Module (Digital_twin.py)**
    - Creates a virtual replica of the physical network.
    - Real-time synchronization and simulation of physical system states.
    - Predicts potential outcomes of different decisions.
    - Provides a virtual environment for reinforcement learning.

2. **Graph Neural Network Module (GNN.py)**
    - Processes network topology information.
    - Extracts node features and relationships.
    - Learns the network state representation.
    - Supports function approximation for decision-making.

3. **Vehicle Module (Vehicle.py)**
    - Simulates vehicle movement and computational capability.
    - Generates computation tasks.
    - Manages local computation resources.
    - Interacts with edge servers.

4. **Edge Server Module (EdgeServer.py)**
    - Simulates edge server computational resources.
    - Processes computation tasks from vehicles.
    - Manages resource allocation.
    - Tracks service quality metrics.

5. **Task Module (Task.py)**
    - Defines computation task attributes (size, complexity, deadline).
    - Tracks task processing status.
    - Calculates task completion time and energy consumption.

6. **Network Module (Network.py)**
    - Builds network topology.
    - Manages node connections.
    - Simulates network delay and bandwidth limitations.
    - Updates network dynamics.

7. **Communication Module (Comm.py)**
    - Simulates data transmission process.
    - Calculates communication delay and energy consumption.
    - Handles data packet loss and retransmission.
    - Simulates communication quality changes.

8. **Reinforcement Learning Agent (agent.py)**
    - A policy network based on GNN.
    - Decides task processing methods (local computation or edge offloading).
    - Learns to optimize long-term system performance.
    - Adapts to dynamic environmental changes.

9. **Environment Simulation (environment.py)**
    - Integrates the interactions of various components.
    - Implements state transition logic.
    - Computes reward signals.
    - Provides a standard reinforcement learning interface.

### Workflow

1. **Initialization Stage:**
    - Creates vehicle and edge server networks.
    - Initializes the digital twin system.
    - Loads GNN model parameters.

2. **Task Generation Stage:**
    - Vehicles generate computation tasks based on preset distributions.
    - System records task characteristics (computation requirements, deadlines).

3. **Decision-Making Stage:**
    - GNN processes the current network state.
    - The reinforcement learning agent selects processing methods for each task:
        - Local computation: uses the vehicle’s own resources.
        - Edge offloading: sends the task to nearby edge servers.

4. **Execution Stage:**
    - Executes task processing based on the decision.
    - The digital twin system simulates the processing process.
    - Records task completion time, energy consumption, and resource utilization.

5. **Learning Stage:**
    - Calculates system performance metrics as rewards.
    - Updates GNN model parameters using experience replay.
    - Optimizes decision-making strategies.

### Performance Metrics

The system uses the following metrics to evaluate performance:

- **Task Delay:** The difference between task completion time and generation time.
- **Energy Consumption:** Energy consumed during computation and communication processes.
- **Task Completion Rate:** The percentage of tasks successfully completed.
- **Resource Utilization:** The efficiency of using computational resources.
- **System Throughput:** The number of tasks processed per unit of time.

### User Guide

**System Requirements:**

- Python 3.8+
- PyTorch 1.7+
- NumPy
- Matplotlib (for visualization)

### Experimental Results and Analysis

The experiments demonstrate that the decision-making system combining digital twin and GNN outperforms traditional
methods by:

- Reducing average task completion time.
- Lowering overall system energy consumption.
- Increasing computational resource utilization.
- Adapting to network topology changes and traffic fluctuations.

### Parameter Configuration

You can adjust the following configurations by modifying the parameters in `main.py`:

- Number and distribution of vehicles.
- Location and capabilities of edge servers.
- Task generation frequency and computation requirements.
- GNN model structure and learning parameters.
- Simulation duration and environmental dynamics.
