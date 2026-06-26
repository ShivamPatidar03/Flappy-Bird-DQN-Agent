# Deep Q-Network (DQN) for Flappy Bird using PyTorch

[![Python](https://img.shields.io/badge/python-3.8%20%7C%203.9%20%7C%203.10%20%7C%203.11-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=flat&logo=PyTorch&logoColor=white)](https://pytorch.org/)
[![Gymnasium](https://img.shields.io/badge/Gymnasium-0.28.1-green.svg)](https://gymnasium.farama.org/)

An end-to-end Deep Reinforcement Learning project that trains an autonomous agent to play Flappy Bird using a Deep Q-Network (DQN) implemented in PyTorch. The agent learns optimal flight control policies solely by interacting with the environment and maximizing its cumulative reward.

---

## 🎮 Game Preview & Demo
The agent observes the spatial coordinates of the gaps between pipes, the bird's vertical velocity, and relative pipe alignments, mapping them directly to discrete actions: `0` (do nothing) or `1` (flap).

```
   State (12 Dimensions)                                Action (2 Actions)
┌──────────────────────┐                             ┌──────────────────────┐
│  Distance to Pipe    │    ┌───────────────────┐    │                      │
│  Pipe Gap Height     │ ──>│  Deep Q-Network   │───>│    [0] No Flap       │
│  Bird Y-Velocity     │    │  (policy_dqn)     │    │    [1] Flap          │
│  ... (12 features)   │    └───────────────────┘    │                      │
└──────────────────────┘                             └──────────────────────┘
```

---

## ✨ Features
* **Deep Q-Network (DQN):** Learns state-action value approximations mapping 12 env parameters to optimal flap actions.
* **Experience Replay Buffer:** Prevents temporal correlations and stabilizes SGD by training on randomly sampled mini-batches of historic experiences.
* **Target Network Syncing:** Implements a distinct Target DQN updated periodically to stabilize target Q-value estimations and prevent optimization feedback loops.
* **Epsilon-Greedy Exploration:** Gradually shifts from random exploration ($\epsilon = 1.0$) to exploitation ($\epsilon_{min} = 0.05$) via a decay factor of $0.9995$.
* **Robust Hardware Support:** Automatic GPU acceleration utilizing CUDA (for Nvidia GPUs) or MPS (for Apple Silicon metal acceleration) with CPU fallback.
* **Declarative Parameter Config:** Centralized configuration of reinforcement learning hyper-parameters in a YAML file.
* **Auto-Checkpointing:** Automatically saves the best-performing models to disk during the training run.

---

## 🧬 Reinforcement Learning Workflow
```
                       ┌──────────────────────────────┐
                       │            State             │
                       └──────────────┬───────────────┘
                                      │
                                      ▼
                       ┌──────────────────────────────┐
                       │    Policy Network (DQN)      │
                       └──────────────┬───────────────┘
                                      │
                                      ├──────────────────────────────┐
                        (Exploration: Random)            (Exploitation: argmax Q)
                                      │                              │
                                      ▼                              ▼
                       ┌──────────────────────────────┐              │
                       │            Action            │<─────────────┘
                       └──────────────┬───────────────┘
                                      │
                                      ▼
                       ┌──────────────────────────────┐
                       │  Environment (Flappy Bird)   │
                       └──────────────┬───────────────┘
                                      │
                                      ▼
                       ┌──────────────────────────────┐
                       │     Reward + Next State      │
                       └──────────────┬───────────────┘
                                      │
                                      ▼
                       ┌──────────────────────────────┐
                       │        Replay Memory         │
                       └──────────────┬───────────────┘
                                      │ (Sample Batch)
                                      ▼
                       ┌──────────────────────────────┐
                       │     Mini-Batch Training      │
                       └──────────────┬───────────────┘
                                      │ (Compute Target Q)
                                      ▼
                       ┌──────────────────────────────┐
                       │        Target Network        │
                       └──────────────┬───────────────┘
                                      │
                                      ▼
                       ┌──────────────────────────────┐
                       │       Mean Squared Error     │
                       └──────────────┬───────────────┘
                                      │
                                      ▼
                       ┌──────────────────────────────┐
                       │  Backpropagation Optimizer   │
                       └──────────────┬───────────────┘
                                      │
                                      ▼
                       ┌──────────────────────────────┐
                       │    Updated Policy Network    │
                       └──────────────────────────────┘
```

---

## 📐 Neural Network Architecture
The network is a Multilayer Perceptron (MLP) built dynamically using PyTorch's `nn.Sequential` module:

```
          Input Layer               Hidden Layer               Output Layer
       (State Dimension)            (ReLU Active)           (Action Dimension)
      ┌─────────────────┐        ┌─────────────────┐        ┌─────────────────┐
      │     X[0] Y      │        │     Node 1      │        │                 │
      │     X[1] Vel    │        │     Node 2      │        │                 │
      │     X[2] Pipe   │        │     Node 3      │        │                 │
      │        •        │ ──────>│        •        │ ──────>│   Q(s, "no-flap")
      │        •        │ [Linear]        •        │ [Linear]                 │
      │        •        │        │        •        │        │   Q(s, "flap")  │
      │        •        │        │    Node 256     │        │                 │
      └─────────────────┘        └─────────────────┘        └─────────────────┘
          [Dim: 12]                  [Dim: 256]                  [Dim: 2]
```

---

## 📂 Project Structure
* **`agent.py`:** Core agent execution system. Orchestrates training steps, action selection, optimization steps, logging, and evaluation playback.
* **`dqn.py`:** PyTorch implementation of the Deep Q-Network. Defined as a fully-connected feedforward network (`12 -> 256 -> 2`).
* **`experience_replay.py`:** Replay memory class implemented as a double-ended queue (FIFO buffer) supporting random batch sampling.
* **`parameters.yaml`:** Configuration parameters for training environments, epsilon schedules, target network synchronization rates, and learning rates.
* **`game_flappy_bird.py`:** Standard script allowing keyboard-controlled manual playthrough using PyGame bindings.
* **`requirements.txt`:** Core python dependencies for reproducing the environment.

---

## 🚀 Installation & Usage

### 1. Clone the Repository
```bash
git clone https://github.com/ShivamPatidar03/Flappy-Bird-DQN-Agent.git
cd Flappy-Bird-DQN-Agent
```

### 2. Set Up a Virtual Environment (Recommended)
```bash
# MacOS/Linux
python3 -m venv venv
source venv/bin/activate

# Windows
python -m venv venv
venv\Scripts\activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Play the Game Manually
Test your human performance using the keyboard (press Space to flap):
```bash
python game_flappy_bird.py
```

### 5. Train the DQN Agent
Run the training pipeline. Training progress is saved automatically under `runs/flappybirdv0.pt` and metrics logged in `runs/flappybirdv0.log`.
```bash
python agent.py flappybirdv0 --train
```

### 6. Evaluate the Trained Agent
Watch the trained DQN agent play the game by loading the cached model weights:
```bash
python agent.py flappybirdv0
```

---

## 🛠️ Future Improvements
To build upon this DQN baseline, several modern reinforcement learning enhancements could be implemented:
* **Double DQN (DDQN):** Mitigates Q-value overestimation bias by decoupled action selection and value evaluation.
* **Dueling DQN:** Decouples state value function $V(s)$ and action advantage function $A(s, a)$ within the network.
* **Prioritized Experience Replay (PER):** Selects experience tuples based on Temporal Difference error magnitude instead of uniform random sampling.
* **Rainbow DQN:** Combines DDQN, Dueling DQN, PER, Multi-step learning, Distributional RL, and Noisy Nets.
* **Policy Gradient Methods:** PPO (Proximal Policy Optimization), A2C (Advantage Actor-Critic), or SAC (Soft Actor-Critic) for policy-space search.

---

## 📚 Tech Stack
* **Language:** Python 3
* **Deep Learning Framework:** PyTorch
* **RL Frameworks:** Gymnasium, flappy-bird-gymnasium
* **Rendering Engine:** PyGame

---

## 🎓 Learning Outcomes
This project serves as a showcase for core Deep Reinforcement Learning concepts:
* **Function Approximation:** Using Deep Neural Networks to represent high-dimensional continuous state spaces.
* **Off-Policy Learning:** Updating target policy weights using actions and rewards collected from previous policies stored in the Replay Memory.
* **Temporal Difference Learning:** Updating estimations based on other active estimations (bootstrapping).
* **Gradient Optimization:** Implementation of Adam optimizer with backpropagation for parameter updates.

---

## 🤝 Acknowledgements
* The Farama Foundation for maintaining [Gymnasium](https://gymnasium.farama.org/).
* The authors of the [flappy-bird-gymnasium](https://github.com/TalSnauler/flappy-bird-gymnasium) wrapper.
* [PyTorch](https://pytorch.org/) development community.
