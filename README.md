# Learning Plasticity

Learning Plasticity is a comprehensive project focused on exploring neural encoding mechanisms and synaptic plasticity models. The repository is divided into two main parts: the implementation of various neural encoding schemes and the study of synaptic plasticity through both STDP and R-STDP models. This project aims to provide a deep understanding of how neurons encode information and adapt their synaptic strengths in response to different stimuli and learning paradigms.

## Features

### Part 1: Neural Encoding Schemes

#### 1. Time-To-First-Spike Encoding
- **Description**: Encodes information based on the timing of the first spike of a neuron.
- **Implementation**: Provides functions to convert input signals into spike times.

#### 2. Poisson Encoding
- **Description**: Uses a Poisson process to encode the firing rate of neurons.
- **Implementation**: Generates spike trains based on Poisson statistics.

#### 3. Numerical Values Encoding
- **Description**: Encodes numerical values directly into the firing rates or patterns of spikes.
- **Implementation**: Converts numerical inputs into spike trains or firing rates suitable for neural network input.

### Part 2: Synaptic Plasticity Models

#### 1. Synaptic Time-Dependent Plasticity (STDP)
- **Mechanism**: Models synaptic changes based on the relative timing of pre- and post-synaptic spikes.
- **Implementation**: Simulates STDP with various input patterns and analyzes the resulting synaptic adjustments.

#### 2. Reinforcement Learning-based Reward-Modulated STDP (R-STDP)
- **Mechanism**: Combines STDP with reinforcement learning principles, modulating synaptic changes based on reward signals.
- **Implementation**: Implements R-STDP with different reward structures and input patterns, providing tools for in-depth analysis of learning performance and synaptic dynamics.

## Getting Started

### Prerequisites
- Python 3.7+
- Numpy
- Matplotlib
- Pymonntorch

### Installation
Clone the repository:
```bash
git clone https://github.com/yourusername/LearningPlasticity.git
```
Navigate to the project directory and install dependencies:
```bash
cd LearningPlasticity
pip install -r requirements.txt
```

### Usage
0. **All Together** : Explore all the features of these networks and play around with paramters to understand it better as a whole (**Highly Recommend**) - other method not completely functional (just to understand the structure and the code better, for experimental purposes use the Notebook)
   ```bash
   CNS-P03-Notebook.ipynb
   ```

2. **Neural Encoding Schemes**: Explore different encoding mechanisms using provided scripts.
   ```bash
   python neural_encoding.py
   ```

3. **STDP and R-STDP Models**: Simulate and analyze synaptic plasticity.
   ```bash
   python stdp_simulation.py
   python r_stdp_simulation.py
   ```

## Contributing and Stars
We welcome contributions! Please fork and star the repository, create a new branch for your features or bug fixes, and submit a pull request. Make sure your code follows our style guidelines and is well-documented.

---

Learning Plasticity aims to serve as a valuable tool for researchers and enthusiasts in computational neuroscience and machine learning, providing insights into neural encoding and adaptive learning mechanisms. Feel free to explore, contribute, and advance the field of neural plasticity!
