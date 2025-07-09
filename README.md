# Lambda³ Fluid Simulation

[![MIT License](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![NumPy](https://img.shields.io/badge/numpy-%23013243.svg?style=flat&logo=numpy&logoColor=white)](https://numpy.org/)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/12tUo7u9wBRPjeGwgws4v_8_McsTkwfaS)

A revolutionary fluid dynamics simulation framework based on **Semantic Tensor Theory** that solves the blow-up problem of traditional Navier-Stokes equations while maintaining structural stability.

## 🌊 Overview

Lambda³ (Lambda Cube) introduces a novel approach to fluid simulation using **Semantic Tensors (Λ)** that integrate particle position, velocity, and temperature into a unified structural framework. This method enables:

- **Blow-up Prevention**: Eliminates numerical instabilities common in NSE
- **Structural Event Detection**: Automatic classification of ΔΛC events (Split, Merge, Annihilate, Create)
- **Topological Tracking**: Real-time monitoring of topological invariants (Q_Λ)
- **Chaos Diagnosis**: Network-based quantification of structural disorder
- **NSE Comparison**: Performance benchmarking against traditional methods

## 🚀 Key Features

### 🎮 Try it Now!
**[📱 Run Live Demo in Google Colab](https://colab.research.google.com/drive/12tUo7u9wBRPjeGwgws4v_8_McsTkwfaS)** - No installation required!

### Core Innovations
- **Semantic Tensor Integration**: Unified representation of physical fields
- **Progression Vector (ΛF)**: Dynamic structural evolution tracking
- **Synchronization Rate (σ_s)**: Temperature-velocity field coherence measurement
- **Event Network Analysis**: Causal relationship mapping between structural changes

### Simulation Modes
| Mode | Description | Use Case |
|------|-------------|----------|
| `thermal_flow` | Temperature-velocity coupling emphasis | Thermal-fluid synchronization studies |
| `stable` | Low noise, obstacle-free | Basic structural evolution research |
| `turbulent` | High noise, turbulence promotion | Chaotic structure analysis |
| `super_stable` | Ultra-low noise, high precision | Long-term stability verification |
| `simulation2` | Multiple obstacle configuration | Complex boundary flow analysis |

## 📦 Installation

### Local Installation

#### Requirements
```bash
pip install numpy>=1.21.0 pandas>=1.3.0 matplotlib>=3.5.0 scipy>=1.7.0 networkx>=2.6.0
```

### System Requirements
- **Python**: 3.8+
- **CPU**: 4+ cores recommended
- **Memory**: 8GB+ RAM
- **Storage**: 1GB+ free space

## 🔧 Quick Start

### Basic Usage
```python
from lambda3_simulation import Lambda3SimConfig, initialize_particles, run_simulation

# Configure simulation
Lambda3SimConfig.MODE = "thermal_flow"
Lambda3SimConfig.N = 200  # Number of particles
Lambda3SimConfig.Nsteps = 150  # Simulation steps
Lambda3SimConfig.apply_mode()

# Initialize particle system
particles = initialize_particles()

# Run simulation
results = run_simulation(particles)

# Visualize results
visualize_results(results)
```

### Advanced Configuration
```python
# Custom parameter setup
config = Lambda3SimConfig()
config.VISCOSITY = 0.001           # Kinematic viscosity
config.ALPHA_LAMBDAF = 0.04        # ΛF update rate
config.THRESHOLD_LAMBDA3 = 5.0     # Λ³ distance threshold
config.OBSTACLE_ON = True          # Enable obstacles
config.EXTERNAL_INJECTION_OMEGA_ON = True  # Rotational forcing

# Apply configuration
config.apply_mode()
```

## 🏗️ Architecture

### Core Classes

#### SemanticTensor
The fundamental data structure representing particle state:
```python
class SemanticTensor:
    position: np.array      # 2D position [x, y]
    velocity: np.array      # Velocity vector [vx, vy]
    temperature: float      # Scalar temperature field
    Lambda_core: np.array   # Core structural tensor
    Lambda_F: np.array      # Progression vector
    sigma_s: float          # Synchronization rate
    QLambda: float          # Topological invariant
```

#### Lambda3SimConfig
Centralized simulation configuration:
```python
class Lambda3SimConfig:
    MODE: str = "thermal_flow"
    N: int = 200              # Particle count
    L: float = 180.0          # System size
    dt: float = 0.12          # Time step
    VISCOSITY: float = 0.001  # Kinematic viscosity
    # ... extensive parameter set
```

### Key Algorithms

#### 1. Neighbor Search with Periodic Boundaries
```python
def periodic_kdtree_neighbors(tensor, all_tensors, k=5):
    """Efficient k-nearest neighbor search with periodic BC"""
```

#### 2. ΛF Progression Vector Update
```python
def update_LambdaF_with_neighbors(tensor, neighbors):
    """Gradient-driven ΛF update with viscosity and noise"""
```

#### 3. ΔΛC Event Classification
```python
def classify_transaction(tensor_a, tensor_b, neighbors):
    """Automatic structural change event detection
    Returns: "Merge", "Split", "Annihilate", "Create", "Stable"
    """
```

## 📊 Visualization & Analysis

### Real-time Visualization
- **Structural Entropy Maps**: Spatial distribution of local disorder
- **ΛF Streamplots**: Progression vector field visualization
- **Topological Jump Events**: Q_Λ discontinuity detection
- **ΔΛC Event Density**: Spatial concentration of structural changes

### Network Analysis
- **Particle Networks**: Spatial proximity-based connections
- **Λ³ Event Networks**: Causal relationships between ΔΛC events
- **Centrality Analysis**: Identification of influential particles/events

### Statistical Analysis
- **Q_Λ Conservation Tracking**: Temporal evolution of topological charge
- **Chaos Diagnosis**: Multi-indicator disorder quantification
- **Causality Analysis**: Directional correlation detection

## 🔬 Theoretical Background

### Lambda³ Theory Core Concepts

#### 1. Semantic Tensor Integration
Unlike traditional fluid equations where position, velocity, and pressure are treated independently, Lambda³ represents them as a unified structural tensor:

```
Λ = block_diag(Λ_T, Λ_u)
```
where Λ_T is the temperature tensor and Λ_u is the velocity tensor.

#### 2. Progression Vector ΛF
Represents particle "intention" and evolves through local interactions:

```
dΛF/dt = α∇(ρ_T + σ_s) + ν∇²ΛF + η
```

#### 3. Synchronization Rate σ_s
Measures coherence between temperature gradient and velocity:

```
σ_s = (∇T · u) / (|∇T| |u|)
```

#### 4. ΔΛC Structural Change Events
- **Merge**: Fusion of similar structures (overlap < ε_merge)
- **Split**: Structure fragmentation (∇σ_s > threshold)
- **Annihilate**: Efficiency-driven disappearance (eff < ε_annihilate)
- **Create**: Divergence-density driven generation

## ⚖️ NSE Comparison

### Performance Benchmarking
```python
# Traditional NSE simulation
nse_results = run_nse_simulation(nse_config)

# Lambda³ simulation
l3_results = run_lambda3_simulation(l3_config)

# Comparative analysis
compare_kinetic_energy(nse_results, l3_results)
compare_pressure_fields(nse_results, l3_results)
```

### Key Differences
| Aspect | NSE | Lambda³ |
|--------|-----|---------|
| Blow-up Problem | Present | Eliminated |
| Computational Complexity | O(N³) | O(N log N) |
| Physical Representation | Continuum | Particle + Structure |
| Chaos Detection | Difficult | Automatic |
| Stability | Conditional | Inherent |

## 🛠️ Troubleshooting

### Common Issues

#### Memory Insufficient
**Symptoms**: Crashes during large simulations  
**Solutions**:
- Reduce particle count `N`
- Decrease grid size `GRID_NX`, `GRID_NY`
- Increase `SAVE_INTERVAL`

#### Numerical Instability
**Symptoms**: NaN values, abnormal velocities  
**Solutions**:
- Decrease time step `dt`
- Increase viscosity `VISCOSITY`
- Reduce noise strength `NOISE_STRENGTH`

#### Visualization Errors
**Symptoms**: matplotlib-related errors  
**Solutions**:
```python
import matplotlib
matplotlib.use('Agg')  # For headless environments
```

## 🤝 Contributing

We welcome contributions! Please see our [Contributing Guidelines](CONTRIBUTING.md) for details.

### Development Setup
```bash
git clone https://github.com/yourusername/lambda3-simulation.git
cd lambda3-simulation
pip install -r requirements.txt
python -m pytest tests/
```

### Contribution Areas
- [ ] GPU acceleration (CUDA/OpenCL)
- [ ] 3D extension
- [ ] Real-time web viewer
- [ ] ML-based parameter optimization
- [ ] Distributed computing support

## 📚 Examples

### 🚀 Quick Demo
**[Try Lambda³ in Google Colab](https://colab.research.google.com/drive/12tUo7u9wBRPjeGwgws4v_8_McsTkwfaS)** - Interactive notebook with full simulation

### Basic Thermal Flow Simulation
```python
# Set thermal flow mode
Lambda3SimConfig.MODE = "thermal_flow"
Lambda3SimConfig.apply_mode()

# Run simulation
particles = initialize_particles()
for step in range(Lambda3SimConfig.Nsteps):
    for tensor in particles:
        update_tensor_state(tensor, step, particles)
    detect_and_apply_DeltaLambdaC(particles, step)

# Analyze results
analyze_thermal_synchronization(particles)
```

### Network Analysis Example
```python
# Build event network
lambda3_events = collect_lambda3_events(simulation_results)
G_lambda3 = build_lambda3_event_network(lambda3_events)

# Analyze network properties
components = list(nx.connected_components(G_lambda3))
centrality = nx.betweenness_centrality(G_lambda3)

# Visualize network
plot_event_network_with_energy(G_lambda3, positions, energies)
```

## 📖 Documentation

- [API Reference](docs/api.md)
- [Theory Guide](docs/theory.md)
- [Examples](examples/)
- [FAQ](docs/faq.md)

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

```
MIT License

Copyright (c) 2025 IIZUMI MASAMICHI / Miosync

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

## 🙋‍♂️ Author

**IIZUMI MASAMICHI** / **Miosync**

- Email: [contact@miosync.com]
- GitHub: [@miosync]
- Research: Semantic Tensor Theory & Fluid Dynamics

## 🏆 Citation

If you use Lambda³ in your research, please cite:

```bibtex
@software{lambda3_simulation,
  title={Lambda³ Fluid Simulation: A Semantic Tensor Approach},
  author={IIZUMI, MASAMICHI},
  organization={Miosync},
  year={2025},
  url={https://github.com/miosync/lambda3-simulation}
}
```

## 🌟 Acknowledgments

- Special thanks to the fluid dynamics and computational physics communities
- Inspiration from chaos theory and complex systems research
- NetworkX and SciPy communities for excellent tools

---

**⭐ Star this repository if Lambda³ helps your research!**
