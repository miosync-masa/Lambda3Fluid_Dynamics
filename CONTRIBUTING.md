# Contributing to Lambda³ Fluid Simulation

Thank you for your interest in contributing to Lambda³! This project aims to advance fluid dynamics simulation through innovative Semantic Tensor Theory. We welcome contributions from researchers, developers, and enthusiasts from all backgrounds.

## 🤝 How to Contribute

### Types of Contributions We Welcome

- 🐛 **Bug Reports** - Help us identify and fix issues
- 💡 **Feature Requests** - Suggest new capabilities or improvements
- 📝 **Documentation** - Improve explanations, examples, and guides
- 🔬 **Research Extensions** - Theoretical improvements and new algorithms
- 🎨 **Visualizations** - Enhanced plotting and analysis tools
- ⚡ **Performance Optimizations** - GPU acceleration, distributed computing
- 🧪 **Test Cases** - Unit tests and validation scenarios
- 📊 **Benchmarks** - Comparative studies with other simulation methods

## 🚀 Getting Started

### 1. Development Environment Setup

```bash
# Fork and clone the repository
git clone https://github.com/yourusername/lambda3-simulation.git
cd lambda3-simulation

# Create a virtual environment
python -m venv lambda3-env
source lambda3-env/bin/activate  # On Windows: lambda3-env\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Install development dependencies
pip install pytest pytest-cov black flake8 mypy

# Verify installation
python -m pytest tests/
```

### 2. Try the Demo
Before diving into development, familiarize yourself with the project:
- **[Run the Colab Demo](https://colab.research.google.com/drive/12tUo7u9wBRPjeGwgws4v_8_McsTkwfaS)**
- Explore different simulation modes
- Understand the Lambda³ theoretical framework

### 3. Development Workflow

```bash
# Create a feature branch
git checkout -b feature/your-feature-name

# Make your changes
# ... code, test, document ...

# Run tests
python -m pytest tests/ -v

# Format code
black lambda3_simulation/
flake8 lambda3_simulation/

# Commit your changes
git add .
git commit -m "feat: descriptive commit message"

# Push and create PR
git push origin feature/your-feature-name
```

## 📋 Contribution Guidelines

### Code Standards

#### Python Style Guide
- **PEP 8 Compliance**: Use `black` for automatic formatting
- **Type Hints**: Add type annotations for new functions
- **Docstrings**: Follow NumPy docstring convention

```python
def update_tensor_state(tensor: SemanticTensor, 
                       step: int, 
                       all_tensors: List[SemanticTensor],
                       config: Lambda3SimConfig) -> None:
    """
    Update the state of a single SemanticTensor for one simulation step.
    
    Parameters
    ----------
    tensor : SemanticTensor
        The tensor to update
    step : int
        Current simulation step
    all_tensors : List[SemanticTensor]
        All tensors in the simulation
    config : Lambda3SimConfig
        Simulation configuration
        
    Returns
    -------
    None
        Modifies tensor in-place
        
    Examples
    --------
    >>> tensor = SemanticTensor([0, 0], [1, 1], 4.0)
    >>> update_tensor_state(tensor, 0, [tensor], config)
    """
```

#### Performance Considerations
- **Vectorization**: Use NumPy operations when possible
- **Memory Efficiency**: Avoid unnecessary array copies
- **Profiling**: Use `cProfile` for performance-critical code

```python
# Good: Vectorized operation
distances = np.linalg.norm(positions[:, None, :] - positions[None, :, :], axis=2)

# Avoid: Nested loops
# for i in range(len(positions)):
#     for j in range(len(positions)):
#         distances[i, j] = np.linalg.norm(positions[i] - positions[j])
```

### Testing Requirements

#### Unit Tests
All new functionality must include comprehensive tests:

```python
import pytest
import numpy as np
from lambda3_simulation import SemanticTensor, Lambda3SimConfig

def test_semantic_tensor_initialization():
    """Test SemanticTensor initializes correctly."""
    position = np.array([1.0, 2.0])
    velocity = np.array([0.5, -0.3])
    temperature = 4.0
    
    tensor = SemanticTensor(position, velocity, temperature)
    
    assert np.allclose(tensor.position, position)
    assert np.allclose(tensor.velocity, velocity)
    assert tensor.temperature == temperature
    assert tensor.Lambda is not None
    assert tensor.eff is not None

def test_neighbor_search():
    """Test periodic neighbor search functionality."""
    # Setup test scenario
    config = Lambda3SimConfig()
    config.L = 10.0
    config.NEIGHBOR_RADIUS = 2.0
    
    # Create test tensors
    tensors = [
        SemanticTensor([1, 1], [0, 0], 1.0),
        SemanticTensor([2, 1], [0, 0], 1.0),  # Should be neighbor
        SemanticTensor([8, 1], [0, 0], 1.0),  # Should be neighbor (periodic)
        SemanticTensor([5, 5], [0, 0], 1.0),  # Should not be neighbor
    ]
    
    neighbors = get_neighbors(tensors[0], tensors)
    
    assert len(neighbors) == 2  # Should find 2 neighbors
    assert tensors[1] in neighbors
    assert tensors[2] in neighbors
    assert tensors[3] not in neighbors
```

#### Integration Tests
Test complete simulation workflows:

```python
def test_full_simulation_thermal_flow():
    """Test complete thermal_flow simulation runs without errors."""
    Lambda3SimConfig.MODE = "thermal_flow"
    Lambda3SimConfig.N = 50  # Smaller for faster testing
    Lambda3SimConfig.Nsteps = 10
    Lambda3SimConfig.apply_mode()
    
    particles = initialize_particles()
    
    # Should run without exceptions
    for step in range(Lambda3SimConfig.Nsteps):
        for tensor in particles:
            update_tensor_state(tensor, step, particles)
        detect_and_apply_DeltaLambdaC(particles, step)
    
    # Verify particles maintain reasonable states
    for tensor in particles:
        assert not np.isnan(tensor.position).any()
        assert not np.isnan(tensor.velocity).any()
        assert not np.isnan(tensor.temperature)
```

### Documentation Standards

#### Code Documentation
- **Inline Comments**: Explain complex algorithms and Lambda³ theory
- **API Documentation**: Complete parameter and return value descriptions
- **Examples**: Include usage examples in docstrings

#### Research Documentation
For theoretical contributions, include:
- **Mathematical Derivations**: Clear step-by-step explanations
- **Physical Interpretations**: Connect math to physical meaning
- **Validation Studies**: Compare results with known solutions

#### Visualization Documentation
- **Plot Explanations**: Describe what each visualization shows
- **Parameter Sensitivity**: Document how parameters affect outputs
- **Interpretation Guides**: Help users understand results

## 🐛 Bug Reports

### Before Reporting
1. **Search existing issues** to avoid duplicates
2. **Test with latest version** from main branch
3. **Try different simulation modes** to isolate the problem
4. **Check the [FAQ](docs/faq.md)** for common issues

### Bug Report Template
```markdown
**Bug Description**
Clear description of the issue

**Reproduction Steps**
1. Set configuration: `Lambda3SimConfig.MODE = "thermal_flow"`
2. Run simulation with: `N=200, Nsteps=150`
3. Error occurs at step: 45

**Expected Behavior**
What should happen

**Actual Behavior**
What actually happens

**Environment**
- OS: [e.g., Ubuntu 20.04, Windows 10, macOS 12]
- Python version: [e.g., 3.9.7]
- NumPy version: [e.g., 1.21.0]
- Lambda³ version/commit: [e.g., main branch, commit abc123]

**Additional Context**
- Simulation parameters used
- Any custom modifications
- Error logs/stack traces
```

## 💡 Feature Requests

### Feature Request Template
```markdown
**Feature Description**
Clear description of the proposed feature

**Use Case**
Why is this feature needed? What problem does it solve?

**Proposed Implementation**
- High-level approach
- Key components that would need modification
- Potential challenges

**Alternative Solutions**
Other ways to achieve the same goal

**Research Background**
- Relevant papers or theoretical foundations
- Connection to Lambda³ theory
- Validation approach
```

### Priority Areas for New Features

#### High Priority
- **GPU Acceleration**: CUDA/OpenCL implementations
- **3D Extensions**: Volumetric Lambda³ simulations
- **Machine Learning Integration**: Parameter optimization, pattern recognition
- **Distributed Computing**: MPI support for large-scale simulations

#### Medium Priority
- **Advanced Visualizations**: Interactive 3D plots, real-time streaming
- **Export Formats**: VTK, HDF5, custom binary formats
- **Boundary Conditions**: More sophisticated wall interactions
- **Multi-phase Flows**: Extension to multiple fluid phases

#### Research Extensions
- **Quantum Lambda³**: Quantum mechanical extensions
- **Relativistic Extensions**: Special/general relativity incorporation
- **Stochastic Methods**: Advanced noise models and uncertainty quantification

## 🔬 Research Contributions

### Theoretical Contributions
We especially welcome contributions that:

- **Extend Lambda³ Theory**: New mathematical frameworks
- **Improve Convergence**: Better stability and accuracy
- **Novel Applications**: New domains for Lambda³ simulation
- **Validation Studies**: Comparisons with experimental data

### Publication Guidelines
- **Open Science**: Contributions should be open and reproducible
- **Collaboration**: We're happy to collaborate on publications
- **Citation**: Please cite the Lambda³ framework appropriately

## 📞 Communication

### Channels
- **GitHub Issues**: Bug reports, feature requests
- **GitHub Discussions**: General questions, research discussions
- **Email**: [m.iizumi@miosync.com] for private inquiries

### Response Times
- **Bug Reports**: We aim to respond within 48 hours
- **Feature Requests**: Initial response within 1 week
- **Pull Requests**: Code review within 3-5 days

### Community Guidelines
- **Respectful Communication**: Treat all contributors with respect
- **Constructive Feedback**: Focus on improving the project
- **Inclusive Environment**: Welcome contributors from all backgrounds
- **Scientific Rigor**: Maintain high standards for research quality

## 🏆 Recognition

### Contributor Recognition
- **Contributors List**: All contributors are acknowledged in README.md
- **Release Notes**: Significant contributions highlighted in releases
- **Research Collaboration**: Opportunity for co-authorship on publications
- **Conference Presentations**: Support for presenting Lambda³ research

### Types of Contributions Recognized
- Code contributions (features, fixes, optimizations)
- Documentation improvements
- Research extensions and theoretical work
- Community building and support
- Testing and quality assurance
- Educational content and tutorials

## 📄 License

By contributing to Lambda³, you agree that your contributions will be licensed under the same MIT License that covers the project. See [LICENSE](LICENSE) for details.

## 🙏 Acknowledgments

Thank you to all contributors who help advance Lambda³ and push the boundaries of computational fluid dynamics!

---

**Questions?** Feel free to reach out through any of our communication channels. We're here to help you contribute successfully to Lambda³!

**Happy Contributing! 🌊✨**
