# --- Basic libraries ---
import numpy as np        # Numerical operations and array handling
import pandas as pd       # Data analysis and table structures

# --- Plotting / Visualization ---
import matplotlib.pyplot as plt        # Core plotting library
from matplotlib import animation       # Animation support
import matplotlib.animation as animation
from IPython.display import HTML       # For inline animations in Jupyter/Colab
from matplotlib.colors import ListedColormap

# --- Scientific analysis ---
from scipy.ndimage import gaussian_filter    # Gaussian smoothing for spatial fields
from scipy.spatial import cKDTree            # Efficient nearest neighbor search
from scipy.stats import ks_2samp, zscore     # Statistical tests (KS-test, Z-score)
from scipy.fft import fft2, ifft2
from scipy.linalg import block_diag

# --- Network analysis ---
import networkx as nx         # Network/graph structures and analysis

# ==============================
# 🌍 Simulation Initialization Parameters
# ==============================
class Lambda3SimConfig:
    # --- Simulation mode setting ---
    MODE = "thermal_flow"   # Simulation mode: affects parameter set and boundary conditions ("default", "stable", "turbulent", "simulation2", "super_stable")

    # --- 🧪 Particle system / structural scale ---
    N = 200               # Number of fluid particles
    L = 180.0             # System size (domain length)
    T = 4.0               # Initial temperature (sets initial speed/entropy)
    dt = 0.12             # Time increment per update ("timestep" for structure evolution)
    Nsteps = 150          # Total number of simulation steps
    DIMS = 2              # Number of spatial dimensions (2D system)

    # --- 🌊 Viscosity (fluid property) ---
    VISCOSITY = 0.001  # Kinematic viscosity (ν), controls diffusion and momentum dissipation in the simulated fluid.

    # --- 🌪️ Chaos detection window ---
    CHAOS_WINDOW = 30     # Window size for moving statistics in local chaos detection

    # --- 🤠 Lambda3 network analysis parameters ---
    W_SPACE = 1.0                 # Spatial clustering weight for event/particle networks
    W_STEP = 0.5                  # Temporal distance weight in network analysis
    W_TOPO = 1.5                  # Topological weight (e.g., for $Q_\Lambda$ events)
    R_PARTICLE_CONNECT = 10       # Distance threshold for connecting particles (spatial network)
    THRESHOLD_LAMBDA3 = 5.0       # Threshold for Lambda3 distance-based event classification
    R_SPACE_FILTER = 12.0         # Radius for spatial smoothing in analysis

    # --- 🔄 LambdaF dynamic updating ---
    ALPHA_LAMBDAF = 0.04          # Learning rate for updating the LambdaF progression vector
    NOISE_STRENGTH = 0.02         # Amplitude of noise added to progression dynamics
    K_NEIGHBORS = 5               # Number of neighbors for local structure calculation
    NEIGHBOR_RADIUS = 7.0         # Radius for defining local neighbors

    # --- 💥 ΔΛC (Pulsation event) thresholds ---
    EPSILON_MERGE = 1.0           # Threshold for merge events (structure similarity)
    THRESHOLD_SPLIT = 1.2         # Threshold for split events (structure divergence)
    EPSILON_ANNIHILATE = 0.1      # Threshold for annihilation (vanishing event)
    PHI_DIV = 1.2                 # Angular threshold for event classification
    TAU_DENSITY = 0.5             # Density-based event threshold

    # --- Grid and output settings ---
    GRID_NX = 60                  # Number of grid cells (X direction)
    GRID_NY = 60                  # Number of grid cells (Y direction)
    SAVE_INTERVAL = 5             # Number of steps between data saves
    Q_JUMP_THRESHOLD = np.pi / 2  # Threshold for detecting topological charge jumps
    R_CLUSTER_LOCAL_STD = 5.0     # Radius for local standard deviation (anomaly detection)
    SNAPSHOT_STEPS = [0, 75, 149] # Timesteps to save system snapshots
    HEATMAP_WINDOW = 100          # Temporal window for heatmap aggregation

    # --- Obstacle configuration ---
    OBSTACLE_ON = True                        # Whether to enable obstacles in domain
    RECT_OBSTACLES = [((60, 80), (60, 80))]   # List of rectangular obstacles [(x_range, y_range)]
    CIRC_OBSTACLES = [((100, 40), 12)]        # List of circular obstacles [(center, radius)]

    # --- Structural switches (physical/numerical effects) ---
    WALL_ON = False                # Enable solid walls
    WALL_FRICTION_ON = False       # Wall friction
    EXTERNAL_FORCE_ON = False      # External force field
    WALL_REFLECT_ON = False        # Wall Reflet
    MULTISCALE_ON = False          # Enable multi-scale coupling
    INLET_OUTLET_ON = False        # Inlet/outlet boundary conditions
    TURBULENCE_ON = False          # Enable turbulence mode

    # --- base injection ---
    GRAVITY = np.array([0.0, -0.001])           # Downward gravity vector (e.g., Earth gravity)
    ELECTRIC_FIELD = np.array([0.5, 0.0])     # Electric field (example: x-direction)
    MAGNETIC_FIELD = np.array([0.0, 0.0, 1.0]) # Magnetic field (2D: use only z-component)

    # --- External noise injection ---
    EXTERNAL_INJECTION_OMEGA_ON = True
    OMEGA_INJECTION_START = 30         # Start step
    OMEGA_INJECTION_END = 150           # End step
    INJECTION_OMEGA_LEVEL = 0.2       # Amplitude of injected rotational noise
    INJECTION_OMEGA_STRENGTH = 1.0     # Strength (scaling) of injected rotational noise

    EXTERNAL_INJECTION_NOISE_ON = True
    NOISE_INJECTION_START = 0          # Noise Start step
    NOISE_INJECTION_END = 50          # Noise End step
    INJECTION_NOISE_LEVEL = 0.02       # Level of injected external noise

        @classmethod
    def apply_mode(cls):
        if cls.MODE == "super_stable":
            cls.ALPHA_LAMBDAF = 0.02
            cls.NOISE_STRENGTH = 0.005
            cls.K_NEIGHBORS = 7
            cls.OBSTACLE_ON = False
            cls.TURBULENCE_ON = False

        elif cls.MODE == "turbulent":
            cls.ALPHA_LAMBDAF = 0.05
            cls.NOISE_STRENGTH = 0.03
            cls.K_NEIGHBORS = 3
            cls.OBSTACLE_ON = False
            cls.TURBULENCE_ON = True

        elif cls.MODE == "simulation2":
            cls.RECT_OBSTACLES = [((30, 40), (30, 120)), ((100, 120), (60, 100))]
            cls.CIRC_OBSTACLES = [((70, 70), 8), ((120, 40), 20)]
            cls.OBSTACLE_ON = True

        elif cls.MODE == "stable":
            cls.OBSTACLE_ON = False
            cls.TURBULENCE_ON = False
            cls.EXTERNAL_INJECTION_ON = False
            cls.EXTERNAL_INJECTION_OMEGA_ON = True
            cls.WALL_ON = False
            cls.WALL_FRICTION_ON = False
            cls.RECT_OBSTACLES = []
            cls.CIRC_OBSTACLES = []

        # 🔥 Additional: Special settings for observing thermal and fluid synchronization and jump events
        elif cls.MODE == "thermal_flow":
            cls.ALPHA_LAMBDAF = 0.03          # Moderate LambdaF progression rate (emphasis on ease of observation)
            cls.NOISE_STRENGTH = 0.015        # Slightly lower noise strength for clearer jump observation
            cls.K_NEIGHBORS = 5               # Standard number of neighbors
            cls.NEIGHBOR_RADIUS = 8.0         # Sufficient spatial extent for synchronization rate

            cls.EXTERNAL_INJECTION_OMEGA_ON = True
            cls.OMEGA_INJECTION_START = 20
            cls.OMEGA_INJECTION_END = 150
            cls.INJECTION_OMEGA_LEVEL = 0.25  # Promotes clear interaction between temperature and velocity fields
            cls.INJECTION_OMEGA_STRENGTH = 1.5

            cls.EXTERNAL_INJECTION_NOISE_ON = True
            cls.NOISE_INJECTION_START = 0
            cls.NOISE_INJECTION_END = 50
            cls.INJECTION_NOISE_LEVEL = 0.015 # Suppresses noise injection into the temperature field to clarify synchronization

            cls.OBSTACLE_ON = True
            cls.RECT_OBSTACLES = [((60, 80), (60, 80))]
            cls.CIRC_OBSTACLES = [((100, 40), 12)]

            cls.TURBULENCE_ON = False          # Turn off turbulence to focus on pure interaction between temperature and velocity fields
            cls.WALL_ON = True                 # Enable wall boundary conditions to observe phenomena at the boundary
            cls.WALL_FRICTION_ON = True        # Enable wall friction to observe boundary layer phenomena

            # Set weak external fields like gravity and electric fields to emphasize natural jump phenomena
            cls.GRAVITY = np.array([0.0, -0.0005])
            cls.ELECTRIC_FIELD = np.array([0.1, 0.0])

# ---mode apply ---
Lambda3SimConfig.apply_mode()

# --- Adaptive time step calculation based on Lambda_F norm ---
def adaptive_dt(tensor, threshold=0.12):
    """
    Dynamically adjust the time step according to the norm of Lambda_F (progression vector).
    Use a smaller step for large movements, and a larger step for stable regions.
    """
    return 0.01 if np.linalg.norm(tensor.Lambda_F) > threshold else 0.12

# --- Check if a position is inside any defined obstacle ---
def is_in_obstacle(x, config):
    """
    Determine if the given position 'x' is inside any rectangular or circular obstacle
    defined in the config. Returns True if the point is within an obstacle.
    """
    # Check rectangular obstacles
    for (x_range, y_range) in config.RECT_OBSTACLES:
        if x_range[0] <= x[0] <= x_range[1] and y_range[0] <= x[1] <= y_range[1]:
            return True

    # Check circular obstacles
    for (center, radius) in config.CIRC_OBSTACLES:
        if np.linalg.norm(x - np.array(center)) <= radius:
            return True

    return False

# ==============================
# 0. Definition of SemanticTensor structure
# ==============================
class SemanticTensor:
    def __init__(self, position, velocity, temperature, Lambda=None, from_mode=None):
        """
        Initialize SemanticTensor with position, velocity, and temperature.
        Integrates temperature and velocity fields explicitly into the structural tensor Λ.
        """
        self.position = np.array(position)  # 2D position
        self.velocity = np.array(velocity)  # Velocity vector [vx, vy]
        self.temperature = temperature      # Scalar temperature field

        # Initialize Lambda tensor based on provided mode
        if from_mode == "vortex_seed":
            self.Lambda_core = self.generate_vortex_tensor(position)
        elif Lambda is not None:
            self.Lambda_core = Lambda
        else:
            self.Lambda_core = np.random.rand(4)  # Default random 4-component tensor

        # Create structured tensor Λ integrating temperature and velocity fields
        self.Lambda_T = np.array([[temperature]])
        self.Lambda_u = np.outer(self.velocity, self.velocity)
        self.Lambda = block_diag(self.Lambda_T, self.Lambda_u)

        # Progression vector ΛF set as velocity
        self.Lambda_F = self.velocity

        # Synchronization rate σ_s as cosine similarity between ∇T and u
        grad_T_dummy = np.array([temperature, temperature])  # Placeholder gradient
        epsilon = 1e-8
        self.sigma_s = (
            np.dot(grad_T_dummy, self.velocity) /
            (np.linalg.norm(grad_T_dummy) * np.linalg.norm(self.velocity) + epsilon)
        )

        # Tension density ρ_T calculated based on velocity magnitude
        vel_norm = np.linalg.norm(self.velocity)
        if Lambda3SimConfig.MULTISCALE_ON:
            self.rho_T = np.log1p(vel_norm)
        else:
            self.rho_T = vel_norm

        # Efficiency calculated by projecting Λ_core onto Λ_F
        self.eff = self.calc_efficiency()

        # Placeholder for topological invariant
        self.QLambda = None

        # State flags
        self.is_active = not Lambda3SimConfig.OBSTACLE_ON or not is_in_obstacle(position, Lambda3SimConfig)
        self.is_obstacle = False
        self.tags = set()

        # History of Λ matrix properties
        self.lambda_hist = {key: [] for key in ["det", "trace", "norm", "eig1", "eig2", "angle_det"]}

    def update_QLambda(self):
        """Compute and record topological invariant QΛ and other Λ properties."""
        try:
            Lambda_mat = self.Lambda_core.reshape(2, 2)
            det = np.linalg.det(Lambda_mat)
            eigvals = np.linalg.eigvals(Lambda_mat)
            trace = np.trace(Lambda_mat)
            norm = np.linalg.norm(Lambda_mat)
            angle = np.angle(det)
            self.QLambda = angle if not np.isnan(det) else 0.0

            # Update history
            self.lambda_hist["det"].append(det)
            self.lambda_hist["trace"].append(trace)
            self.lambda_hist["norm"].append(norm)
            self.lambda_hist["eig1"].append(eigvals[0])
            self.lambda_hist["eig2"].append(eigvals[1])
            self.lambda_hist["angle_det"].append(angle)

        except Exception as e:
            print(f"QLambda Error: {e}")
            self.QLambda = 0.0
            for key in self.lambda_hist:
                self.lambda_hist[key].append(np.nan)

    def calc_efficiency(self):
        """Calculate efficiency of Λ_core projection onto Λ_F."""
        norm_LF = np.linalg.norm(self.Lambda_F)
        proj = np.dot(self.Lambda_core[:2], self.Lambda_F) / (norm_LF + 1e-8)
        return proj * np.exp(-np.var(self.Lambda_core))

    def generate_vortex_tensor(self, pos):
        """Generate vortex-structured initial tensor Λ_core based on position."""
        cx, cy = 90.0, 90.0
        dx, dy = pos[0] - cx, pos[1] - cy
        strength = 1.0 / (np.hypot(dx, dy) + 1e-3)
        return np.array([0.0, strength, -strength, 0.0])

# ========================
# 1. Neighbor Search Function
# ========================
def get_neighbors(tensor, all_tensors, radius=None):
    """
    Returns a list of neighboring particles within a given radius from the current tensor.
    - Uses Euclidean distance in position space.
    - Excludes the tensor itself from the list.
    """
    if radius is None:
        radius = Lambda3SimConfig.NEIGHBOR_RADIUS
    return [t for t in all_tensors if np.linalg.norm(t.position - tensor.position) < radius and t is not tensor]

# ========================
# 2. Advanced Gradient-driven Update for LambdaF
# ========================
def update_LambdaF_with_neighbors(tensor, neighbors):
    """
    Optimized and robust version for updating Lambda_F using neighboring tensors.
    """
    if not neighbors:
        return

    epsilon = 1e-8  # Stability term to prevent division by zero

    # Precompute distances and avoid redundant calculations
    delta_positions = np.array([n.position - tensor.position for n in neighbors])
    distances = np.linalg.norm(delta_positions, axis=1)[:, np.newaxis] + epsilon

    # Gradient vectors (weighted average toward neighbors)
    grad_rho_vec = np.mean([
        (n.rho_T - tensor.rho_T) * delta / dist
        for n, delta, dist in zip(neighbors, delta_positions, distances)
    ], axis=0)

    grad_sigma_vec = np.mean([
        (n.sigma_s - tensor.sigma_s) * delta / dist
        for n, delta, dist in zip(neighbors, delta_positions, distances)
    ], axis=0)

    # Total gradient contribution
    delta_vec = grad_rho_vec + grad_sigma_vec

    # Viscosity contribution
    mean_LambdaF = np.mean([n.Lambda_F for n in neighbors], axis=0)
    viscosity_term = Lambda3SimConfig.VISCOSITY * (mean_LambdaF - tensor.Lambda_F)

    # Final Lambda_F update
    tensor.Lambda_F += (
        Lambda3SimConfig.ALPHA_LAMBDAF * delta_vec +
        viscosity_term +
        Lambda3SimConfig.NOISE_STRENGTH * np.random.normal(size=2, scale=0.03)
    )

def periodic_kdtree_neighbors(tensor, all_tensors, k=None):
    """
    Finds k nearest neighbors using periodic boundary conditions.
    - Uses cKDTree for efficient nearest neighbor search.
    - Accounts for periodic images by tiling positions across the domain boundaries.
    - Returns k nearest neighbors (excluding the tensor itself).
    """
    if k is None:
        k = Lambda3SimConfig.K_NEIGHBORS

    L = Lambda3SimConfig.L
    orig_pos = tensor.position
    shifts = np.array([
        [0, 0], [L, 0], [-L, 0], [0, L], [0, -L],
        [L, L], [-L, -L], [L, -L], [-L, L]
    ])
    all_positions = np.array([t.position for t in all_tensors])
    all_images = np.concatenate([all_positions + shift for shift in shifts], axis=0)

    tree = cKDTree(all_images)
    dists, idxs = tree.query(orig_pos, k=k+1)

    N = len(all_tensors)
    neighbors_idx = np.mod(idxs, N)
    neighbors_idx = [idx for idx in neighbors_idx if not np.allclose(all_tensors[idx].position, orig_pos)]
    neighbors = [all_tensors[j] for j in neighbors_idx[:k]]

    return neighbors

# ==============================
# 3. ΔΛC (Event Classification and State Transition) Logic
# ==============================
def compute_Q_Lambda(tensors):
    Q_Lambda = sum(np.trace(t.Lambda) for t in tensors)
    return Q_Lambda

def detect_pulsation(tensor, neighbors, epsilon_T=0.05, epsilon_vorticity=0.1):
    # 温度勾配
    grad_T = np.mean([abs(n.temperature - tensor.temperature) for n in neighbors])

    # 速度場の渦度 (簡略化)
    vorticity = np.linalg.norm(np.mean([np.cross(n.velocity, tensor.velocity) for n in neighbors]))

    pulsation_event = grad_T > epsilon_T or vorticity > epsilon_vorticity
    return pulsation_event

def add_magnetic_force(tensor):
    """
    Add Lorentz force (due to magnetic field) to the progression vector Lambda_F.
    - v: 2D velocity vector (converted to 3D for cross product)
    - B: 3D magnetic field vector ([Bx, By, Bz]); in 2D, only the z-component is typically used.
    - Lorentz force: F = v × B (cross product), only the first two components are used in 2D.
    The effect is added to Lambda_F, scaled by the simulation time step.
    """
    v = np.array([tensor.Lambda_F[0], tensor.Lambda_F[1], 0.0])     # 2D velocity as 3D vector
    B = Lambda3SimConfig.MAGNETIC_FIELD                             # Magnetic field (3D)
    lorentz = np.cross(v, B)[:2]                                    # Only x, y components for 2D
    tensor.Lambda_F += lorentz * Lambda3SimConfig.dt                # Update progression vector

def compute_lambda_c_divergence(LambdaF_grid):
    """
    Calculate convergence rate λc as the negative divergence of the progression vector field (LambdaF).
    Positive values: convergence (flow in), Negative: divergence (flow out).
    LambdaF_grid: shape (Nx, Ny, 2)
    Returns: lambda_c (Nx, Ny)
    """
    dFdx = np.gradient(LambdaF_grid[:, :, 0], axis=0)
    dFdy = np.gradient(LambdaF_grid[:, :, 1], axis=1)
    lambda_c = -(dFdx + dFdy)
    return lambda_c

def compute_lambda_c_laplacian(sigma_field):
    """
    Calculate convergence rate λc as the Laplacian of a scalar field (e.g., sigma_field ≈ rho_T).
    High positive values: local clustering/aggregation, Negative: local depletion.
    sigma_field: shape (Nx, Ny)
    Returns: lambda_c (Nx, Ny)
    """
    grad_x, grad_y = np.gradient(sigma_field)
    laplacian = np.gradient(grad_x, axis=0) + np.gradient(grad_y, axis=1)
    return laplacian

def solve_poisson_2d(rhs):
    """
    Solve the 2D Poisson equation: ∇²p = rhs
    rhs: right-hand side (2D numpy array)
    Returns: p (2D numpy array), solved with periodic boundary
    """
    Nx, Ny = rhs.shape
    # Fourier wavenumbers for periodic BC
    kx = np.fft.fftfreq(Nx).reshape(-1,1)
    ky = np.fft.fftfreq(Ny).reshape(1,-1)
    k2 = (kx**2 + ky**2)
    k2[0,0] = 1e-12  # avoid division by zero (DC component)

    rhs_hat = fft2(rhs)
    p_hat = rhs_hat / (-4 * np.pi**2 * k2)
    p = np.real(ifft2(p_hat))
    p -= np.mean(p)  # Remove global mean (pressure only unique up to constant)
    return p

def compute_sigma_s(tensor, neighbors, epsilon=1e-8):
    grad_T = np.mean([n.temperature - tensor.temperature for n in neighbors])
    grad_T_vec = np.array([grad_T, grad_T])  # Temperature Gradient Vector (same for x and y components)
    u = tensor.velocity

    numerator = np.dot(grad_T_vec, u)
    denominator = (np.linalg.norm(grad_T_vec) * np.linalg.norm(u)) + epsilon

    sigma_s = numerator / denominator
    return sigma_s

def compute_sigma_gradient(tensor, neighbors):
    """
    Calculate the mean gradient of the local synchronization rate (sigma_s) relative to neighbors.
    Large sigma gradients often indicate the onset of structure separation, i.e., a "Split" event.
    """
    if not neighbors:
        return 0.0
    grad = np.mean([
        (n.sigma_s - tensor.sigma_s) / (np.linalg.norm(n.position - tensor.position) + 1e-8)
        for n in neighbors
    ])
    return abs(grad)

def classify_transaction(tensor_a, tensor_b=None, neighbors=None):
    """
    Classify the ΔΛC (pulsation/transition) event for a given tensor.
    - If another tensor is supplied (tensor_b), check for 'Merge' (overlap + sigma_s criteria).
    - If neighbors are provided, compute local sigma_s gradient to check for 'Split'.
    - If efficiency drops below threshold, classify as 'Annihilate'.
    - If divergence (sum of Lambda_F) and energy density both exceed thresholds, classify as 'Create'.
    - Otherwise, event is 'Stable'.
    """
    if tensor_b is not None:
        overlap = np.linalg.norm(tensor_a.Lambda - tensor_b.Lambda)
        if (overlap < Lambda3SimConfig.EPSILON_MERGE and
            tensor_a.sigma_s > tensor_b.sigma_s):
            # If two structures are very close and one is more coherent, merge them.
            return "Merge"

    if neighbors is not None:
        grad_sigma = compute_sigma_gradient(tensor_a, neighbors)
    else:
        grad_sigma = 0.0

    if grad_sigma > Lambda3SimConfig.THRESHOLD_SPLIT:
        # Sharp local change in sigma_s indicates structure splitting.
        return "Split"

    if tensor_a.eff < Lambda3SimConfig.EPSILON_ANNIHILATE:
        # If efficiency collapses, treat as annihilation (decay/disappearance).
        return "Annihilate"

    divergence = np.sum(tensor_a.Lambda_F)
    if (divergence > Lambda3SimConfig.PHI_DIV and
        tensor_a.rho_T > Lambda3SimConfig.TAU_DENSITY):
        # Strong divergence and high energy density signals spontaneous structure creation.
        return "Create"

    # Default: no event, remain stable.
    return "Stable"

def handle_event(event_type, tensor_a, tensor_b):
    """
    Apply state transition rules based on the classified ΔΛC event.
    - Merge: Average Lambda and sigma_s between both tensors.
    - Split: Add/subtract symmetric noise to Lambda of both tensors.
    - Annihilate: Set tensor to inactive (effectively 'delete').
    - Create: Re-initialize Lambda and sigma_s randomly.
    """
    if event_type == "Merge":
        Lambda_new = (tensor_a.Lambda + tensor_b.Lambda) / 2
        tensor_a.Lambda = tensor_b.Lambda = Lambda_new
        avg_sigma = np.mean([tensor_a.sigma_s, tensor_b.sigma_s])
        tensor_a.sigma_s = tensor_b.sigma_s = avg_sigma
    elif event_type == "Split":
        noise = np.random.normal(0, 0.3, size=tensor_a.Lambda.shape)
        tensor_a.Lambda += noise
        tensor_b.Lambda -= noise
    elif event_type == "Annihilate":
        tensor_a.is_active = False
    elif event_type == "Create":
        tensor_a.Lambda = np.random.rand(*tensor_a.Lambda.shape)
        tensor_a.sigma_s = np.random.rand()

def analyze_causality(x, y, max_lag=20):
    """
    Perform lagged correlation analysis between x (cause) and y (effect) series.
    Useful for extracting directional causality in time-series or structure progression.
    Returns: array of correlations for each lag value.
    """
    corrs = [np.corrcoef(x[:-lag], y[lag:])[0,1] if lag>0 else np.corrcoef(x, y)[0,1]
             for lag in range(max_lag)]
    return np.array(corrs)

def is_chaos_by_network(G_lambda3, min_n_components=2, min_size=100, frac_thr=0.95):
    """
    Diagnoses whether the event network G_lambda3 exhibits 'chaos' or 'fragmentation':
    - At least min_n_components (large, independent clusters) exist.
    - Or, no cluster dominates (>95% of all nodes).
    Returns True if the network structure is sufficiently fragmented/disordered.
    """
    components = list(nx.connected_components(G_lambda3))
    sizes = [len(c) for c in components]
    large_components = [s for s in sizes if s >= min_size]
    if len(large_components) >= min_n_components:
        return True
    if sizes and max(sizes) / sum(sizes) < frac_thr:
        return True
    return False

def max_q_lambda_disorder(components_lambda3, QLambdas_E, std_thr=2.0):
    """
    Returns True if any Lambda^3 network component has a standard deviation of Q_Lambda (topological charge)
    greater than the given threshold (std_thr). Used to detect topological 'chaos' or structural disorder.
    """
    stds = [np.std([QLambdas_E[i] for i in comp]) for comp in components_lambda3 if len(comp) > 1]
    return np.max(stds) > std_thr if stds else False

def sample_maxwellian_velocity():
    """
    Sample initial velocities for all particles from a Maxwellian (Gaussian) distribution,
    matching the system temperature and dimensionality. The net momentum is zeroed (center of mass frame).
    """
    N = Lambda3SimConfig.N
    T = Lambda3SimConfig.T
    dims = Lambda3SimConfig.DIMS
    velocities = np.random.normal(0, np.sqrt(T), size=(N, dims))
    velocities -= velocities.mean(axis=0)
    return velocities

def rotational_field(position, center, omega, strength):
    """
    Returns the local rotational vector field (e.g., for externally injected vorticity/rotation)
    at a given position relative to a center point, with angular velocity omega and amplitude strength.
    Used for inducing vortices or angular momentum.
    """
    position = np.array(position, dtype=float)
    center = np.array(center, dtype=float)
    rel = position - center
    norm = np.linalg.norm(rel) + 1e-8
    if norm == 0:  # Avoid division by zero at the origin
        return np.zeros_like(rel)
    perp = np.array([-rel[1], rel[0]])  # Perpendicular vector (2D rotation)
    return strength * omega * perp / norm

def log_chaos_diagnosis(step, is_chaos_net, is_chaos_qstd, dlc_trend, angle_std, norm_jump, eig1_sign_changes):
    """
    Log and print detailed reasons for diagnosing chaos at a given step.
    Combines network structure, topological disorder, event trends, and tensor fluctuations.
    """
    if chaos_flag == 1:
        reasons = []
        if is_chaos_net:
            components = list(nx.connected_components(G_lambda3))
            sizes = [len(c) for c in components]
            large_components = [s for s in sizes if s >= 50]
            reasons.append(f"is_chaos_by_network: large_components={len(large_components)}, max_size_ratio={max(sizes)/sum(sizes):.3f}")
        if is_chaos_qstd:
            stds = [np.std([QLambdas_E[i] for i in comp]) for comp in components_lambda3 if len(comp) > 1]
            reasons.append(f"max_q_lambda_disorder: max_std={np.max(stds):.3f}")
        if dlc_trend:
            reasons.append(f"dlc_trend: mean_DeltaLambdaC={np.mean(DeltaLambdaC_count_history[-5:]):.3f}")
        reasons.append(f"angle_std={angle_std:.3f}, norm_jump={norm_jump:.3e}, eig1_sign_changes={eig1_sign_changes:.1f}")
        print(f"Step {step}: Chaos detected! Reasons: {', '.join(reasons)}")

def build_lambda3_event_network(lambda3_events, positions_only, r_filter=Lambda3SimConfig.R_SPACE_FILTER):
    """
    Build an event network graph based on Lambda^3 (pulsation/structural change) event positions and steps.
    Each node is an event; edges connect events that are 'close' in space, time, and topology.
    Returns a NetworkX graph for event clustering/chaos analysis.
    """
    if not lambda3_events or len(positions_only) == 0:
        return nx.Graph()  # Return empty graph if no events
    tree = cKDTree(positions_only)
    pairs = tree.query_pairs(r=r_filter)
    G_lambda3 = nx.Graph()
    G_lambda3.add_nodes_from(range(len(lambda3_events)))
    for i, j in pairs:
        pos1, step1, q1 = lambda3_events[i]
        pos2, step2, q2 = lambda3_events[j]
        d = lambda3_progress_distance(pos1, step1, q1, pos2, step2, q2)
        if d < Lambda3SimConfig.THRESHOLD_LAMBDA3:
            G_lambda3.add_edge(i, j)
    return G_lambda3

def lambda3_progress_distance(pos1, step1, q1, pos2, step2, q2):
    """
    Generalized distance metric for event network construction:
    Combines spatial, temporal, and topological (Q_Lambda) differences with configurable weights.
    """
    d_space = np.linalg.norm(pos1 - pos2)
    d_step  = abs(step1 - step2)
    d_topo  = min(abs(q1 - q2), 2 * np.pi - abs(q1 - q2))  # Topological phase wrapping
    return (
        Lambda3SimConfig.W_SPACE * d_space +
        Lambda3SimConfig.W_STEP  * d_step +
        Lambda3SimConfig.W_TOPO  * d_topo
    )

def initialize_particles():
    """
    Initialize the fluid particle system based on simulation mode.

    - In "stable" (or "super_stable") mode:
      Particles are placed on a uniform square grid for reproducibility and controlled experiments.
      Each particle is given a very small initial velocity (to avoid perfect symmetry).

    - In all other modes:
      Particle positions are randomized in a square box of size L, and initial velocities
      are sampled from a Maxwellian (thermal) distribution (via sample_maxwellian_velocity()).

    Returns:
        particles: List of SemanticTensor objects, each representing a fluid particle with position, velocity, and tensor attributes
    """
    if Lambda3SimConfig.MODE in ["stable", "super_stable"]:
        particles = []
        grid_size = int(np.sqrt(Lambda3SimConfig.N))
        spacing = Lambda3SimConfig.L / grid_size
        for i in range(grid_size):
            for j in range(grid_size):
                if len(particles) >= Lambda3SimConfig.N:
                    break
                x = i * spacing + spacing / 2
                y = j * spacing + spacing / 2
                position = np.array([x, y], dtype=np.float64)
                velocity = np.array([np.sin(i * 0.1) * 1e-4, np.cos(j * 0.1) * 1e-4])
                temperature = Lambda3SimConfig.T
                particle = SemanticTensor(position, velocity, temperature)
                particles.append(particle)
    else:
        positions = np.random.uniform(0, Lambda3SimConfig.L, (Lambda3SimConfig.N, 2))
        velocities = sample_maxwellian_velocity()
        temperatures = np.full(Lambda3SimConfig.N, Lambda3SimConfig.T)
        # temperatures = Lambda3SimConfig.T + 0.5 * np.sin(positions[:, 0] / Lambda3SimConfig.L * 2 * np.pi)
        particles = [
            SemanticTensor(positions[i], velocities[i], temperatures[i])
            for i in range(Lambda3SimConfig.N)
        ]
    return particles

# Initialize all SemanticTensor particles for the simulation
SemanticTensors = initialize_particles()

# ==============================
# 4. Event Tracking & History Data Structures
# ==============================
# --- Lists to record key events and metrics throughout the simulation ---
DeltaLambdaC_events = []         # List of all detected pulsation (ΔΛC) events
Split_events_pos = []            # Positions of split events for visualization
Q_Lambda_history = []            # Q_Λ (topological charge) for each particle at each step
LambdaF_history = []             # Lambda_F (progression vector) for each particle at each step
Q_Lambda_sum_history = []        # Global sum of Q_Λ (topological invariant)
Q_Lambda_std_history = []        # Std of Q_Λ (spatial/topological disorder indicator)

def apply_wall_boundary(tensor, config):
    """
    Apply wall reflection and/or friction at system edges to the particle.
    """
    if config.WALL_REFLECT_ON:
        if tensor.position[0] < 5 or tensor.position[0] > config.L - 5:
            tensor.Lambda_F[0] *= -1
        if tensor.position[1] < 5 or tensor.position[1] > config.L - 5:
            tensor.Lambda_F[1] *= -1

    if config.WALL_FRICTION_ON:
        friction = 0.90
        if tensor.position[0] < 5 or tensor.position[0] > config.L - 5:
            tensor.Lambda_F[0] *= friction
        if tensor.position[1] < 5 or tensor.position[1] > config.L - 5:
            tensor.Lambda_F[1] *= friction

def apply_obstacle_boundary(tensor, config):
    """Zero out progression vector and mark as obstacle if inside an obstacle."""
    if config.OBSTACLE_ON and is_in_obstacle(tensor.position, config):
        tensor.Lambda_F[:] = 0
        tensor.is_obstacle = True
    else:
        tensor.is_obstacle = False

def inject_external_noise(tensor, config):
    """Inject random tensor values at the boundaries (external perturbation)."""
    if config.EXTERNAL_INJECTION_NOISE_ON:
        if (np.random.rand() < config.INJECTION_NOISE_LEVEL and
            (tensor.position[0] < 5 or tensor.position[0] > config.L - 5)):
            tensor.Lambda = np.random.rand(*tensor.Lambda.shape)
            tensor.sigma_s = np.random.rand()

def inject_rotational_tension(tensor, config):
    """Inject rotational tension (vorticity) into Lambda_F for initial turbulence."""
    if config.EXTERNAL_INJECTION_OMEGA_ON:
        center = np.array([config.L / 2, config.L / 2])
        omega = config.INJECTION_OMEGA_LEVEL
        strength = config.INJECTION_OMEGA_STRENGTH
        rel = tensor.position - center
        norm = np.linalg.norm(rel) + 1e-8
        perp = np.array([-rel[1], rel[0]])
        rotational = strength * omega * perp / norm
        tensor.Lambda_F += rotational

def classify_tensor_state(tensor):
    if tensor.sigma_s > 0.95:
        tensor.state = 'equilibrium'
    elif tensor.sigma_s > 0.5:
        tensor.state = 'nonequilibrium'
    else:
        tensor.state = 'turbulence_burst'

def initialize_tensor_properties(tensor, config):
    """Apply all initial conditions, boundaries, and external fields to a SemanticTensor."""
    if config.WALL_ON:
        apply_wall_boundary(tensor, config)
    apply_obstacle_boundary(tensor, config)
    inject_external_noise(tensor, config)
    inject_rotational_tension(tensor, config)
    # Initialize Q_Lambda and adaptive timestep
    tensor.update_QLambda()
    tensor.prev_QLambda = tensor.QLambda
    tensor.local_dt = adaptive_dt(tensor)

# --- Apply all initialization logic to each SemanticTensor (particle) ---
for tensor in SemanticTensors:
    initialize_tensor_properties(tensor, Lambda3SimConfig)

# ==============================
# 5. Main Simulation Loop: State Update, Event Detection, and Tensor Field Recording
# ==============================

# --- Main event and network analysis history arrays ---
lambda3_events = []          # Records all ΔΛC (pulsation) event information for network analysis
positions_only = []          # Stores only the positions of ΔΛC events (for spatial heatmaps, etc.)
QLambdas_E = []              # Q_Λ value at each ΔΛC event

Q_jump_events = []           # History of Q_Λ "jumps" (topological transitions)
Q_jump_positions = []
Q_jump_intensities = []
Q_jump_threshold = Lambda3SimConfig.Q_JUMP_THRESHOLD

# --- Tensor attribute histories (determinant, trace, eigenvalues, etc. for each particle at each step) ---
det_history   = np.zeros((Lambda3SimConfig.Nsteps, Lambda3SimConfig.N))
trace_history = np.zeros((Lambda3SimConfig.Nsteps, Lambda3SimConfig.N))
norm_history  = np.zeros((Lambda3SimConfig.Nsteps, Lambda3SimConfig.N))
eig1_history  = np.zeros((Lambda3SimConfig.Nsteps, Lambda3SimConfig.N))
eig2_history  = np.zeros((Lambda3SimConfig.Nsteps, Lambda3SimConfig.N))
angle_history = np.zeros((Lambda3SimConfig.Nsteps, Lambda3SimConfig.N))

# --- Network & chaos diagnostics ---
LambdaF_var_history = []                 # Variance of Lambda_F (progression vector) for each step
prev_LambdaF = np.mean([tensor.Lambda_F for tensor in SemanticTensors], axis=0)
DeltaLambdaC_count_history = []          # Number of ΔΛC events per step
Q_Lambda_std_history_global = []         # Global std of Q_Λ per step
Q_Lambda_std_history_local_mean = []     # Local (per-cluster) mean std of Q_Λ per step
Q_Lambda_std_history_local_all = []      # Local stds (full distribution) of Q_Λ per step
vel_init = np.linalg.norm([tensor.Lambda_F for tensor in SemanticTensors], axis=1)
mean_local_norm_std_history = []
chaos_series = []                        # Binary flag: chaos detected (1) or not (0) for each step

# --- Diagnostic variables for chaos detection & explanation ---
chaos_flags = []             # (Alternative label for chaos series; can also include reason codes)
angle_std_list = []          # Standard deviation of angle_det for each step
norm_jump_list = []          # Norm jump value per step
eig1_sign_change_list = []   # Number of sign changes in first eigenvalue per step
det_jump_list = []           # Jump in determinant per step
trace_std_list = []          # Standard deviation of trace per step

# --- Grid size for tensor field visualizations ---
Nx = Lambda3SimConfig.GRID_NX
Ny = Lambda3SimConfig.GRID_NY
save_interval = Lambda3SimConfig.SAVE_INTERVAL
frames = []                  # Storage for animation/video frames (visualization of system)
l3_results = []

# --- Utility functions for tensor field analysis (vorticity, divergence) ---
def compute_curl_LambdaF(LambdaF_grid):
    # Compute curl (vorticity) of the progression vector field
    dFdx = np.gradient(LambdaF_grid[:, :, 1], axis=0)
    dFdy = np.gradient(LambdaF_grid[:, :, 0], axis=1)
    return dFdx - dFdy

def compute_div_LambdaF(LambdaF_grid):
    # Compute divergence of the progression vector field
    dFdx = np.gradient(LambdaF_grid[:, :, 0], axis=0)
    dFdy = np.gradient(LambdaF_grid[:, :, 1], axis=1)
    return dFdx + dFdy

def reflect_on_obstacle(tensor, config):
    # Rectangular obstacles
    for (x_rng, y_rng) in config.RECT_OBSTACLES:
        x0, x1 = x_rng
        y0, y1 = y_rng
        # Check for "hit" at the edge with a ±margin
        margin = 1.0
        if (x0 - margin <= tensor.position[0] <= x1 + margin) and (y0 - margin <= tensor.position[1] <= y1 + margin):
            # Reflect the velocity component based on proximity to the wall
            if abs(tensor.position[0] - x0) < margin or abs(tensor.position[0] - x1) < margin:
                tensor.Lambda_F[0] *= -1  # Reflect in the x direction
            if abs(tensor.position[1] - y0) < margin or abs(tensor.position[1] - y1) < margin:
                tensor.Lambda_F[1] *= -1  # Reflect in the y direction

    # Circular obstacles
    for (center, radius) in config.CIRC_OBSTACLES:
        cx, cy = center
        dist = np.linalg.norm(tensor.position - np.array([cx, cy]))
        margin = 1.0
        if radius - margin < dist < radius + margin:
            direction = (tensor.position - np.array([cx, cy])) / (dist + 1e-8)
            # Reflect the normal component of the velocity vector (general formula for wall reflection)
            tensor.Lambda_F -= 2 * np.dot(tensor.Lambda_F, direction) * direction

def inject_rotational_flow(tensor, config, step):
    """Inject rotational vorticity (ΛF boost) for turbulence/rotation experiments."""
    if (
        getattr(config, "EXTERNAL_INJECTION_OMEGA_ON", False)
        and config.OMEGA_INJECTION_START <= step < config.OMEGA_INJECTION_END
    ):
        core_pos = np.array([config.L / 2, config.L / 2])
        omega = config.INJECTION_OMEGA_LEVEL
        strength = config.INJECTION_OMEGA_STRENGTH
        tensor.Lambda_F += rotational_field(tensor.position, core_pos, omega, strength)

def inject_external_noise(tensor, config, step):
    """Inject external random noise (for stochastic experiments) within a given time window."""
    if (
        getattr(config, 'EXTERNAL_INJECTION_NOISE_ON', False)
        and config.NOISE_INJECTION_START <= step < config.NOISE_INJECTION_END
    ):
        tensor.Lambda += np.random.normal(0, 0.1, size=tensor.Lambda.shape)
        tensor.sigma_s = np.abs(np.random.normal(0.8, 0.2))

def apply_wall_conditions(tensor, config):
    """Apply wall boundary conditions: reflection and friction (only at system edge)."""
    if getattr(config, "WALL_ON", False):
        if getattr(config, 'WALL_REFLECT_ON', False):
            if tensor.position[0] < 5 or tensor.position[0] > config.L - 5:
                tensor.Lambda_F[0] *= -1
            if tensor.position[1] < 5 or tensor.position[1] > config.L - 5:
                tensor.Lambda_F[1] *= -1
        if getattr(config, 'WALL_FRICTION_ON', False):
            friction = 0.9
            if tensor.position[0] < 5 or tensor.position[0] > config.L - 5:
                tensor.Lambda_F[0] *= friction
            if tensor.position[1] < 5 or tensor.position[1] > config.L - 5:
                tensor.Lambda_F[1] *= friction

def update_scalar_quantities(tensor):
    """Update local scalars (sigma_s, rho_T, efficiency)."""
    tensor.sigma_s = np.linalg.norm(tensor.Lambda_F)
    tensor.rho_T = tensor.sigma_s
    tensor.eff = tensor.calc_efficiency()

def update_topological_invariant(tensor, step, Q_jump_threshold):
    """Update Q_Lambda and record topological jumps if detected."""
    tensor.update_QLambda()
    prev = getattr(tensor, "prev_QLambda", None)
    curr = tensor.QLambda
    if prev is not None and curr is not None:
        jump = abs(curr - prev)
        jump = min(jump, 2 * np.pi - jump)
        if jump > Q_jump_threshold:
            Q_jump_events.append((step, tensor, tensor.position.copy(), jump))
            Q_jump_positions.append(tensor.position.copy())
            Q_jump_intensities.append(jump)
    tensor.prev_QLambda = curr

def update_position_and_obstacle_bounce(tensor, config):
    """Update position with periodic boundary, but prevent entering obstacles."""
    # Save current position
    prev_pos = tensor.position.copy()
    # Normal position update
    tensor.position += tensor.Lambda_F * config.dt
    tensor.position = tensor.position % config.L  # periodic

    # If inside obstacle, move back and reflect/dampen velocity
    if is_in_obstacle(tensor.position, config):
        tensor.position = prev_pos
        tensor.Lambda_F *= -0.2  # or *= -1.0 for perfect reflection

def update_tensor_state(tensor, step, SemanticTensors, config=Lambda3SimConfig):
    """
    Update the state of a single SemanticTensor (fluid particle) for one simulation step.
    Applies external fields, computes local progression, injects noise/rotation,
    enforces wall/obstacle boundaries, prevents obstacle penetration, and updates invariants.
    """
    # 0. External field (gravity)
    tensor.Lambda_F += config.GRAVITY * config.dt

    # 1. Local neighborhood interaction
    neighbors = periodic_kdtree_neighbors(tensor, SemanticTensors)
    update_LambdaF_with_neighbors(tensor, neighbors)
    tensor.sigma_s = compute_sigma_s(tensor, neighbors)

    # 2. Rotational flow injection (if enabled and in window)
    inject_rotational_flow(tensor, config, step)

    # 3. External noise injection (if enabled and in window)
    inject_external_noise(tensor, config, step)

    # 4. Wall boundary (reflection/friction at edges)
    apply_wall_conditions(tensor, config)

    # 5. Obstacle boundary (surface reflection)
    reflect_on_obstacle(tensor, config)

    # 6. Update position and **prevent obstacle penetration**
    update_position_and_obstacle_bounce(tensor, config)  # ← safe_ver

    # 7. Update local scalar/efficiency quantities
    update_scalar_quantities(tensor)

    # 8. Update Q_Lambda and check for topological jumps
    update_topological_invariant(tensor, step, config.Q_JUMP_THRESHOLD)

# --- ΔΛC Event Detection and Application (per step for all particles) ---
def detect_and_apply_DeltaLambdaC(SemanticTensors, step):
    events_step = []
    DeltaLambdaC_count = 0
    N = len(SemanticTensors)
    for i in range(N):
        tensor_a = SemanticTensors[i]
        tensor_b = SemanticTensors[(i + 1) % N]
        neighbors = periodic_kdtree_neighbors(tensor_a, SemanticTensors)
        event_type = classify_transaction(tensor_a, tensor_b, neighbors=neighbors)
        if event_type != "Stable":
            events_step.append((step, i, (i + 1) % N, event_type, tensor_a.position.copy(), tensor_b.position.copy()))
            if event_type == "Split":
                Split_events_pos.append(tensor_a.position.copy())
            handle_event(event_type, tensor_a, tensor_b)
            DeltaLambdaC_count += 1
    DeltaLambdaC_events.extend(events_step)
    DeltaLambdaC_count_history.append(DeltaLambdaC_count)
    for event in events_step:
        step_, idx_a, _, _, pos_a, _ = event
        qlambda = SemanticTensors[idx_a].QLambda
        lambda3_events.append((pos_a.copy(), step_, qlambda))
        positions_only.append(pos_a.copy())
        QLambdas_E.append(qlambda)

# ==============================
# 6. Main Loop: ΛF Update, ΔΛC Event Detection, and Visualization Data Collection
# ==============================

# Set up spatial grids for field visualization (Nx × Ny grid covering the system domain)
x = np.linspace(0, Lambda3SimConfig.L, Nx)
y = np.linspace(0, Lambda3SimConfig.L, Ny)
X, Y = np.meshgrid(x, y)

# Main loop: core simulation time evolution
for step in range(Lambda3SimConfig.Nsteps):
    # 1. Update state for all particles (position, Lambda, Lambda_F, Q_Lambda, etc.)
    for tensor in SemanticTensors:
        update_tensor_state(tensor, step, SemanticTensors)

    # 2. Detect and apply ΔΛC events (pulsation transitions: Merge, Split, etc.)
    detect_and_apply_DeltaLambdaC(SemanticTensors, step)

    # 3. Record time series/statistics for each step (topological, clustering, and local fluctuation analysis)
    Q_Lambda_vals = [tensor.QLambda for tensor in SemanticTensors]
    Q_Lambda_sum_history.append(np.sum(Q_Lambda_vals))
    Q_Lambda_std_history_global.append(np.std(Q_Lambda_vals))
    Q_Lambda_history.append(np.sum([tensor.eff for tensor in SemanticTensors]))
    LambdaF_history.append(np.mean([tensor.Lambda_F for tensor in SemanticTensors], axis=0))

    # --- Local spatial clusters based on proximity (for local Q_Lambda std, etc.) ---
    positions = np.array([tensor.position for tensor in SemanticTensors])
    tree_p = cKDTree(positions)
    r_spatial = 5
    pairs_p = tree_p.query_pairs(r=r_spatial)
    G_p = nx.Graph()
    G_p.add_nodes_from(range(len(SemanticTensors)))
    for i, j in pairs_p:
        G_p.add_edge(i, j)
    components_p = list(nx.connected_components(G_p))

    local_stds = []
    for comp in components_p:
        comp = list(comp)
        q_vals = [Q_Lambda_vals[i] for i in comp if i < len(Q_Lambda_vals)]
        if len(q_vals) > 1:
            local_stds.append(np.std(q_vals))
    Q_Lambda_std_history_local_mean.append(np.mean(local_stds) if local_stds else 0)
    Q_Lambda_std_history_local_all.append(local_stds)

    # Local norm std for each spatial cluster
    local_norm_stds = []
    for comp in components_p:
        comp = list(comp)
        norms = [SemanticTensors[i].lambda_hist["norm"][-1] for i in comp]
        if len(norms) > 1:
            local_norm_stds.append(np.std(norms))
    mean_local_norm_std = np.mean(local_norm_stds) if local_norm_stds else np.nan
    mean_local_norm_std_history.append(mean_local_norm_std)

    # ΛF variance tracking (network/chaos analysis)
    curr_LambdaF = np.mean([tensor.Lambda_F for tensor in SemanticTensors], axis=0)
    LambdaF_var_history.append(np.linalg.norm(curr_LambdaF - prev_LambdaF))
    prev_LambdaF = curr_LambdaF.copy()

    # 4. Per-particle matrix/tensor history (det, trace, norm, eigenvalues, angle, etc.)
    for i, tensor in enumerate(SemanticTensors):
        if tensor.Lambda.shape == (2, 2):
            Lambda_mat = tensor.Lambda
        elif tensor.Lambda.size == 4:
            Lambda_mat = tensor.Lambda.reshape(2, 2)
        elif tensor.Lambda.size == 9:
            Lambda_mat = tensor.Lambda.reshape(3, 3)
        else:
            raise ValueError(f"Unexpected Lambda shape: {tensor.Lambda.shape}")

        det = np.linalg.det(Lambda_mat)
        eigvals = np.linalg.eigvals(Lambda_mat)
        trace = np.trace(Lambda_mat)
        norm = np.linalg.norm(Lambda_mat)
        angle = np.angle(det)
        det_history[step, i] = det.real
        trace_history[step, i] = trace.real
        norm_history[step, i] = norm.real
        eig1_history[step, i] = eigvals[0].real
        eig2_history[step, i] = eigvals[1].real
        angle_history[step, i] = angle.real

    # 5. Chaos detection and diagnostics based on Λ³ network and Q_Λ fluctuations
    # (see original code above for full details, includes adaptive detection intervals, metrics, etc.)
    # Results stored in chaos_flags, angle_std_list, norm_jump_list, eig1_sign_change_list, det_jump_list, trace_std_list

    # 6. Visualization data collection (save fields every SAVE_INTERVAL steps)
    if step % save_interval == 0:
        # --- 1. Λ max Dimension　（4, 9, 16...）---
        max_Lambda_size = max(t.Lambda.size for t in SemanticTensors)

        eff_field = np.zeros((Nx, Ny))
        sigma_field = np.zeros((Nx, Ny))
        count_grid = np.zeros((Nx, Ny))
        Lambda_grid = np.zeros((Nx, Ny, max_Lambda_size))
        LambdaF_grid = np.zeros((Nx, Ny, 2))
        for t in SemanticTensors:
            ix = min(max(int((t.position[0] % Lambda3SimConfig.L) / Lambda3SimConfig.L * Nx), 0), Nx - 1)
            iy = min(max(int((t.position[1] % Lambda3SimConfig.L) / Lambda3SimConfig.L * Ny), 0), Ny - 1)
            eff_field[ix, iy] += t.eff
            sigma_field[ix, iy] += t.sigma_s
            # --- ΛVector zero---
            Lambda_val = np.zeros(max_Lambda_size)
            Lambda_val[:t.Lambda.size] = t.Lambda.reshape(-1)
            Lambda_grid[ix, iy] += Lambda_val
            LambdaF_grid[ix, iy] += t.Lambda_F
            count_grid[ix, iy] += 1
        for i in range(Nx):
            for j in range(Ny):
                if count_grid[i, j] > 0:
                    eff_field[i, j] /= count_grid[i, j]
                    sigma_field[i, j] /= count_grid[i, j]
                    Lambda_grid[i, j] /= count_grid[i, j]
                    LambdaF_grid[i, j] /= count_grid[i, j]
                else:
                    LambdaF_grid[i, j] = [0, 0]
        local_entropy = -eff_field * np.log(np.abs(eff_field) + 1e-12)
        curl_map = compute_curl_LambdaF(LambdaF_grid)
        div_map = compute_div_LambdaF(LambdaF_grid)
        # --- Lambda0_field　safe ---
        Lambda0_field = Lambda_grid[:, :, 0]
        grad_x, grad_y = np.gradient(Lambda0_field)
        grad_mag_map = np.sqrt(grad_x ** 2 + grad_y ** 2)
        # --- Event heatmap for Split events ---
        if Split_events_pos:
            Split_events_pos_arr = np.array(Split_events_pos)
            heatmap_event, _, _ = np.histogram2d(
                Split_events_pos_arr[:, 0], Split_events_pos_arr[:, 1],
                bins=(Nx, Ny), range=[[0, Lambda3SimConfig.L], [0, Lambda3SimConfig.L]]
            )
            heatmap_event = gaussian_filter(heatmap_event, sigma=2.0)
        else:
            heatmap_event = np.zeros((Nx, Ny))

        # --- Collect all metrics and fields for animation or later analysis ---
        last_idx = len(chaos_flags) - 1  # Most recent step index
        frames.append({
            "step": step,
            "entropy": local_entropy.copy(),
            "curl": curl_map.copy(),
            "div": div_map.copy(),
            "grad_mag": grad_mag_map.copy(),
            "event_density": heatmap_event.copy(),
            "LambdaF_grid": LambdaF_grid.copy(),
            # Chaos/transition metrics
            "chaos_flag": chaos_flags[last_idx]      if chaos_flags else None,
            "angle_std": angle_std_list[last_idx]    if angle_std_list else None,
            "norm_jump": norm_jump_list[last_idx]    if norm_jump_list else None,
            "eig1_sign_change": eig1_sign_change_list[last_idx] if eig1_sign_change_list else None,
            "det_jump": det_jump_list[last_idx]      if det_jump_list else None,
            "trace_std": trace_std_list[last_idx]    if trace_std_list else None
        })

         # Lambda_c, pressure, and stepwise averages for comparison (to NSE)
        lambda_c_div = compute_lambda_c_divergence(LambdaF_grid)
        lambda_c_lap = compute_lambda_c_laplacian(sigma_field)
        pressure_field_from_div = solve_poisson_2d(lambda_c_div)
        pressure_field_from_lap = solve_poisson_2d(lambda_c_lap)
        kinetic_energy_L3 = 0.5 * np.mean([np.dot(t.Lambda_F, t.Lambda_F) for t in SemanticTensors])

        l3_results.append({
            "step": step,
            "mean_sigma_s": np.mean(sigma_field),
            "mean_LambdaF": np.mean(np.linalg.norm(LambdaF_grid, axis=2)),
            "mean_lambda_c_div": np.mean(lambda_c_div),
            "mean_lambda_c_lap": np.mean(lambda_c_lap),
            "mean_pressure_div": np.mean(pressure_field_from_div),
            "mean_pressure_lap": np.mean(pressure_field_from_lap),
            "kinetic_energy_L3": kinetic_energy_L3,
        })

# ==============================
# 7.ΔΛC Event Information/Energy Array Construction
# ==============================

# --- Obstacle mask generation for visualization overlays ---
obstacle_mask = np.zeros((Nx, Ny))
for i in range(Nx):
    for j in range(Ny):
        x = (i / Nx) * Lambda3SimConfig.L
        y = (j / Ny) * Lambda3SimConfig.L
        if is_in_obstacle((x, y), Lambda3SimConfig):
            obstacle_mask[i, j] = 1.0

# --- Snapshots for key simulation steps (beginning, middle, end) ---
num_frames = len(frames)
snapshot_steps = [0, num_frames // 2, num_frames - 1]

for idx, step in enumerate(snapshot_steps):
    # 1. Local entropy map
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(
        frames[step]["entropy"].T,
        origin="lower",
        cmap="cividis",
        extent=[0, Lambda3SimConfig.L, 0, Lambda3SimConfig.L]
    )
    # Obstacle overlay
    ax.imshow(
        obstacle_mask.T,
        cmap='gray',
        alpha=0.3,
        extent=[0, Lambda3SimConfig.L, 0, Lambda3SimConfig.L]
    )
    plt.colorbar(im, ax=ax)
    plt.title(f"Λ³ Local Structural Entropy (step={step * Lambda3SimConfig.SAVE_INTERVAL})")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.tight_layout()
    plt.show()

    # 2. ΛF Streamplot (vector field visualization)
    fig2, ax2 = plt.subplots(figsize=(7, 6))
    U = frames[step]["LambdaF_grid"][:, :, 0].T
    V = frames[step]["LambdaF_grid"][:, :, 1].T
    speed = np.sqrt(U ** 2 + V ** 2)

    strm = ax2.streamplot(X, Y, U, V, color=speed, cmap="cool", density=1.3, linewidth=1)
    ax2.imshow(
        obstacle_mask.T,
        cmap='gray',
        alpha=0.3,
        extent=[0, Lambda3SimConfig.L, 0, Lambda3SimConfig.L]
    )
    ax2.set_title(f"ΛF Streamplot (step={step * Lambda3SimConfig.SAVE_INTERVAL})")
    ax2.set_xlabel("x")
    ax2.set_ylabel("y")
    plt.tight_layout()
    plt.show()


E_DeltaLambdaC_events = []
for event in DeltaLambdaC_events:
    # DeltaLambdaC_events: (step, i, j, event_type, pos_a, pos_b)
    step, i, j, event_type, pos_a, pos_b = event
    eff_a = SemanticTensors[i].eff
    sigma_s_a = SemanticTensors[i].sigma_s
    qlambda_a = SemanticTensors[i].QLambda  # Optionally store Q_Lambda here as well
    # Information energy: Negative log of efficiency × coherence (numerically stable)
    E = -np.log(np.abs(eff_a * sigma_s_a) + 1e-12)
    E_DeltaLambdaC_events.append((step, E, np.array(pos_a), qlambda_a))

# Separate arrays for plotting/analysis
steps_E = np.array([e[0] for e in E_DeltaLambdaC_events])
energies_E = np.array([e[1] for e in E_DeltaLambdaC_events])
positions_E = np.array([e[2] for e in E_DeltaLambdaC_events])
QLambdas_E = np.array([e[3] for e in E_DeltaLambdaC_events])

# =======================
# 8. Q_Lambda Jump Event Heatmap
# =======================
if Q_jump_positions:
    Q_jump_positions = np.array(Q_jump_positions)
    Q_jump_intensities = np.array(Q_jump_intensities)
    # 2D histogram (heatmap) for topological jump events, weighted by intensity
    jump_heatmap, _, _ = np.histogram2d(
        Q_jump_positions[:, 0], Q_jump_positions[:, 1],
        bins=60, weights=Q_jump_intensities, range=[[0, Lambda3SimConfig.L], [0, Lambda3SimConfig.L]]
    )
    plt.imshow(
        gaussian_filter(jump_heatmap.T, sigma=2.5),
        origin='lower', cmap="cool", extent=[0, Lambda3SimConfig.L, 0, Lambda3SimConfig.L]
    )
    plt.title("Q_Lambda Jump Event Density")
    plt.colorbar(label="Topological Breaks (Jump Intensity)")
    plt.show()

# ==============================
# 9. ΔΛC Event / Energy Density Visualization
# ==============================
if Split_events_pos:
    Split_events_pos = np.array(Split_events_pos)
    heatmap, xedges, yedges = np.histogram2d(
        Split_events_pos[:, 0], Split_events_pos[:, 1], bins=60, range=[[0, Lambda3SimConfig.L], [0, Lambda3SimConfig.L]]
    )
    plt.imshow(
        gaussian_filter(heatmap.T, sigma=2.5),
        origin='lower', cmap="inferno", extent=[0, Lambda3SimConfig.L, 0, Lambda3SimConfig.L]
    )
    plt.title("ΔΛC Split Intensity Field")
    plt.colorbar(label="Cumulative Event Density")
    plt.show()

# ==============================
# 10. Particle Network / Event Network Analysis (Dual Chaos Diagnosis)
# ==============================

### (A) Particle Network (Current Structural Snapshot)
# Build a spatial network of all particles based on proximity
positions_p = np.array([t.position for t in SemanticTensors])
tree_p = cKDTree(positions_p)
pairs_p = tree_p.query_pairs(r=Lambda3SimConfig.R_PARTICLE_CONNECT)
G_p = nx.Graph()
G_p.add_nodes_from(range(len(SemanticTensors)))
for i, j in pairs_p:
    G_p.add_edge(i, j)

# Calculate Q_Lambda values for all particles
Q_Lambda_vals = np.array([t.QLambda for t in SemanticTensors])
# Identify connected components (local clusters)
components_p = list(nx.connected_components(G_p))
# Compute standard deviation of Q_Lambda within each component (cluster)
stds_p = [np.std([Q_Lambda_vals[i] for i in comp]) for comp in components_p if len(comp) > 1]

# Visualize distribution of local topological disorder (within-patch fluctuations)
plt.figure()
plt.hist(stds_p, bins=30, alpha=0.7)
plt.xlabel("Q_Lambda Std (per component)")
plt.ylabel("Count")
plt.title("Local QΛ Disorder (All Particles)")
plt.show()

### (B) λ³ Event Network Structure Analysis (Accelerated via Spatial Filtering)
# Construct λ³ event network: nodes = events, edges = close in space/time/topology
lambda3_events = [(positions_E[i], steps_E[i], QLambdas_E[i]) for i in range(len(positions_E))]
positions_only = np.array([pos for pos, _, _ in lambda3_events])
G_lambda3 = build_lambda3_event_network(lambda3_events, positions_only)

# Analyze the λ³ event network components
components_lambda3 = list(nx.connected_components(G_lambda3))
stds_lambda3 = [
    np.std([QLambdas_E[i] for i in comp])
    for comp in components_lambda3 if len(comp) > 1
]

# Plot histogram of topological disorder for λ³ event network clusters
plt.figure()
plt.hist(stds_lambda3, bins=30, alpha=0.7)
plt.xlabel("Q_Lambda Std (per component)")
plt.ylabel("Count")
plt.title("Local QΛ Disorder (Λ³ Event Network)")
plt.show()

# ==============================
# 11. Time Series Plots of Conserved Quantities & Progression Vector, Network Statistics
# ==============================
# --- 1. Q_Lambda statistics & Progression Vector ---
def safe_zscore_plot(arr, label, color):
    arr = np.array(arr)
    std = np.std(arr)
    if std < 1e-8:
        plt.plot(arr, label=f'{label} (const)', color=color)
        print(f"[Warn] {label}: Skipping z-score (std too small: {std:.2e})")
    else:
        plt.plot(zscore(arr), label=label + " (z-score)", color=color)

# Q_Lambda sum: In a closed (periodic) system, the total topological charge (Q_Lambda) is theoretically conserved.
# Since the simulation uses a fixed grid and particles cannot exit the domain, the sum of Q_Lambda remains (nearly) constant by design.
# This is a physical conservation law: deviations may only occur due to numerical round-off or discrete event handling.
# Therefore, the z-score of Q_Lambda_sum_history may yield precision loss warnings, but this is expected and not an error.

plt.figure(figsize=(8, 5))
safe_zscore_plot(Q_Lambda_sum_history, "Q_Lambda sum (topo)", 'blue')
safe_zscore_plot(mean_local_norm_std_history, "Mean Local norm std (per comp.)", 'magenta')
safe_zscore_plot(DeltaLambdaC_count_history, "ΔΛC event count", 'orange')
plt.xlabel("Step")
plt.ylabel("Z-score or Value")
plt.legend()
plt.title("Q_Lambda sum & Local Chaos/Disorder Indicators")
plt.tight_layout()
plt.show()

plt.figure()
plt.plot(Q_Lambda_history, label="Q_Lambda (eff sum)")
plt.xlabel("Step")
plt.ylabel("Q_Lambda")
plt.legend()
plt.title("Sum of Tensor Efficiency Q_Lambda")
plt.show()

plt.figure()
LambdaF_history_arr = np.array(LambdaF_history)
if LambdaF_history_arr.shape[1] >= 2:
    plt.plot(LambdaF_history_arr[:, 0], label="LambdaF_x")
    plt.plot(LambdaF_history_arr[:, 1], label="LambdaF_y")
else:
    plt.plot(LambdaF_history_arr, label="LambdaF (mean)")
plt.xlabel("Step")
plt.ylabel("LambdaF")
plt.legend()
plt.title("Progression Vector LambdaF (mean)")
plt.show()

# --- 2. Spatial Gradient Magnitude ---
grad_x = np.gradient(Lambda_grid[:,:,0], axis=0)
grad_y = np.gradient(Lambda_grid[:,:,1], axis=1)
grad_mag = np.sqrt(grad_x**2 + grad_y**2)
plt.imshow(grad_mag.T, origin='lower', cmap='viridis', extent=[0, Lambda3SimConfig.L, 0, Lambda3SimConfig.L])
plt.title("∇Λ Field Magnitude")
plt.colorbar(label="Gradient Magnitude")
plt.xlabel("x")
plt.ylabel("y")
plt.show()

# --- 3. Causality between LambdaF and ΔΛC ---
corrs_LF_DLC = analyze_causality(LambdaF_var_history, DeltaLambdaC_count_history)
plt.figure()
plt.plot(corrs_LF_DLC, label="ΛF→ΔΛC")
plt.xlabel("Lag step")
plt.ylabel("Causal correlation")
plt.title("Causality: LambdaF → ΔΛC Events")
plt.legend()
plt.show()

# --- 4. Fourier Spectrum of LambdaF Fluctuations ---
fft_LF = np.abs(np.fft.fft(LambdaF_var_history - np.mean(LambdaF_var_history)))
plt.figure()
plt.plot(fft_LF[:len(fft_LF)//2])
plt.title("ΛF Fluctuation Spectrum (FFT)")
plt.xlabel("Frequency")
plt.ylabel("Amplitude")
plt.show()

# --- 5. Spatial Distribution of ΔΛC Event Energies ---
steps_E, energies_E, positions_E, qlambdas_E = [], [], [], []
for event in E_DeltaLambdaC_events:
    step, E, pos, qlambda = event
    steps_E.append(step)
    energies_E.append(E)
    positions_E.append(np.array(pos))
    qlambdas_E.append(qlambda)
steps_E = np.array(steps_E)
energies_E = np.array(energies_E)
positions_E = np.array(positions_E)
qlambdas_E = np.array(qlambdas_E)

heatmap_event, _, _ = np.histogram2d(
    positions_E[:, 0], positions_E[:, 1], bins=60,
    weights=energies_E, range=[[0, Lambda3SimConfig.L], [0, Lambda3SimConfig.L]]
)
plt.figure()
plt.imshow(gaussian_filter(heatmap_event.T, sigma=2), origin="lower", cmap="magma", extent=[0, Lambda3SimConfig.L, 0, Lambda3SimConfig.L])
plt.colorbar(label="Cumulative ΔΛC Energy")
plt.title("ΔΛC Event Energy Field")
plt.xlabel("x")
plt.ylabel("y")
plt.show()

# --- 6. Spatial Correlation (Energy vs. Gradient Field) ---
corr_val = np.sum(heatmap_event * grad_mag) / (np.sqrt(np.sum(heatmap_event**2)) * np.sqrt(np.sum(grad_mag**2)) + 1e-8)

# --- 7. Visualization of Pulsation Wave Propagation Paths ---
plt.figure()
for idx, (t0, E0, pos0) in enumerate(zip(steps_E, energies_E, positions_E)):
    time_mask = np.abs(steps_E - t0) < 10
    dist_mask = np.linalg.norm(positions_E - pos0, axis=1) < 15
    mask = time_mask & dist_mask
    after_x = positions_E[mask][:, 0]
    after_y = positions_E[mask][:, 1]
    if len(after_x) > 2:
        plt.plot(after_x, after_y, 'o-', alpha=0.5)
plt.title("Pulsation Wave Propagation (ΔΛC events)")
plt.xlabel("x")
plt.ylabel("y")
plt.show()

# --- 8. Energy Distribution of ΔΛC Events (Scatter Plot) ---
plt.figure()
plt.scatter(positions_E[:,0], positions_E[:,1], c=energies_E, cmap="hot", s=5)
plt.colorbar(label="ΔΛC event energy")
plt.title("ΔΛC Events (by energy)")
plt.xlabel("x")
plt.ylabel("y")
plt.show()

# --- 9. Temperature-based Velocity Field (Quiver) ---
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

q = axes[0].quiver(
    X, Y, LambdaF_grid[:, :, 0], LambdaF_grid[:, :, 1], sigma_field,
    cmap='jet', scale=50, width=0.002
)
cb1 = fig.colorbar(q, ax=axes[0])
cb1.set_label('Temperature-based σₛ')
axes[0].set_title("Velocity Field (Λ_F)\nwith Temperature-based σₛ")
axes[0].set_xlabel("x")
axes[0].set_ylabel("y")

# --- 10. Spatial Distribution of σₛ (Heatmap) ---
im = axes[1].imshow(
    sigma_field.T, extent=(0, Lambda3SimConfig.L, 0, Lambda3SimConfig.L),
    origin='lower', cmap='coolwarm', aspect='auto'
)
cb2 = fig.colorbar(im, ax=axes[1])
cb2.set_label('Synchronization rate σₛ')
axes[1].set_title("Spatial Distribution of σₛ")
axes[1].set_xlabel("x")
axes[1].set_ylabel("y")

# --- 11. Topological Invariant QΛ (Timeseries) ---
axes[2].plot(Q_Lambda_sum_history, label='QΛ (Topological Invariant)', color='navy')
for event in DeltaLambdaC_events:
    axes[2].axvline(event[0], color='red', linestyle='--', alpha=0.4)
axes[2].legend()
axes[2].set_title("Topological Invariant QΛ\nand ΔΛC Events over Time")
axes[2].set_xlabel("Step")
axes[2].set_ylabel("QΛ (sum)")

plt.tight_layout()
plt.show()

# --- 12. State Classification---
classification_map = np.zeros_like(sigma_field)
classification_map[sigma_field > 0.8] = 0  # equilibrium
classification_map[(sigma_field <= 0.8) & (sigma_field > 0.4)] = 1  # non-equilibrium
classification_map[sigma_field <= 0.4] = 2  # turbulence burst

fig, ax = plt.subplots(figsize=(7,6))
im = ax.imshow(classification_map.T, extent=(0, Lambda3SimConfig.L, 0, Lambda3SimConfig.L), origin='lower', cmap='Set1')
cbar = plt.colorbar(im, ticks=[0,1,2], label='State')
cbar.ax.set_yticklabels(['Equilibrium', 'Non-equilibrium', 'Turbulence Burst'])
ax.set_title("Λ³ State Classification")
ax.set_xlabel("X")
ax.set_ylabel("Y")
plt.tight_layout()
plt.show()

# ==============================
# Reconstruction of ΔΛC Event Network Based on Λ³ Distance Function
# ==============================

# --- Prepare the event list ---
lambda3_events = [
    (positions_E[i], steps_E[i], QLambdas_E[i])
    for i in range(len(positions_E))
]
positions_only = np.array([pos for pos, _, _ in lambda3_events])

# --- Construct the new Λ³ event network ---
G_lambda3 = build_lambda3_event_network(lambda3_events, positions_only)

# --- Visualization of the network ---
node_list = list(G_lambda3.nodes)
node_colors = np.array([energies_E[i] for i in node_list])
pos_map = {i: (positions_E[i, 0], positions_E[i, 1]) for i in node_list}
vmin, vmax = node_colors.min(), node_colors.max()
plt.figure(figsize=(8, 6))
nodes = nx.draw_networkx_nodes(
    G_lambda3, pos=pos_map,
    nodelist=node_list, node_size=15,
    node_color=node_colors, cmap="plasma",
    vmin=vmin, vmax=vmax
)
nx.draw_networkx_edges(G_lambda3, pos=pos_map, alpha=0.3)
plt.colorbar(nodes, label="Event Energy")
plt.title("Lambda³ Distance-Based Event Network")
plt.xlabel("x"); plt.ylabel("y")
plt.tight_layout(); plt.show()

# --- Distribution of phase component count and fluctuations ---
components_lambda3 = list(nx.connected_components(G_lambda3))

stds_lambda3 = [
    np.std([QLambdas_E[i] for i in comp])
    for comp in components_lambda3 if len(comp) > 1
]

plt.figure()
plt.hist(stds_lambda3, bins=30, alpha=0.7)
plt.xlabel("Q_Lambda Std (per component)")
plt.ylabel("Count")
plt.title("Local QΛ Disorder (Λ³ Event Network)")
plt.show()

# --- Network centrality analysis (Lambda³) ---
degrees_lambda3 = dict(G_lambda3.degree())
most_influential = max(degrees_lambda3, key=degrees_lambda3.get)

plt.figure(figsize=(6,5))
plt.scatter(positions_E[:,0], positions_E[:,1], c=energies_E, cmap="hot", s=5, alpha=0.6)
plt.scatter(positions_E[most_influential,0], positions_E[most_influential,1],
            color='aqua', s=60, edgecolor='black', label='Most Central')
plt.legend()
plt.title("ΔΛC Events & Most Central Source (Λ³)")
plt.tight_layout()
plt.show()

# --- Q_Lambda Jump Events: Time Series ---
config = Lambda3SimConfig()
Q_jump_counts = np.zeros(config.Nsteps)
for event in Q_jump_events:
    step = event[0]
    Q_jump_counts[step] += 1

def safe_zscore(x):
    std = np.std(x)
    if std < 1e-12:  # or some tiny threshold
        return np.zeros_like(x)
    else:
        return (x - np.mean(x)) / std

z_sum = safe_zscore(Q_Lambda_sum_history)
z_jump = safe_zscore(Q_jump_counts)
z_dlc  = safe_zscore(DeltaLambdaC_count_history)

plt.figure(figsize=(10,4))
plt.plot(z_sum, label='Q_Lambda sum (z-score)', color='blue')
plt.plot(z_jump, label='Q_Lambda jump count (z-score)', color='lime')
plt.plot(z_dlc, label='ΔΛC event count (z-score)', color='orange')
plt.xlabel("Step")
plt.ylabel("Z-score")
plt.legend()
plt.title("Topological Q_Lambda Sum & Event Counts (Z-score Normalized)")
plt.tight_layout()
plt.show()

# --- Degree Distribution of the Λ³ Event Network ---
degrees_list = [G_lambda3.degree(n) for n in G_lambda3.nodes()]
plt.figure()
plt.hist(degrees_list, bins=50, log=True)
plt.xlabel('Degree')
plt.ylabel('Count (log)')
plt.title('Degree Distribution of Λ³ Event Network (log-log)')
plt.yscale('log')
plt.xscale('log')
plt.show()

# --- Betweenness Centrality (Approximate, for Particle Network G_p) ---
num_nodes_Gp = G_p.number_of_nodes()
k_betw = min(200, num_nodes_Gp)
betweenness = list(nx.betweenness_centrality(G_p, k=k_betw, seed=42).values())

plt.figure()
plt.hist(betweenness, bins=50, log=True)
plt.xlabel('Betweenness')
plt.ylabel('Count (log)')
plt.title('Betweenness Centrality Distribution (Particle Network, log-log)')
plt.yscale('log')
plt.xscale('log')
plt.tight_layout()
plt.show()

# --- Component Size Distributions ---
# Particle network component sizes
comp_sizes_p = [len(c) for c in components_p]
# ΔΛC event network component sizes
comp_sizes_lambda3 = [len(c) for c in components_lambda3]

plt.figure(figsize=(12,5))
# Particle network
plt.subplot(1,2,1)
plt.hist(comp_sizes_p, bins=50, log=True)
plt.xlabel("Component Size (Particle)")
plt.ylabel("Count (log)")
plt.title("Spatial Particle Network: Component Size Distribution")
# ΔΛC event network
plt.subplot(1,2,2)
plt.hist(comp_sizes_lambda3, bins=50, log=True)
plt.xlabel("Component Size (ΔΛC Events)")
plt.ylabel("Count (log)")
plt.title("Pulsation Event Network: Component Size Distribution")
plt.tight_layout()
plt.show()

# --- Time Series of Efficiency, Local Entropy, and ΔΛC Event Count (Z-score Normalized) ---
entropy_series = [np.mean(frame["entropy"]) for frame in frames]
DeltaLambdaC_count_series = DeltaLambdaC_count_history[:len(entropy_series)]

# Z-score normalization
entropy_z = zscore(entropy_series)
DeltaLambdaC_z = zscore(DeltaLambdaC_count_series)

plt.figure()
plt.plot(entropy_z, label='Tensor entropy (z-score)')
plt.plot(DeltaLambdaC_z, label='ΔΛC event count (z-score)')
plt.xlabel('Frame')
plt.legend()
plt.title("Entropy and ΔΛC Event Count (Z-score, Frame Series)")
plt.show()

# --- Velocity Distribution Comparison & KS Test ---
vel_now = np.linalg.norm([tensor.Lambda_F for tensor in SemanticTensors], axis=1)
plt.figure()
plt.hist(vel_init, bins=40, alpha=0.5, label='Initial')
plt.hist(vel_now, bins=40, alpha=0.5, label='Current')
plt.legend(); plt.title("Velocity Norm Distribution (Initial vs. Current)")
plt.show()
stat, pval = ks_2samp(vel_init, vel_now)

# --- Q_Lambda Statistics Per Phase Component (sum, std, min, max) ---
component_stats = []
num_events = len(positions_E)
for comp in components_lambda3:
    comp_indices = [i for i in comp if i < num_events]
    if len(comp_indices) <= 1:
        continue
    q_vals = QLambdas_E[comp_indices]
    total = np.sum(q_vals)
    std = np.std(q_vals)
    min_ = np.min(q_vals)
    max_ = np.max(q_vals)
    component_stats.append({
        "Component Size": len(comp_indices),
        "Q_Lambda Sum": total,
        "Q_Lambda Std": std,
        "Q_Lambda Min": min_,
        "Q_Lambda Max": max_,
    })
df_stats = pd.DataFrame(component_stats)

plt.figure()
plt.hist(df_stats[df_stats["Component Size"] > 2]["Q_Lambda Std"], bins=30, alpha=0.7)
plt.xlabel("Q_Lambda Std (per component)")
plt.ylabel("Count")
plt.title("Distribution of Local Q_Lambda Disorder (Component Size > 2)")
plt.show()

# --- Display of Anomalous Components ---
anomaly_comps = df_stats[(df_stats["Q_Lambda Std"] > 0.5) & (df_stats["Component Size"] > 2)]
for idx, row in anomaly_comps.iterrows():
    print(f"Anomalous Component #{idx}:")
    print(row)

# --- Network Centrality Comparison (Particle Network G_p) ---
num_nodes_Gp = G_p.number_of_nodes()
k_Gp = min(100, num_nodes_Gp)

degree_centrality = nx.degree_centrality(G_p)
betweenness_centrality = nx.betweenness_centrality(G_p, k=k_Gp, seed=42)

plt.figure(figsize=(6,5))
plt.scatter(list(degree_centrality.values()), list(betweenness_centrality.values()), alpha=0.6)
plt.xlabel("Degree Centrality")
plt.ylabel("Betweenness Centrality")
plt.title("Centrality Comparison in Particle Network (G_p)")
plt.grid(True)
plt.tight_layout()
plt.show()

# --- Centrality Analysis in the ΔΛC Event Network (G_lambda3) ---
num_nodes_Gl3 = G_lambda3.number_of_nodes()
k_Gl3 = min(100, num_nodes_Gl3)

degree_centrality_events = nx.degree_centrality(G_lambda3)
betweenness_centrality_events = nx.betweenness_centrality(G_lambda3, k=k_Gl3, seed=42)

plt.figure(figsize=(6, 5))
plt.scatter(
    list(degree_centrality_events.values()),
    list(betweenness_centrality_events.values()),
    alpha=0.6
)
plt.xlabel("Degree Centrality (ΔΛC Network)")
plt.ylabel("Betweenness Centrality (ΔΛC Network)")
plt.title("Centrality Comparison in ΔΛC Event Network")
plt.grid(True)
plt.tight_layout()
plt.show()

# --- Visualization of Propagation Paths (Shortest Paths, Sampled) ---
most_influential = max(degree_centrality, key=degree_centrality.get)
paths = nx.single_source_shortest_path(G_lambda3, most_influential, cutoff=5)
max_paths = 20
colors = plt.cm.jet(np.linspace(0, 1, max_paths))
plt.figure(figsize=(7,6))
for idx, (target, path) in enumerate(list(paths.items())[:max_paths]):
    coords = positions_E[path]
    plt.plot(coords[:,0], coords[:,1], '-', alpha=0.5, color=colors[idx])
plt.scatter(positions_E[:,0], positions_E[:,1], c='gray', s=3, alpha=0.3)
plt.scatter(positions_E[most_influential,0], positions_E[most_influential,1],
            color='red', s=60, label="Central Source")
plt.legend()
plt.title("Propagation Paths from Most Central Node")
plt.xlabel("x")
plt.ylabel("y")
plt.tight_layout()
plt.show()

# --- Visualization of Convergence Rate λ_c (at Final or Saved Steps) ---
lambda_c_div = compute_lambda_c_divergence(LambdaF_grid)
lambda_c_lap = compute_lambda_c_laplacian(sigma_field)

plt.figure(figsize=(10, 4))

plt.subplot(1, 2, 1)
plt.imshow(lambda_c_div.T, origin='lower', cmap='coolwarm')
plt.title("Lambda_c (Divergence-based)")
plt.colorbar(label="Convergence Rate (div)")

plt.subplot(1, 2, 2)
plt.imshow(lambda_c_lap.T, origin='lower', cmap='plasma')
plt.title("Lambda_c (Laplacian-based)")
plt.colorbar(label="Convergence Rate (lap)")

plt.tight_layout()
plt.show()

# --- Solve Poisson Equation Using Lambda_c as Source Term ---
pressure_field_from_div = solve_poisson_2d(lambda_c_div)
pressure_field_from_lap = solve_poisson_2d(lambda_c_lap)

plt.figure(figsize=(12,5))

plt.subplot(1,2,1)
plt.imshow(pressure_field_from_div.T, origin='lower', cmap='coolwarm')
plt.title("Pressure Field (from Lambda_c_div)")
plt.colorbar(label="Pressure (div)")

plt.subplot(1,2,2)
plt.imshow(pressure_field_from_lap.T, origin='lower', cmap='plasma')
plt.title("Pressure Field (from Lambda_c_lap)")
plt.colorbar(label="Pressure (lap)")

plt.tight_layout()
plt.show()

def plot_multivariate_heatmaps(SemanticTensors, step_window=None):
    """
    Plot multivariate heatmaps for key tensor properties (angle_det, norm, eig1, eig2, det, trace)
    for all particles over the last 'step_window' steps.
    """
    if step_window is None:
        step_window = Lambda3SimConfig.HEATMAP_WINDOW

    keys = ["angle_det", "norm", "eig1", "eig2", "det", "trace"]
    fig, axes = plt.subplots(2, 3, figsize=(16, 8))
    for idx, key in enumerate(keys):
        data = np.array([np.real(t.lambda_hist[key][-step_window:]) for t in SemanticTensors])
        ax = axes[idx // 3, idx % 3]
        im = ax.imshow(data, aspect='auto', cmap='RdYlBu',
                       extent=[-step_window, 0, 0, len(SemanticTensors)])
        ax.set_title(key)
        ax.set_ylabel("Particle"); ax.set_xlabel("Step (from last)")
        plt.colorbar(im, ax=ax)
    plt.tight_layout(); plt.show()

def get_chaos_judgement_array(
    chaos_flags, angle_std_list, norm_jump_list, eig1_sign_change_list, det_jump_list, trace_std_list
):
    """
    Returns a DataFrame containing chaos judgement flags and key indicators
    (for diagnostics and event annotation).
    """
    N = len(chaos_flags)
    data = {
        "Step": list(range(N)),
        "ChaosFlag": chaos_flags,
        "angle_std": angle_std_list,
        "norm_jump": norm_jump_list,
        "eig1_sign_change": eig1_sign_change_list,
        "det_jump": det_jump_list,
        "trace_std": trace_std_list
    }
    df = pd.DataFrame(data)
    return df

def plot_event_network_with_energy(G_lambda3, positions_E, energies_E):
    """
    Visualize the Λ³ event network, coloring nodes by ΔΛC event energy.
    """
    node_list = list(G_lambda3.nodes)
    node_colors = np.array([energies_E[i] for i in node_list])
    pos_map = {i: (positions_E[i, 0], positions_E[i, 1]) for i in node_list}
    vmin, vmax = node_colors.min(), node_colors.max()

    plt.figure(figsize=(8, 6))
    nodes = nx.draw_networkx_nodes(G_lambda3, pos=pos_map, nodelist=node_list, node_size=15,
                                   node_color=node_colors, cmap="hot", vmin=vmin, vmax=vmax)
    nx.draw_networkx_edges(G_lambda3, pos=pos_map, alpha=0.3)
    plt.colorbar(nodes, label="Event Energy")
    plt.title("ΔΛC Event Network Topology (Chaos Branching)")
    plt.xlabel("x"); plt.ylabel("y")
    plt.tight_layout(); plt.show()

def plot_chaos_indicators_hist(SemanticTensors, window=None):
    """
    Plot histograms for local chaos indicators:
    - Standard deviation of angle_det
    - Maximum norm jump
    """
    if window is None:
        window = Lambda3SimConfig.HEATMAP_WINDOW

    angle_std_list = []
    norm_jump_list = []

    for t in SemanticTensors:
        if len(t.lambda_hist["angle_det"]) >= window:
            angle_std_list.append(np.std(np.array(t.lambda_hist["angle_det"][-window:])))
        else:
            angle_std_list.append(np.nan)

        if len(t.lambda_hist["norm"]) >= window:
            arr = np.array(t.lambda_hist["norm"][-window:])
            norm_jump_list.append(np.max(np.abs(np.diff(arr))))
        else:
            norm_jump_list.append(np.nan)

    plt.figure(figsize=(8, 3))
    plt.hist([v for v in angle_std_list if not np.isnan(v)], bins=30, alpha=0.7, label='angle_det std')
    plt.hist([v for v in norm_jump_list if not np.isnan(v)], bins=30, alpha=0.7, label='norm jump')
    plt.xlabel('Value'); plt.ylabel('Count')
    plt.legend(); plt.title("Distribution of Local Chaos Indicators")
    plt.tight_layout(); plt.show()

# --- Example usage and composite event/structure plots ---
plot_multivariate_heatmaps(SemanticTensors)
plot_event_network_with_energy(G_lambda3, positions_E, energies_E)
plot_chaos_indicators_hist(SemanticTensors, window=100)

# --- Visualization: Spatial Distribution of Largest Network Component ---
max_component = max(components_lambda3, key=len)
max_component_positions = np.array([positions_E[i] for i in max_component])
heatmap_max, _, _ = np.histogram2d(
    max_component_positions[:, 0], max_component_positions[:, 1],
    bins=(Nx, Ny), range=[[0, Lambda3SimConfig.L], [0, Lambda3SimConfig.L]]
)
plt.imshow(gaussian_filter(heatmap_max.T, sigma=2.0), origin='lower', cmap="inferno", extent=[0, Lambda3SimConfig.L, 0, Lambda3SimConfig.L])
plt.scatter([117.334], [160.753], c="cyan", marker="x", s=100, label="Max Source")
plt.title("Spatial Distribution of Largest Component (Size: 24962) with Max Source")
plt.colorbar(label="Event Density")
plt.legend()
plt.show()

# --- Save or Display Summary Table ---
df_chaos_root = pd.DataFrame(frames)
df_chaos_root.to_csv("chaos_root_table.csv", index=False)
print(df_chaos_root.head())

# --- Key Statistics and Correlation Outputs ---
print("Entropy–ΔΛC correlation (z-score):", np.corrcoef(entropy_z, DeltaLambdaC_z)[0,1])
print("Number of Λ³ network components:", len(components_lambda3))
print("Spatial correlation coefficient (ΔΛC vs. ∇Λ):", corr_val)
print("Index of the maximal wave source (central particle):", most_influential)
print("Its position:", positions_E[most_influential])
print("Total Λ³ events:", len(DeltaLambdaC_events), "(total number of structural transition events)")
print("Total Q_Λ jumps:", int(np.sum(Q_jump_counts)), "(number of topological invariant jump events)")
print("Number of event network components:", len(components_lambda3), "(number of independent structural clusters)")
print("Number of G_p network components:", len(components_p), "(number of independent clusters in particle network)")
print("Size of the largest component:", max(len(c) for c in components_lambda3), "(number of events in the largest cluster)")
print("Q_Λ fluctuation (max/mean):", f"{np.max(stds_lambda3):.3f}/{np.mean(stds_lambda3):.3f}", "(std. dev. of topological invariant per component)")
print("Q_Λ sum (initial/final):", f"{Q_Lambda_sum_history[0]:.3f}/{Q_Lambda_sum_history[-1]:.3f}", "(total topological invariant across all particles)")
print("ΔΛC events (max/mean per step):", f"{np.max(DeltaLambdaC_count_history):.3f}/{np.mean(DeltaLambdaC_count_history):.3f}", "(max/average number of structural transition events per step)")
print("Entropy–ΔΛC correlation:", f"{np.corrcoef(entropy_z, DeltaLambdaC_z)[0,1]:.3f}", "(correlation between structure entropy and ΔΛC events)")
print("KS statistic:", f"{stat:.3f}, p-value: {pval:.3f}", "(Kolmogorov–Smirnov test for initial vs. final velocity distribution)")

# ==============================
# 　Λ³ vs NSE: Comparative
# ==============================

# --- Simulation configuration class (for consistency with Lambda3SimConfig) ---
class NSEConfig:
    def __init__(self, l3cfg=None):
        if l3cfg is not None:
            self.Nx = getattr(l3cfg, 'GRID_NX', 60)
            self.Ny = getattr(l3cfg, 'GRID_NY', 60)
            self.Lx = getattr(l3cfg, 'L', 180)
            self.Ly = getattr(l3cfg, 'L', 180)
            self.dx = self.Lx / self.Nx
            self.dy = self.Ly / self.Ny
            self.dt = getattr(l3cfg, 'dt', 0.12)
            self.Nt = getattr(l3cfg, 'Nsteps', 150)
            self.nu = getattr(l3cfg, 'VISCOSITY', 0.01)
            self.rho = 1.0
            self.g = getattr(l3cfg, 'GRAVITY', np.array([0.0, -0.05]))[1]  # y成分だけ抜く
            # 回転強制なども同様に
            self.ROTATION_INJECTION_ON = getattr(l3cfg, 'EXTERNAL_INJECTION_OMEGA_ON', True)
            self.OMEGA_LEVEL = getattr(l3cfg, 'INJECTION_OMEGA_LEVEL', 0.05)
            self.OMEGA_STRENGTH = getattr(l3cfg, 'INJECTION_OMEGA_STRENGTH', 1.0)
            self.OMEGA_START = getattr(l3cfg, 'OMEGA_INJECTION_START', 0)
            self.OMEGA_END = getattr(l3cfg, 'OMEGA_INJECTION_END', 150)
        else:
            # 手動定義（旧スタイル互換）
            self.Nx = 60
            self.Ny = 60
            self.Lx = 180
            self.Ly = 180
            self.dx = self.Lx / self.Nx
            self.dy = self.Ly / self.Ny
            self.dt = 0.12
            self.Nt = 150
            self.nu = 0.01
            self.rho = 1.0
            self.g = -0.05
            self.ROTATION_INJECTION_ON = True
            self.OMEGA_LEVEL = 0.05
            self.OMEGA_STRENGTH = 1.0
            self.OMEGA_START = 0
            self.OMEGA_END = 150

# --- Helper functions ---
def laplacian(f, dx):
    return (
        np.roll(f, 1, axis=0) + np.roll(f, -1, axis=0) +
        np.roll(f, 1, axis=1) + np.roll(f, -1, axis=1) - 4 * f
    ) / dx**2

def divergence(u, v, dx, dy):
    return (np.roll(u, -1, axis=0) - np.roll(u, 1, axis=0)) / (2*dx) + \
           (np.roll(v, -1, axis=1) - np.roll(v, 1, axis=1)) / (2*dy)

def poisson_fft(rhs, dx):
    Nx, Ny = rhs.shape
    kx = np.fft.fftfreq(Nx).reshape(-1,1)
    ky = np.fft.fftfreq(Ny).reshape(1,-1)
    k2 = (kx**2 + ky**2)
    k2[0,0] = 1e-12  # avoid division by zero
    rhs_hat = np.fft.fft2(rhs)
    p_hat = rhs_hat / (-4 * np.pi**2 * k2)
    p = np.real(np.fft.ifft2(p_hat))
    p -= np.mean(p)
    return p

# --- Initialize fields ---
def initialize_fields(config):
    u = np.random.normal(0, 0.05, size=(config.Nx, config.Ny))  # x-velocity (with small noise)
    v = np.random.normal(0, 0.05, size=(config.Nx, config.Ny))  # y-velocity (with small noise)
    p = np.zeros((config.Nx, config.Ny))                        # Pressure
    return u, v, p

# --- Compute rotational forcing field ---
def rotational_forcing(config):
    X, Y = np.meshgrid(np.arange(config.Nx)*config.dx, np.arange(config.Ny)*config.dy, indexing='ij')
    xc, yc = config.Lx/2, config.Ly/2
    rx = X - xc
    ry = Y - yc
    u_force_rot = -config.OMEGA_LEVEL * config.OMEGA_STRENGTH * ry
    v_force_rot =  config.OMEGA_LEVEL * config.OMEGA_STRENGTH * rx
    return u_force_rot, v_force_rot

# --- Main simulation ---
def run_nse_simulation(config):
    u, v, p = initialize_fields(config)
    nse_results = []
    u_force_rot, v_force_rot = rotational_forcing(config)
    for n in range(config.Nt):
        # Advection terms
        u_adv = u * (np.roll(u, -1, axis=0) - np.roll(u, 1, axis=0)) / (2 * config.dx) + \
                v * (np.roll(u, -1, axis=1) - np.roll(u, 1, axis=1)) / (2 * config.dy)
        v_adv = u * (np.roll(v, -1, axis=0) - np.roll(v, 1, axis=0)) / (2 * config.dx) + \
                v * (np.roll(v, -1, axis=1) - np.roll(v, 1, axis=1)) / (2 * config.dy)
        # Diffusion (viscosity)
        u_diff = config.nu * laplacian(u, config.dx)
        v_diff = config.nu * laplacian(v, config.dx)
        # Gravity (acts on v only)
        v_force = config.g * config.dt

        # --- Rotational forcing ON/OFF & time window ---
        if config.ROTATION_INJECTION_ON and (config.OMEGA_START <= n < config.OMEGA_END):
            u_force = u_force_rot
            v_force_rot_t = v_force_rot
        else:
            u_force = 0.0
            v_force_rot_t = 0.0

        # Predictor step: update velocities without pressure
        u_star = u + config.dt * (-u_adv + u_diff + u_force)
        v_star = v + config.dt * (-v_adv + v_diff + v_force + v_force_rot_t)
        # Compute divergence of intermediate velocity
        div_star = divergence(u_star, v_star, config.dx, config.dy)
        # Pressure Poisson equation
        p = poisson_fft(div_star, config.dx)
        # Correct velocities to enforce incompressibility
        u = u_star - config.dt / (2*config.dx) * (np.roll(p, -1, axis=0) - np.roll(p, 1, axis=0))
        v = v_star - config.dt / (2*config.dy) * (np.roll(p, -1, axis=1) - np.roll(p, 1, axis=1))

        # Save results for analysis and plotting
        if n % 10 == 0 or n == config.Nt - 1:
            kinetic_energy = np.mean(u**2 + v**2)
            mean_pressure = np.mean(p)
            nse_results.append({
                "step": n,
                "kinetic_energy": kinetic_energy,
                "mean_pressure": mean_pressure,
                "u": u.copy(),
                "v": v.copy(),
                "p": p.copy()
            })
    return nse_results

# --- Run simulation ---
l3_cfg = Lambda3SimConfig
nse_cfg = NSEConfig(l3cfg=l3_cfg)
nse_results = run_nse_simulation(nse_cfg)

# --- Λ³ vs NSE: Comparative Plotting ---
ke_nse = np.array([r["kinetic_energy"] for r in nse_results])
ke_l3  = np.array([r.get("kinetic_energy_L3", np.nan) for r in l3_results])
mean_lf = np.array([r["mean_LambdaF"] for r in l3_results])

ke_nse_norm = ke_nse / np.nanmax(ke_nse)
ke_l3_norm  = ke_l3  / np.nanmax(ke_l3)
mean_lf_norm = mean_lf / np.nanmax(mean_lf)

plt.figure(figsize=(8,5))
plt.plot(
    [r["step"] for r in nse_results],
    ke_nse_norm,
    label="NSE Kinetic Energy (normalized)"
)
plt.plot(
    [r["step"] for r in l3_results],
    ke_l3_norm,
    label="Λ³ Kinetic Energy (normalized)"
)
plt.plot(
    [r["step"] for r in l3_results],
    mean_lf_norm,
    label="Λ³ Mean |Lambda_F| (normalized)",
    linestyle="dashed"
)
plt.xlabel("Step")
plt.ylabel("Normalized Energy / |ΛF|")
plt.title("Kinetic Energy (normalized): NSE vs Λ³ Simulation")
plt.legend()
plt.tight_layout()
plt.show()

plt.figure(figsize=(8,5))
plt.plot(
    [r["step"] for r in nse_results],
    [r["mean_pressure"] for r in nse_results],
    label="NSE Mean Pressure"
)
plt.plot(
    [r["step"] for r in l3_results],
    [r["mean_pressure_div"] for r in l3_results],
    label="Λ³ Mean Pressure (div)"
)
plt.plot(
    [r["step"] for r in l3_results],
    [r["mean_pressure_lap"] for r in l3_results],
    label="Λ³ Mean Pressure (lap)"
)
plt.xlabel("Step")
plt.ylabel("Mean Pressure")
plt.title("Mean Pressure: NSE vs Λ³ Simulation")
plt.legend()
plt.tight_layout()
plt.show()
