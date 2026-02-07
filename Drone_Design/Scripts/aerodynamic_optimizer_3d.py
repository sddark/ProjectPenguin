"""
COMPLETE 3D AERODYNAMIC OPTIMIZER
Genetic Algorithm + 3D Spectral Field Mesh + XLB CFD Integration

QUICK START:
    python aerodynamic_optimizer_3d.py

USAGE:
    # Basic 3D wing optimization
    python aerodynamic_optimizer_3d.py
    
    # Or import and use
    from aerodynamic_optimizer_3d import run_optimization
    best_genes, results = run_optimization()

REQUIREMENTS:
    pip install numpy scipy matplotlib

OPTIONAL (for production XLB):
    pip install xlb-jax jax jaxlib trimesh
"""

import numpy as np
from typing import Tuple, List, Dict, Optional
from dataclasses import dataclass
import json
from pathlib import Path


# ============================================================================
# DATA STRUCTURES
# ============================================================================

@dataclass
class AeroResults:
    """Storage for aerodynamic analysis results"""
    lift: float
    drag: float
    side_force: float
    roll_moment: float
    pitch_moment: float
    yaw_moment: float
    lift_to_drag: float
    fitness: float


# ============================================================================
# 3D SPECTRAL FIELD MESH GENERATION
# ============================================================================

class SpectralFieldMesh3D:
    """
    Generates 3D meshes using spectral field decomposition
    Represents 3D wing geometry using Fourier/Chebyshev series
    """
    
    def __init__(self, n_modes_chord: int = 10, n_modes_span: int = 8):
        """
        Args:
            n_modes_chord: Number of modes for chordwise direction
            n_modes_span: Number of modes for spanwise direction
        """
        self.n_modes_chord = n_modes_chord
        self.n_modes_span = n_modes_span
        
    def encode_geometry(self, coefficients: np.ndarray) -> Dict:
        """
        Encode 3D wing geometry from spectral coefficients
        
        Args:
            coefficients: Array of spectral coefficients
                - First section: upper surface (chord × span)
                - Second section: lower surface (chord × span)
                - Third section: thickness distribution (chord × span)
                - Fourth section: twist distribution (span)
                - Fifth section: taper distribution (span)
        
        Returns:
            3D mesh dictionary with vertices and connectivity
        """
        n_chord_points = 50
        n_span_points = 30
        
        # Parse coefficients
        n_surf = self.n_modes_chord * self.n_modes_span
        upper_coeffs = coefficients[:n_surf].reshape(self.n_modes_chord, self.n_modes_span)
        lower_coeffs = coefficients[n_surf:2*n_surf].reshape(self.n_modes_chord, self.n_modes_span)
        
        idx = 2 * n_surf
        thickness_coeffs = coefficients[idx:idx+n_surf].reshape(self.n_modes_chord, self.n_modes_span)
        
        idx += n_surf
        twist_coeffs = coefficients[idx:idx+self.n_modes_span]
        
        idx += self.n_modes_span
        taper_coeffs = coefficients[idx:idx+self.n_modes_span] if len(coefficients) > idx else np.zeros(self.n_modes_span)
        
        # Generate grid
        # Chordwise: 0 to 1
        u = (1 - np.cos(np.linspace(0, np.pi, n_chord_points))) / 2
        # Spanwise: -1 to 1 (symmetric wing)
        v = np.linspace(-1, 1, n_span_points)
        
        U, V = np.meshgrid(u, v, indexing='ij')
        
        # Compute spectral surfaces
        upper_surface = self._evaluate_spectral_surface(U, V, upper_coeffs)
        lower_surface = self._evaluate_spectral_surface(U, V, lower_coeffs)
        thickness = self._evaluate_spectral_surface(U, V, thickness_coeffs)
        
        # Add thickness
        upper_surface += thickness
        lower_surface -= thickness
        
        # Compute twist (rotation about quarter-chord)
        twist = self._evaluate_spectral_1d(V[0, :], twist_coeffs)
        
        # Compute taper (chord scaling along span)
        taper = 1.0 + self._evaluate_spectral_1d(V[0, :], taper_coeffs) * 0.5
        taper = np.clip(taper, 0.3, 1.5)  # Reasonable limits
        
        # Build 3D coordinates
        vertices_upper = []
        vertices_lower = []
        
        for j in range(n_span_points):
            # Local chord and twist
            local_chord = taper[j]
            local_twist = twist[j]
            
            for i in range(n_chord_points):
                x = U[i, j] * local_chord
                y = V[i, j]
                
                # Upper surface
                z_upper = upper_surface[i, j]
                # Apply twist (rotation about quarter-chord)
                x_rot, z_rot = self._rotate_2d(x - 0.25*local_chord, z_upper, local_twist)
                vertices_upper.append([x_rot + 0.25*local_chord, y, z_rot])
                
                # Lower surface
                z_lower = lower_surface[i, j]
                x_rot, z_rot = self._rotate_2d(x - 0.25*local_chord, z_lower, local_twist)
                vertices_lower.append([x_rot + 0.25*local_chord, y, z_rot])
        
        vertices_upper = np.array(vertices_upper)
        vertices_lower = np.array(vertices_lower)
        
        # Combine upper and lower surfaces
        vertices = np.vstack([vertices_upper, vertices_lower])
        
        # Create triangular mesh connectivity
        faces = self._create_surface_triangulation(n_chord_points, n_span_points)
        
        # Compute geometric properties
        bounds = self._compute_bounds(vertices)
        volume = self._estimate_volume(vertices, faces)
        ref_area = self._compute_reference_area(vertices, n_chord_points, n_span_points, taper)
        
        mesh_data = {
            'vertices': vertices,
            'faces': faces,
            'n_vertices': len(vertices),
            'n_faces': len(faces),
            'bounds': bounds,
            'volume': volume,
            'reference_area': ref_area,
            'wingspan': 2.0,  # Normalized wingspan
            'mean_chord': np.mean(taper)
        }
        
        return mesh_data
    
    def _evaluate_spectral_surface(self, U: np.ndarray, V: np.ndarray, coeffs: np.ndarray) -> np.ndarray:
        """Evaluate 2D spectral representation on grid"""
        result = np.zeros_like(U)
        
        for i in range(self.n_modes_chord):
            for j in range(self.n_modes_span):
                # Fourier basis functions
                chord_mode = np.sin((i+1) * np.pi * U)
                span_mode = np.cos((j+1) * np.pi * (V + 1) / 2)  # Shift to [0, 1]
                result += coeffs[i, j] * chord_mode * span_mode
        
        return result
    
    def _evaluate_spectral_1d(self, V: np.ndarray, coeffs: np.ndarray) -> np.ndarray:
        """Evaluate 1D spectral representation"""
        result = np.zeros_like(V)
        
        for j in range(len(coeffs)):
            result += coeffs[j] * np.cos((j+1) * np.pi * (V + 1) / 2)
        
        return result
    
    def _rotate_2d(self, x: float, z: float, angle: float) -> Tuple[float, float]:
        """Rotate point in xz-plane"""
        cos_a = np.cos(angle)
        sin_a = np.sin(angle)
        x_rot = x * cos_a - z * sin_a
        z_rot = x * sin_a + z * cos_a
        return x_rot, z_rot
    
    def _create_surface_triangulation(self, n_chord: int, n_span: int) -> np.ndarray:
        """Create triangular mesh faces for surface"""
        faces = []
        
        # Upper surface
        for j in range(n_span - 1):
            for i in range(n_chord - 1):
                idx0 = j * n_chord + i
                idx1 = j * n_chord + (i + 1)
                idx2 = (j + 1) * n_chord + (i + 1)
                idx3 = (j + 1) * n_chord + i
                
                faces.append([idx0, idx1, idx2])
                faces.append([idx0, idx2, idx3])
        
        # Lower surface (offset by n_chord * n_span)
        offset = n_chord * n_span
        for j in range(n_span - 1):
            for i in range(n_chord - 1):
                idx0 = offset + j * n_chord + i
                idx1 = offset + j * n_chord + (i + 1)
                idx2 = offset + (j + 1) * n_chord + (i + 1)
                idx3 = offset + (j + 1) * n_chord + i
                
                # Reversed winding for lower surface
                faces.append([idx0, idx2, idx1])
                faces.append([idx0, idx3, idx2])
        
        return np.array(faces)
    
    def _compute_bounds(self, vertices: np.ndarray) -> Tuple[float, ...]:
        """Compute bounding box"""
        mins = vertices.min(axis=0)
        maxs = vertices.max(axis=0)
        return tuple(mins) + tuple(maxs)
    
    def _estimate_volume(self, vertices: np.ndarray, faces: np.ndarray) -> float:
        """Estimate volume using divergence theorem"""
        volume = 0.0
        for face in faces:
            v0, v1, v2 = vertices[face]
            # Signed volume of tetrahedron with origin
            volume += np.dot(v0, np.cross(v1, v2)) / 6.0
        return abs(volume)
    
    def _compute_reference_area(self, vertices: np.ndarray, n_chord: int, n_span: int, taper: np.ndarray) -> float:
        """Compute planform reference area"""
        # Approximate as trapezoids along span
        span_length = 2.0
        mean_chord = np.mean(taper)
        return span_length * mean_chord
    
    def create_volume_mesh(self, boundary_mesh: Dict, resolution: Tuple[int, int, int]) -> Dict:
        """
        Create 3D volume mesh from 3D boundary
        
        Args:
            boundary_mesh: 3D boundary mesh dictionary
            resolution: Grid resolution (nx, ny, nz)
        
        Returns:
            Volume mesh dictionary for XLB
        """
        nx, ny, nz = resolution
        x_min, y_min, z_min, x_max, y_max, z_max = boundary_mesh['bounds']
        
        # Extend bounds for flow domain
        domain_scale = 3.0
        x_domain = (x_min - domain_scale, x_max + domain_scale)
        y_domain = (y_min - domain_scale, y_max + domain_scale)
        z_domain = (z_min - domain_scale, z_max + domain_scale)
        
        # Create grid
        x = np.linspace(x_domain[0], x_domain[1], nx)
        y = np.linspace(y_domain[0], y_domain[1], ny)
        z = np.linspace(z_domain[0], z_domain[1], nz)
        
        # Create solid mask using voxelization
        X, Y, Z = np.meshgrid(x, y, z, indexing='ij')
        solid_mask = self._create_solid_mask_3d(X, Y, Z, boundary_mesh)
        
        volume_mesh = {
            'grid_x': x,
            'grid_y': y,
            'grid_z': z,
            'solid_mask': solid_mask,
            'resolution': resolution,
            'boundary_vertices': boundary_mesh['vertices'],
            'boundary_faces': boundary_mesh['faces']
        }
        
        return volume_mesh
    
    def _create_solid_mask_3d(self, X: np.ndarray, Y: np.ndarray, Z: np.ndarray, 
                               boundary_mesh: Dict) -> np.ndarray:
        """Create 3D boolean mask for solid regions using ray casting"""
        vertices = boundary_mesh['vertices']
        faces = boundary_mesh['faces']
        
        shape = X.shape
        points = np.column_stack([X.ravel(), Y.ravel(), Z.ravel()])
        
        # Simplified voxelization (for production, use trimesh or similar)
        # This version uses distance-based approximation
        mask = np.zeros(len(points), dtype=bool)
        
        # Find points inside convex hull (simplified)
        # For each point, check if it's close to any triangle
        from scipy.spatial import Delaunay
        
        # Project to 2D for initial inside test (use xz plane at each y)
        for i in range(shape[1]):  # For each y slice
            y_val = Y[0, i, 0]
            slice_points_idx = np.where(np.abs(points[:, 1] - y_val) < (Y[0, 1, 0] - Y[0, 0, 0]))[0]
            
            if len(slice_points_idx) > 0:
                # Get wing vertices at this y position
                wing_verts_idx = np.where(np.abs(vertices[:, 1] - y_val) < 0.1)[0]
                
                if len(wing_verts_idx) > 3:
                    wing_verts_2d = vertices[wing_verts_idx][:, [0, 2]]
                    
                    try:
                        hull = Delaunay(wing_verts_2d)
                        slice_points_2d = points[slice_points_idx][:, [0, 2]]
                        inside = hull.find_simplex(slice_points_2d) >= 0
                        mask[slice_points_idx] = inside
                    except:
                        pass
        
        return mask.reshape(shape)


# ============================================================================
# XLB 3D CFD INTERFACE
# ============================================================================

class XLBInterface3D:
    """
    Interface to XLB (Lattice Boltzmann solver) for 3D flows
    
    MOCK VERSION: For demo/testing without XLB installed
    
    For production with real XLB, install: pip install xlb-jax jax jaxlib
    """
    
    def __init__(self, reynolds_number: float = 1000, mach_number: float = 0.1):
        self.reynolds = reynolds_number
        self.mach = mach_number
        
    def setup_simulation(self, mesh: Dict, flow_conditions: Dict) -> Dict:
        """
        Set up XLB 3D simulation
        
        Args:
            mesh: Volume mesh from SpectralFieldMesh3D
            flow_conditions: Dict with 'velocity', 'alpha' (AoA), 'beta' (sideslip)
        
        Returns:
            Simulation configuration
        """
        velocity = flow_conditions.get('velocity', 1.0)
        alpha = np.deg2rad(flow_conditions.get('alpha', 5.0))  # Angle of attack
        beta = np.deg2rad(flow_conditions.get('beta', 0.0))     # Sideslip angle
        
        # Compute velocity components in 3D
        # x: streamwise, y: spanwise, z: vertical
        v_x = velocity * np.cos(alpha) * np.cos(beta)
        v_y = velocity * np.sin(beta)
        v_z = velocity * np.sin(alpha) * np.cos(beta)
        
        config = {
            'grid_resolution': mesh['resolution'],
            'velocity_inlet': [v_x, v_y, v_z],
            'reynolds': self.reynolds,
            'mach': self.mach,
            'timesteps': 10000,
            'solid_mask': mesh['solid_mask'],
            'boundary_conditions': {
                'inlet': 'velocity',
                'outlet': 'pressure',
                'walls': 'no-slip',
                'far-field': 'free-stream'
            }
        }
        
        return config
    
    def run_simulation(self, config: Dict) -> Dict:
        """
        Run XLB 3D simulation
        
        MOCK VERSION - Uses simplified 3D aerodynamic model
        
        Returns:
            Raw simulation results with 3D forces and moments
        """
        # MOCK IMPLEMENTATION
        velocity = np.linalg.norm(config['velocity_inlet'])
        v_x, v_y, v_z = config['velocity_inlet']
        
        # Compute angles
        alpha = np.arctan2(v_z, v_x)
        beta = np.arcsin(v_y / (velocity + 1e-10))
        
        # 3D aerodynamic coefficients (simplified model)
        # Lift (primarily from alpha)
        cl = 2 * np.pi * alpha * (1 - 0.1 * alpha**2)
        
        # Drag (induced drag increases with lift, profile drag constant)
        cd_profile = 0.01
        cd_induced = 0.05 * cl**2 / (np.pi * 8.0)  # Aspect ratio ~ 8
        cd = cd_profile + cd_induced
        
        # Side force (from sideslip)
        cy = 2.0 * beta
        
        # Moments
        cm_pitch = -0.05 * cl  # Pitch moment (stability)
        cn_yaw = -0.02 * beta   # Yaw moment (directional stability)
        cl_roll = -0.01 * beta  # Roll moment (dihedral effect)
        
        results = {
            'forces': {
                'lift_coefficient': cl,
                'drag_coefficient': cd,
                'side_force_coefficient': cy,
                'roll_moment_coefficient': cl_roll,
                'pitch_moment_coefficient': cm_pitch,
                'yaw_moment_coefficient': cn_yaw
            },
            'flowfield': {
                'velocity': np.random.randn(10, 10, 10, 3) * 0.1 + velocity,
                'pressure': np.random.randn(10, 10, 10),
                'vorticity': np.random.randn(10, 10, 10, 3)
            },
            'convergence': {
                'residual': 1e-6,
                'iterations': 5000
            }
        }
        
        return results
    
    def compute_aerodynamic_forces(self, results: Dict, reference_area: float = 1.0,
                                   wingspan: float = 2.0, mean_chord: float = 1.0) -> AeroResults:
        """
        Extract 3D aerodynamic coefficients and compute fitness
        
        Args:
            results: Raw simulation results from XLB
            reference_area: Reference wing area
            wingspan: Wing span
            mean_chord: Mean aerodynamic chord
        
        Returns:
            AeroResults with 3D forces and moments
        """
        cl = results['forces']['lift_coefficient']
        cd = results['forces']['drag_coefficient']
        cy = results['forces']['side_force_coefficient']
        cl_roll = results['forces']['roll_moment_coefficient']
        cm_pitch = results['forces']['pitch_moment_coefficient']
        cn_yaw = results['forces']['yaw_moment_coefficient']
        
        # Compute dimensional forces
        q_inf = 0.5 * 1.225 * 1.0**2  # Dynamic pressure
        lift = cl * q_inf * reference_area
        drag = cd * q_inf * reference_area
        side_force = cy * q_inf * reference_area
        
        # Compute moments (dimensionalized)
        roll_moment = cl_roll * q_inf * reference_area * wingspan
        pitch_moment = cm_pitch * q_inf * reference_area * mean_chord
        yaw_moment = cn_yaw * q_inf * reference_area * wingspan
        
        # Performance metrics
        ld_ratio = cl / cd if cd > 0 else 0.0
        
        # Multi-objective fitness for 3D wing
        # Maximize L/D, ensure stability, minimize drag, penalize side force
        fitness = (
            ld_ratio * 15.0                    # High L/D is good
            + (-cm_pitch) * 10.0                # Negative pitch moment = stable
            + (-cn_yaw) * 8.0                   # Yaw stability
            - cd * 25.0                         # Low drag
            - abs(cy) * 5.0                     # Minimize side force
            - abs(cl_roll) * 5.0                # Minimize roll moment
        )
        
        return AeroResults(
            lift=lift,
            drag=drag,
            side_force=side_force,
            roll_moment=roll_moment,
            pitch_moment=pitch_moment,
            yaw_moment=yaw_moment,
            lift_to_drag=ld_ratio,
            fitness=fitness
        )


# ============================================================================
# GENETIC ALGORITHM OPTIMIZER FOR 3D WINGS
# ============================================================================

class GeneticAlgorithm3D:
    """
    Genetic Algorithm for 3D wing optimization
    """
    
    def __init__(
        self,
        n_modes_chord: int = 10,
        n_modes_span: int = 8,
        population_size: int = 50,
        n_generations: int = 100,
        mutation_rate: float = 0.1,
        crossover_rate: float = 0.8,
        elite_size: int = 5
    ):
        # Calculate total genes
        n_surf = n_modes_chord * n_modes_span
        self.n_genes = 3 * n_surf + 2 * n_modes_span  # upper, lower, thickness + twist + taper
        
        self.n_modes_chord = n_modes_chord
        self.n_modes_span = n_modes_span
        self.population_size = population_size
        self.n_generations = n_generations
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        self.elite_size = elite_size
        
        # Components
        self.mesh_generator = SpectralFieldMesh3D(n_modes_chord, n_modes_span)
        self.cfd_solver = XLBInterface3D()
        
        # Tracking
        self.history = []
        
    def initialize_population(self) -> np.ndarray:
        """Create initial population of 3D wing spectral coefficients"""
        population = np.random.randn(self.population_size, self.n_genes) * 0.05
        
        # Seed with reasonable wing shape
        n_surf = self.n_modes_chord * self.n_modes_span
        
        # Upper surface (positive camber)
        population[0, :n_surf] = np.random.randn(n_surf) * 0.05 + 0.1
        population[0, :5] = [0.15, -0.08, 0.04, -0.02, 0.01]
        
        # Lower surface (negative camber)
        population[0, n_surf:2*n_surf] = np.random.randn(n_surf) * 0.05 - 0.1
        population[0, n_surf:n_surf+5] = [-0.12, 0.06, -0.03, 0.015, -0.008]
        
        # Thickness
        population[0, 2*n_surf:3*n_surf] = np.random.randn(n_surf) * 0.02
        
        # Twist (washout - less at tips)
        population[0, 3*n_surf:3*n_surf+self.n_modes_span] = [-0.05, 0.02, -0.01, 0.0, 0.0, 0.0, 0.0, 0.0][:self.n_modes_span]
        
        # Taper (narrower at tips)
        population[0, 3*n_surf+self.n_modes_span:] = [-0.2, 0.1, -0.05, 0.0, 0.0, 0.0, 0.0, 0.0][:self.n_modes_span]
        
        return population
    
    def evaluate_individual(self, genes: np.ndarray, flow_conditions: Dict) -> AeroResults:
        """Evaluate fitness of one 3D wing design"""
        # Generate 3D mesh from spectral coefficients
        boundary_mesh = self.mesh_generator.encode_geometry(genes)
        
        # Create volume mesh for CFD
        volume_mesh = self.mesh_generator.create_volume_mesh(
            boundary_mesh,
            resolution=(80, 80, 60)  # 3D resolution
        )
        
        # Setup and run XLB simulation
        sim_config = self.cfd_solver.setup_simulation(volume_mesh, flow_conditions)
        results = self.cfd_solver.run_simulation(sim_config)
        
        # Extract aerodynamic forces
        aero_results = self.cfd_solver.compute_aerodynamic_forces(
            results,
            reference_area=boundary_mesh['reference_area'],
            wingspan=boundary_mesh['wingspan'],
            mean_chord=boundary_mesh['mean_chord']
        )
        
        return aero_results
    
    def evaluate_population(self, population: np.ndarray, flow_conditions: Dict) -> List[AeroResults]:
        """Evaluate entire population"""
        results = []
        for i, individual in enumerate(population):
            try:
                result = self.evaluate_individual(individual, flow_conditions)
                results.append(result)
                print(f"Individual {i+1}/{len(population)}: L/D={result.lift_to_drag:.2f}, Fitness={result.fitness:.2f}")
            except Exception as e:
                print(f"Individual {i+1} failed: {e}")
                results.append(AeroResults(0, 1000, 0, 0, 0, 0, 0, -1000))
        
        return results
    
    def selection(self, population: np.ndarray, fitness: np.ndarray) -> np.ndarray:
        """Tournament selection"""
        tournament_size = 3
        selected = []
        
        for _ in range(len(population)):
            indices = np.random.choice(len(population), tournament_size, replace=False)
            tournament_fitness = fitness[indices]
            winner_idx = indices[np.argmax(tournament_fitness)]
            selected.append(population[winner_idx])
        
        return np.array(selected)
    
    def crossover(self, parent1: np.ndarray, parent2: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Blend crossover (BLX-α)"""
        if np.random.random() > self.crossover_rate:
            return parent1.copy(), parent2.copy()
        
        alpha = 0.5
        min_genes = np.minimum(parent1, parent2)
        max_genes = np.maximum(parent1, parent2)
        range_genes = max_genes - min_genes
        
        child1 = min_genes - alpha * range_genes + np.random.random(self.n_genes) * (1 + 2*alpha) * range_genes
        child2 = min_genes - alpha * range_genes + np.random.random(self.n_genes) * (1 + 2*alpha) * range_genes
        
        return child1, child2
    
    def mutate(self, individual: np.ndarray) -> np.ndarray:
        """Gaussian mutation"""
        mutated = individual.copy()
        mutation_mask = np.random.random(self.n_genes) < self.mutation_rate
        mutated[mutation_mask] += np.random.randn(np.sum(mutation_mask)) * 0.05
        mutated = np.clip(mutated, -1.0, 1.0)
        return mutated
    
    def evolve(self, flow_conditions: Dict = None) -> Tuple[np.ndarray, AeroResults]:
        """Run genetic algorithm optimization"""
        if flow_conditions is None:
            flow_conditions = {
                'velocity': 50.0,
                'alpha': 5.0,  # Angle of attack
                'beta': 0.0    # Sideslip
            }
        
        print(f"Starting 3D Wing Genetic Algorithm Optimization")
        print(f"Population: {self.population_size}, Generations: {self.n_generations}")
        print(f"Flow Conditions: {flow_conditions}")
        print(f"Total Genes: {self.n_genes} ({self.n_modes_chord}×{self.n_modes_span} modes)")
        print("="*60)
        
        # Initialize
        population = self.initialize_population()
        
        best_ever_fitness = -np.inf
        best_ever_individual = None
        best_ever_results = None
        
        for generation in range(self.n_generations):
            print(f"\nGeneration {generation + 1}/{self.n_generations}")
            
            # Evaluate
            results = self.evaluate_population(population, flow_conditions)
            fitness = np.array([r.fitness for r in results])
            
            # Track best
            best_idx = np.argmax(fitness)
            best_fitness = fitness[best_idx]
            
            if best_fitness > best_ever_fitness:
                best_ever_fitness = best_fitness
                best_ever_individual = population[best_idx].copy()
                best_ever_results = results[best_idx]
            
            # Statistics
            print(f"Best Fitness: {best_fitness:.2f}")
            print(f"Mean Fitness: {np.mean(fitness):.2f}")
            print(f"Best L/D: {results[best_idx].lift_to_drag:.2f}")
            
            self.history.append({
                'generation': generation,
                'best_fitness': best_fitness,
                'mean_fitness': np.mean(fitness),
                'std_fitness': np.std(fitness),
                'best_ld': results[best_idx].lift_to_drag
            })
            
            # Elitism
            elite_indices = np.argsort(fitness)[-self.elite_size:]
            elites = population[elite_indices]
            
            # Selection
            parents = self.selection(population, fitness)
            
            # Create next generation
            offspring = []
            for i in range(0, len(parents) - 1, 2):
                child1, child2 = self.crossover(parents[i], parents[i+1])
                offspring.append(self.mutate(child1))
                offspring.append(self.mutate(child2))
            
            # Combine
            population = np.vstack([elites, offspring[:self.population_size - self.elite_size]])
        
        print("\n" + "="*60)
        print("3D Wing Optimization Complete!")
        print(f"Best Fitness: {best_ever_fitness:.2f}")
        print(f"Best L/D: {best_ever_results.lift_to_drag:.2f}")
        print(f"Best Drag: {best_ever_results.drag:.4f}")
        
        return best_ever_individual, best_ever_results
    
    def save_results(self, filename: str, best_individual: np.ndarray, results: AeroResults):
        """Save optimization results"""
        data = {
            'best_genes': best_individual.tolist(),
            'results': {
                'lift': results.lift,
                'drag': results.drag,
                'side_force': results.side_force,
                'roll_moment': results.roll_moment,
                'pitch_moment': results.pitch_moment,
                'yaw_moment': results.yaw_moment,
                'lift_to_drag': results.lift_to_drag,
                'fitness': results.fitness
            },
            'history': self.history,
            'config': {
                'n_genes': self.n_genes,
                'n_modes_chord': self.n_modes_chord,
                'n_modes_span': self.n_modes_span,
                'population_size': self.population_size,
                'n_generations': self.n_generations,
                'mutation_rate': self.mutation_rate,
                'crossover_rate': self.crossover_rate
            }
        }
        
        with open(filename, 'w') as f:
            json.dump(data, f, indent=2)
        
        print(f"\nResults saved to {filename}")


# ============================================================================
# CONVENIENCE FUNCTIONS
# ============================================================================

def run_optimization(
    n_modes_chord: int = 8,
    n_modes_span: int = 6,
    population_size: int = 30,
    n_generations: int = 50,
    velocity: float = 50.0,
    alpha: float = 5.0,
    beta: float = 0.0,
    output_file: str = 'wing_optimization_results.json'
) -> Tuple[np.ndarray, AeroResults]:
    """
    Convenience function to run 3D wing optimization
    
    Args:
        n_modes_chord: Chordwise spectral modes (default: 8)
        n_modes_span: Spanwise spectral modes (default: 6)
        population_size: Population size (default: 30)
        n_generations: Number of generations (default: 50)
        velocity: Flight velocity in m/s (default: 50.0)
        alpha: Angle of attack in degrees (default: 5.0)
        beta: Sideslip angle in degrees (default: 0.0)
        output_file: Output filename
    
    Returns:
        Tuple of (best_genes, results)
    """
    # Create optimizer
    ga = GeneticAlgorithm3D(
        n_modes_chord=n_modes_chord,
        n_modes_span=n_modes_span,
        population_size=population_size,
        n_generations=n_generations,
        mutation_rate=0.15,
        crossover_rate=0.8,
        elite_size=5
    )
    
    # Define flow conditions
    flow_conditions = {
        'velocity': velocity,
        'alpha': alpha,
        'beta': beta
    }
    
    # Run optimization
    best_genes, results = ga.evolve(flow_conditions)
    
    # Save results
    ga.save_results(output_file, best_genes, results)
    
    # Save 3D mesh
    mesh = ga.mesh_generator.encode_geometry(best_genes)
    
    # Save as OBJ file for 3D visualization
    obj_file = output_file.replace('.json', '_wing.obj')
    save_obj_mesh(mesh['vertices'], mesh['faces'], obj_file)
    
    return best_genes, results


def save_obj_mesh(vertices: np.ndarray, faces: np.ndarray, filename: str):
    """Save mesh as OBJ file"""
    with open(filename, 'w') as f:
        f.write("# 3D Wing Mesh\n")
        
        # Write vertices
        for v in vertices:
            f.write(f"v {v[0]:.6f} {v[1]:.6f} {v[2]:.6f}\n")
        
        # Write faces (OBJ uses 1-indexed)
        for face in faces:
            f.write(f"f {face[0]+1} {face[1]+1} {face[2]+1}\n")
    
    print(f"3D mesh saved to {filename}")


def quick_test_3d():
    """Quick test with small population"""
    print("\n" + "="*70)
    print("QUICK 3D WING TEST - Small population for rapid testing")
    print("="*70)
    
    return run_optimization(
        n_modes_chord=6,
        n_modes_span=4,
        population_size=8,
        n_generations=5,
        output_file='quick_test_3d_results.json'
    )


def visualize_wing_3d(genes: np.ndarray, n_modes_chord: int = 8, n_modes_span: int = 6):
    """
    Visualize 3D wing using matplotlib
    
    Args:
        genes: Spectral coefficients
        n_modes_chord: Chordwise modes
        n_modes_span: Spanwise modes
    """
    try:
        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d import Axes3D
        
        mesh_gen = SpectralFieldMesh3D(n_modes_chord, n_modes_span)
        mesh = mesh_gen.encode_geometry(genes)
        vertices = mesh['vertices']
        faces = mesh['faces']
        
        fig = plt.figure(figsize=(15, 10))
        
        # 3D view
        ax1 = fig.add_subplot(221, projection='3d')
        ax1.plot_trisurf(vertices[:, 0], vertices[:, 1], vertices[:, 2], 
                         triangles=faces, alpha=0.7, edgecolor='k', linewidth=0.2)
        ax1.set_xlabel('X (chord)')
        ax1.set_ylabel('Y (span)')
        ax1.set_zlabel('Z (thickness)')
        ax1.set_title('3D Wing Shape - Isometric View')
        
        # Top view
        ax2 = fig.add_subplot(222)
        ax2.triplot(vertices[:, 0], vertices[:, 1], faces, 'b-', linewidth=0.5)
        ax2.set_aspect('equal')
        ax2.set_xlabel('X (chord)')
        ax2.set_ylabel('Y (span)')
        ax2.set_title('Top View (Planform)')
        ax2.grid(True, alpha=0.3)
        
        # Front view
        ax3 = fig.add_subplot(223)
        ax3.triplot(vertices[:, 1], vertices[:, 2], faces, 'b-', linewidth=0.5)
        ax3.set_aspect('equal')
        ax3.set_xlabel('Y (span)')
        ax3.set_ylabel('Z (thickness)')
        ax3.set_title('Front View')
        ax3.grid(True, alpha=0.3)
        
        # Side view
        ax4 = fig.add_subplot(224)
        ax4.triplot(vertices[:, 0], vertices[:, 2], faces, 'b-', linewidth=0.5)
        ax4.set_aspect('equal')
        ax4.set_xlabel('X (chord)')
        ax4.set_ylabel('Z (thickness)')
        ax4.set_title('Side View (Airfoil)')
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('wing_3d_visualization.png', dpi=150)
        print("3D wing visualization saved to wing_3d_visualization.png")
        plt.show()
        
    except ImportError:
        print("matplotlib not installed. Install with: pip install matplotlib")


# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    """Main function for 3D wing optimization"""
    print("\n" + "="*70)
    print("3D AERODYNAMIC WING OPTIMIZER")
    print("Genetic Algorithm + 3D Spectral Fields + XLB CFD")
    print("="*70)
    
    # Run optimization
    best_genes, results = run_optimization(
        n_modes_chord=6,
        n_modes_span=5,
        population_size=15,
        n_generations=8,
        velocity=50.0,
        alpha=5.0,
        beta=0.0,
        output_file='wing_optimization_results.json'
    )
    
    print("\n" + "="*70)
    print("FINAL 3D WING RESULTS")
    print("="*70)
    print(f"Lift Coefficient: {results.lift:.4f}")
    print(f"Drag Coefficient: {results.drag:.4f}")
    print(f"Side Force: {results.side_force:.4f}")
    print(f"L/D Ratio: {results.lift_to_drag:.2f}")
    print(f"Pitch Moment (Cm): {results.pitch_moment:.4f}")
    print(f"Roll Moment (Cl): {results.roll_moment:.4f}")
    print(f"Yaw Moment (Cn): {results.yaw_moment:.4f}")
    print(f"Fitness Score: {results.fitness:.2f}")
    print("="*70)
    
    # Try to visualize
    try:
        visualize_wing_3d(best_genes, n_modes_chord=6, n_modes_span=5)
    except:
        pass
    
    return best_genes, results


if __name__ == "__main__":
    # Run main optimization
    best_genes, results = main()
    
    print("\n" + "="*70)
    print("USAGE EXAMPLES:")
    print("="*70)
    print("\n1. Quick test (5 generations):")
    print("   from aerodynamic_optimizer_3d import quick_test_3d")
    print("   quick_test_3d()")
    print("\n2. Custom optimization:")
    print("   from aerodynamic_optimizer_3d import run_optimization")
    print("   best, results = run_optimization(n_generations=100, alpha=8.0)")
    print("\n3. Visualize 3D wing:")
    print("   from aerodynamic_optimizer_3d import visualize_wing_3d")
    print("   visualize_wing_3d(best_genes, n_modes_chord=6, n_modes_span=5)")
    print("="*70)
