"""
COMPLETE AERODYNAMIC OPTIMIZER
Genetic Algorithm + Spectral Field Mesh + XLB CFD Integration

QUICK START:
    python aerodynamic_optimizer_complete.py

USAGE:
    # Basic optimization
    python aerodynamic_optimizer_complete.py
    
    # Or import and use
    from aerodynamic_optimizer_complete import run_optimization
    best_genes, results = run_optimization()

REQUIREMENTS:
    pip install numpy scipy matplotlib

OPTIONAL (for production XLB):
    pip install xlb-jax jax jaxlib
"""

import numpy as np
from typing import Tuple, List, Dict, Optional
from dataclasses import dataclass
import json
from pathlib import Path
from matplotlib.path import Path


# ============================================================================
# DATA STRUCTURES
# ============================================================================

@dataclass
class AeroResults:
    """Storage for aerodynamic analysis results"""
    lift: float
    drag: float
    stability_moment: float
    lift_to_drag: float
    fitness: float


# ============================================================================
# SPECTRAL FIELD MESH GENERATION
# ============================================================================

class SpectralFieldMesh:
    """
    Generates meshes using spectral field decomposition
    Represents geometry using Fourier/Chebyshev series
    """
    
    def __init__(self, n_modes: int = 10):
        self.n_modes = n_modes
        
    def encode_geometry(self, coefficients: np.ndarray) -> Dict:
        """
        Encode geometry from spectral coefficients
        
        Args:
            coefficients: Array of spectral coefficients [n_modes * 2] for 2D
                         or [n_modes * 3] for 3D
        
        Returns:
            Mesh dictionary with vertices and connectivity
        """
        # Create airfoil using spectral decomposition
        n_points = 200
        theta = np.linspace(0, 2*np.pi, n_points)
        
        # Upper surface using Fourier series
        upper_coeffs = coefficients[:self.n_modes]
        lower_coeffs = coefficients[self.n_modes:2*self.n_modes]
        thickness_coeffs = coefficients[2*self.n_modes:] if len(coefficients) > 2*self.n_modes else np.zeros(self.n_modes)
        
        # Generate profile
        x = (1 - np.cos(theta[:n_points//2])) / 2  # Cosine spacing
        
        # Upper surface
        y_upper = np.zeros_like(x)
        for i, coeff in enumerate(upper_coeffs):
            y_upper += coeff * np.sin((i+1) * np.pi * x)
        
        # Lower surface
        y_lower = np.zeros_like(x)
        for i, coeff in enumerate(lower_coeffs):
            y_lower += coeff * np.sin((i+1) * np.pi * x)
        
        # Add thickness distribution
        thickness = np.zeros_like(x)
        for i, coeff in enumerate(thickness_coeffs):
            thickness += coeff * np.sin((i+1) * np.pi * x)
        
        y_upper += thickness
        y_lower -= thickness
        
        # Combine into closed contour
        vertices = np.vstack([
            np.column_stack([x, y_upper]),
            np.column_stack([x[::-1], y_lower[::-1]])
        ])
        
        # Create mesh grid for CFD
        mesh_data = {
            'vertices': vertices,
            'n_vertices': len(vertices),
            'bounds': self._compute_bounds(vertices),
            'area': self._compute_area(vertices)
        }
        
        return mesh_data
    
    def _compute_bounds(self, vertices: np.ndarray) -> Tuple[float, float, float, float]:
        """Compute bounding box"""
        x_min, y_min = vertices.min(axis=0)
        x_max, y_max = vertices.max(axis=0)
        return x_min, y_min, x_max, y_max
    
    def _compute_area(self, vertices: np.ndarray) -> float:
        """Compute cross-sectional area using shoelace formula"""
        x = vertices[:, 0]
        y = vertices[:, 1]
        return 0.5 * np.abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))
    
    def create_volume_mesh(self, boundary_mesh: Dict, resolution: Tuple[int, int, int]) -> Dict:
        """
        Create 3D volume mesh from 2D boundary
        
        Args:
            boundary_mesh: 2D boundary mesh dictionary
            resolution: Grid resolution (nx, ny, nz)
        
        Returns:
            Volume mesh dictionary for XLB
        """
        nx, ny, nz = resolution
        x_min, y_min, x_max, y_max = boundary_mesh['bounds']
        
        # Extend bounds for flow domain
        domain_scale = 5.0
        x_domain = (x_min - domain_scale, x_max + domain_scale)
        y_domain = (y_min - domain_scale, y_max + domain_scale)
        z_domain = (-0.5, 0.5)  # Thin domain for 2D-like simulation
        
        # Create grid
        x = np.linspace(x_domain[0], x_domain[1], nx)
        y = np.linspace(y_domain[0], y_domain[1], ny)
        z = np.linspace(z_domain[0], z_domain[1], nz)
        
        # Create solid mask (simplified - in practice use ray casting)
        X, Y, Z = np.meshgrid(x, y, z, indexing='ij')
        solid_mask = self._create_solid_mask(X, Y, boundary_mesh['vertices'])
        
        volume_mesh = {
            'grid_x': x,
            'grid_y': y,
            'grid_z': z,
            'solid_mask': solid_mask,
            'resolution': resolution,
            'boundary_vertices': boundary_mesh['vertices']
        }
        
        return volume_mesh
    
    def _create_solid_mask(self, X: np.ndarray, Y: np.ndarray, vertices: np.ndarray) -> np.ndarray:
        """Create boolean mask for solid regions using point-in-polygon test"""
        from matplotlib.path import Path
        
        shape = X.shape
        # Create 2D mask first
        points_2d = np.column_stack([X[:,:,0].ravel(), Y[:,:,0].ravel()])
        path = Path(vertices)
        mask_2d = path.contains_points(points_2d)
        mask_2d = mask_2d.reshape(shape[0], shape[1])
        
        # Extend to 3D by repeating along z-axis
        mask_3d = np.repeat(mask_2d[:, :, np.newaxis], shape[2], axis=2)
        return mask_3d


# ============================================================================
# XLB CFD INTERFACE
# ============================================================================

class XLBInterface:
    """
    Interface to XLB (Lattice Boltzmann solver)
    
    MOCK VERSION: For demo/testing without XLB installed
    
    For production with real XLB, install: pip install xlb-jax jax jaxlib
    Then uncomment the ProductionXLBInterface at the bottom of this file
    """
    
    def __init__(self, reynolds_number: float = 1000, mach_number: float = 0.1):
        self.reynolds = reynolds_number
        self.mach = mach_number
        
    def setup_simulation(self, mesh: Dict, flow_conditions: Dict) -> Dict:
        """
        Set up XLB simulation
        
        Args:
            mesh: Volume mesh from SpectralFieldMesh
            flow_conditions: Dict with 'velocity', 'angle_of_attack', etc.
        
        Returns:
            Simulation configuration
        """
        angle = flow_conditions.get('angle_of_attack', 0.0)
        velocity = flow_conditions.get('velocity', 1.0)
        
        # Convert angle to radians
        alpha = np.deg2rad(angle)
        
        config = {
            'grid_resolution': mesh['resolution'],
            'velocity_inlet': [velocity * np.cos(alpha), velocity * np.sin(alpha), 0.0],
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
        Run XLB simulation
        
        MOCK VERSION - Replace with actual XLB API calls in production
        
        Returns:
            Raw simulation results
        """
        # MOCK IMPLEMENTATION - Replace with actual XLB API calls
        # This simulates the XLB computation
        
        velocity = np.linalg.norm(config['velocity_inlet'])
        
        # Simplified aerodynamic model for demonstration
        # In reality, XLB would solve Navier-Stokes
        alpha = np.arctan2(config['velocity_inlet'][1], config['velocity_inlet'][0])
        
        # Mock force coefficients (would come from XLB)
        cl = 2 * np.pi * alpha * (1 - 0.1 * alpha**2)  # Thin airfoil theory approximation
        cd = 0.01 + 0.05 * cl**2  # Simple drag polar
        cm = -0.05 * cl  # Moment coefficient
        
        results = {
            'forces': {
                'lift_coefficient': cl,
                'drag_coefficient': cd,
                'moment_coefficient': cm
            },
            'flowfield': {
                'velocity': np.random.randn(10, 10, 3) * 0.1 + velocity,
                'pressure': np.random.randn(10, 10),
                'vorticity': np.random.randn(10, 10)
            },
            'convergence': {
                'residual': 1e-6,
                'iterations': 5000
            }
        }
        
        return results
    
    def compute_aerodynamic_forces(self, results: Dict, reference_area: float = 1.0) -> AeroResults:
        """
        Extract aerodynamic coefficients and compute fitness
        
        Args:
            results: Raw simulation results from XLB
            reference_area: Reference area for force coefficients
        
        Returns:
            AeroResults with lift, drag, stability metrics
        """
        cl = results['forces']['lift_coefficient']
        cd = results['forces']['drag_coefficient']
        cm = results['forces']['moment_coefficient']
        
        # Compute dimensional forces (using reference area)
        q_inf = 0.5 * 1.225 * 1.0**2  # Dynamic pressure (rho * V^2 / 2)
        lift = cl * q_inf * reference_area
        drag = cd * q_inf * reference_area
        
        # Lift-to-drag ratio
        ld_ratio = cl / cd if cd > 0 else 0.0
        
        # Stability metric (want negative moment for stability)
        stability = -cm  # Positive stability value is good
        
        # Compute fitness (multi-objective)
        # Maximize L/D and stability, minimize drag
        fitness = ld_ratio * 10.0 + stability * 5.0 - cd * 20.0
        
        return AeroResults(
            lift=lift,
            drag=drag,
            stability_moment=cm,
            lift_to_drag=ld_ratio,
            fitness=fitness
        )


# ============================================================================
# GENETIC ALGORITHM OPTIMIZER
# ============================================================================

class GeneticAlgorithm:
    """
    Genetic Algorithm for aerodynamic optimization
    """
    
    def __init__(
        self,
        n_genes: int = 30,
        population_size: int = 50,
        n_generations: int = 100,
        mutation_rate: float = 0.1,
        crossover_rate: float = 0.8,
        elite_size: int = 5
    ):
        self.n_genes = n_genes
        self.population_size = population_size
        self.n_generations = n_generations
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        self.elite_size = elite_size
        
        # Components
        self.mesh_generator = SpectralFieldMesh(n_modes=n_genes // 3)
        self.cfd_solver = XLBInterface()
        
        # Tracking
        self.history = []
        
    def initialize_population(self) -> np.ndarray:
        """
        Create initial population of spectral coefficients
        
        Returns:
            Population array of shape (population_size, n_genes)
        """
        # Initialize with small random values near conventional airfoils
        population = np.random.randn(self.population_size, self.n_genes) * 0.1
        
        # Seed with known good shape (NACA-like)
        population[0, :self.n_genes//3] = [0.2, -0.1, 0.05, -0.02, 0.01] + [0.0] * (self.n_genes//3 - 5)
        population[0, self.n_genes//3:2*self.n_genes//3] = [-0.15, 0.08, -0.04, 0.02, -0.01] + [0.0] * (self.n_genes//3 - 5)
        
        return population
    
    def evaluate_individual(self, genes: np.ndarray, flow_conditions: Dict) -> AeroResults:
        """
        Evaluate fitness of one individual
        
        Args:
            genes: Spectral coefficients
            flow_conditions: Flow conditions for simulation
        
        Returns:
            AeroResults with fitness and performance metrics
        """
        # Generate mesh from spectral coefficients
        boundary_mesh = self.mesh_generator.encode_geometry(genes)
        
        # Create volume mesh for CFD
        volume_mesh = self.mesh_generator.create_volume_mesh(
            boundary_mesh,
            resolution=(100, 80, 5)  # Adjustable resolution
        )
        
        # Setup and run XLB simulation
        sim_config = self.cfd_solver.setup_simulation(volume_mesh, flow_conditions)
        results = self.cfd_solver.run_simulation(sim_config)
        
        # Extract aerodynamic forces
        aero_results = self.cfd_solver.compute_aerodynamic_forces(
            results,
            reference_area=boundary_mesh['area']
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
                # Assign poor fitness to invalid geometries
                results.append(AeroResults(0, 1000, 0, 0, -1000))
        
        return results
    
    def selection(self, population: np.ndarray, fitness: np.ndarray) -> np.ndarray:
        """
        Tournament selection
        
        Args:
            population: Current population
            fitness: Fitness values for each individual
        
        Returns:
            Selected parents
        """
        tournament_size = 3
        selected = []
        
        for _ in range(len(population)):
            # Random tournament
            indices = np.random.choice(len(population), tournament_size, replace=False)
            tournament_fitness = fitness[indices]
            winner_idx = indices[np.argmax(tournament_fitness)]
            selected.append(population[winner_idx])
        
        return np.array(selected)
    
    def crossover(self, parent1: np.ndarray, parent2: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Blend crossover (BLX-α)
        
        Args:
            parent1, parent2: Parent genes
        
        Returns:
            Two offspring
        """
        if np.random.random() > self.crossover_rate:
            return parent1.copy(), parent2.copy()
        
        alpha = 0.5
        
        # Blend crossover
        min_genes = np.minimum(parent1, parent2)
        max_genes = np.maximum(parent1, parent2)
        range_genes = max_genes - min_genes
        
        child1 = min_genes - alpha * range_genes + np.random.random(self.n_genes) * (1 + 2*alpha) * range_genes
        child2 = min_genes - alpha * range_genes + np.random.random(self.n_genes) * (1 + 2*alpha) * range_genes
        
        return child1, child2
    
    def mutate(self, individual: np.ndarray) -> np.ndarray:
        """
        Gaussian mutation
        
        Args:
            individual: Genes to mutate
        
        Returns:
            Mutated genes
        """
        mutated = individual.copy()
        
        # Apply mutation to each gene with probability mutation_rate
        mutation_mask = np.random.random(self.n_genes) < self.mutation_rate
        mutated[mutation_mask] += np.random.randn(np.sum(mutation_mask)) * 0.05
        
        # Constraint: limit coefficient magnitudes
        mutated = np.clip(mutated, -1.0, 1.0)
        
        return mutated
    
    def evolve(self, flow_conditions: Dict = None) -> Tuple[np.ndarray, AeroResults]:
        """
        Run genetic algorithm optimization
        
        Args:
            flow_conditions: Flow conditions for evaluation
        
        Returns:
            Best individual and its results
        """
        if flow_conditions is None:
            flow_conditions = {
                'velocity': 1.0,
                'angle_of_attack': 5.0
            }
        
        print(f"Starting Genetic Algorithm Optimization")
        print(f"Population: {self.population_size}, Generations: {self.n_generations}")
        print(f"Flow Conditions: {flow_conditions}")
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
            
            # Elitism: keep best individuals
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
            
            # Combine elites and offspring
            population = np.vstack([elites, offspring[:self.population_size - self.elite_size]])
        
        print("\n" + "="*60)
        print("Optimization Complete!")
        print(f"Best Fitness: {best_ever_fitness:.2f}")
        print(f"Best L/D: {best_ever_results.lift_to_drag:.2f}")
        print(f"Best Drag: {best_ever_results.drag:.4f}")
        
        return best_ever_individual, best_ever_results
    
    def save_results(self, filename: str, best_individual: np.ndarray, results: AeroResults):
        """Save optimization results to file"""
        data = {
            'best_genes': best_individual.tolist(),
            'results': {
                'lift': results.lift,
                'drag': results.drag,
                'lift_to_drag': results.lift_to_drag,
                'stability_moment': results.stability_moment,
                'fitness': results.fitness
            },
            'history': self.history,
            'config': {
                'n_genes': self.n_genes,
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
    n_genes: int = 30,
    population_size: int = 50,
    n_generations: int = 100,
    velocity: float = 50.0,
    angle_of_attack: float = 5.0,
    output_file: str = 'optimization_results.json'
) -> Tuple[np.ndarray, AeroResults]:
    """
    Convenience function to run optimization with default parameters
    
    Args:
        n_genes: Number of spectral coefficients (default: 30)
        population_size: Population size (default: 50)
        n_generations: Number of generations (default: 100)
        velocity: Flight velocity in m/s (default: 50.0)
        angle_of_attack: Angle of attack in degrees (default: 5.0)
        output_file: Output filename (default: 'optimization_results.json')
    
    Returns:
        Tuple of (best_genes, results)
    
    Example:
        >>> best_genes, results = run_optimization(n_generations=50)
        >>> print(f"L/D: {results.lift_to_drag:.2f}")
    """
    # Create optimizer
    ga = GeneticAlgorithm(
        n_genes=n_genes,
        population_size=population_size,
        n_generations=n_generations,
        mutation_rate=0.15,
        crossover_rate=0.8,
        elite_size=5
    )
    
    # Define flow conditions
    flow_conditions = {
        'velocity': velocity,
        'angle_of_attack': angle_of_attack
    }
    
    # Run optimization
    best_genes, results = ga.evolve(flow_conditions)
    
    # Save results
    ga.save_results(output_file, best_genes, results)
    
    # Save geometry
    mesh = ga.mesh_generator.encode_geometry(best_genes)
    geometry_file = output_file.replace('.json', '_geometry.txt')
    np.savetxt(geometry_file, mesh['vertices'], 
               header='x,y coordinates of optimized airfoil',
               fmt='%.6f')
    print(f"Geometry saved to {geometry_file}")
    
    return best_genes, results


def quick_test():
    """
    Quick test with small population for rapid testing
    
    Example:
        >>> quick_test()
    """
    print("\n" + "="*70)
    print("QUICK TEST - Small population for rapid testing")
    print("="*70)
    
    return run_optimization(
        n_genes=30,
        population_size=10,
        n_generations=5,
        output_file='quick_test_results.json'
    )


def visualize_airfoil(genes: np.ndarray, title: str = "Optimized Airfoil"):
    """
    Visualize an airfoil from spectral coefficients
    
    Args:
        genes: Spectral coefficients
        title: Plot title
    
    Example:
        >>> best_genes, results = run_optimization()
        >>> visualize_airfoil(best_genes)
    """
    try:
        import matplotlib.pyplot as plt
        
        mesh_gen = SpectralFieldMesh(n_modes=len(genes) // 3)
        mesh = mesh_gen.encode_geometry(genes)
        vertices = mesh['vertices']
        
        plt.figure(figsize=(12, 6))
        plt.plot(vertices[:, 0], vertices[:, 1], 'b-', linewidth=2)
        plt.fill(vertices[:, 0], vertices[:, 1], alpha=0.3)
        plt.axis('equal')
        plt.grid(True, alpha=0.3)
        plt.xlabel('x/c')
        plt.ylabel('y/c')
        plt.title(title)
        plt.tight_layout()
        plt.savefig('airfoil_shape.png', dpi=150)
        print("Airfoil visualization saved to airfoil_shape.png")
        plt.show()
        
    except ImportError:
        print("matplotlib not installed. Install with: pip install matplotlib")


# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    """
    Main function - runs when script is executed directly
    
    Usage:
        python aerodynamic_optimizer_complete.py
    """
    print("\n" + "="*70)
    print("AERODYNAMIC OPTIMIZER")
    print("Genetic Algorithm + Spectral Fields + XLB CFD")
    print("="*70)
    
    # Run optimization with default parameters
    best_genes, results = run_optimization(
        n_genes=30,
        population_size=20,
        n_generations=10,
        velocity=50.0,
        angle_of_attack=5.0,
        output_file='optimization_results.json'
    )
    
    print("\n" + "="*70)
    print("FINAL RESULTS")
    print("="*70)
    print(f"Lift Coefficient: {results.lift:.4f}")
    print(f"Drag Coefficient: {results.drag:.4f}")
    print(f"L/D Ratio: {results.lift_to_drag:.2f}")
    print(f"Stability (Cm): {results.stability_moment:.4f}")
    print(f"Fitness Score: {results.fitness:.2f}")
    print("="*70)
    
    # Try to visualize
    try:
        visualize_airfoil(best_genes, "Optimized Airfoil Shape")
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
    print("   from aerodynamic_optimizer_complete import quick_test")
    print("   quick_test()")
    print("\n2. Custom optimization:")
    print("   from aerodynamic_optimizer_complete import run_optimization")
    print("   best, results = run_optimization(n_generations=100, velocity=60.0)")
    print("\n3. Visualize results:")
    print("   from aerodynamic_optimizer_complete import visualize_airfoil")
    print("   visualize_airfoil(best_genes)")
    print("="*70)


# ============================================================================
# PRODUCTION XLB INTERFACE (OPTIONAL)
# ============================================================================
"""
PRODUCTION XLB INTERFACE

To use real XLB instead of the mock solver:

1. Install XLB:
   pip install xlb-jax jax jaxlib

2. Uncomment the code below

3. Use ProductionXLBInterface:
   ga = GeneticAlgorithm(n_genes=30)
   ga.cfd_solver = ProductionXLBInterface(reynolds_number=1e5)
   
4. For GPU support:
   pip install jax[cuda12]

# UNCOMMENT BELOW FOR PRODUCTION XLB:
'''
import jax
import jax.numpy as jnp

class ProductionXLBInterface:
    def __init__(self, reynolds_number: float = 1000, mach_number: float = 0.1):
        self.reynolds = reynolds_number
        self.mach = mach_number
        self.lattice_velocity = 0.1
        self.tau = 0.5 + 3.0 * self.lattice_velocity * 1.0 / self.reynolds
        
    def setup_simulation(self, mesh: dict, flow_conditions: dict) -> dict:
        # Import XLB components
        from xlb import LBMSolver
        from xlb.boundary_conditions import BounceBackBC, VelocityBC, PressureBC
        from xlb.grid import Grid
        from xlb.operator.equilibrium import QuadraticEquilibrium
        
        nx, ny, nz = mesh['resolution']
        angle = flow_conditions.get('angle_of_attack', 0.0)
        velocity = flow_conditions.get('velocity', 1.0)
        alpha = np.deg2rad(angle)
        
        # Create grid
        grid = Grid(shape=(nx, ny, nz), velocity_set='D3Q19')
        
        # Boundary conditions
        inlet_velocity = jnp.array([
            velocity * np.cos(alpha),
            velocity * np.sin(alpha),
            0.0
        ]) * self.lattice_velocity
        
        inlet_bc = VelocityBC(grid=grid, indices=jnp.where(jnp.arange(nx) == 0), velocity=inlet_velocity)
        outlet_bc = PressureBC(grid=grid, indices=jnp.where(jnp.arange(nx) == nx-1), pressure=1.0)
        solid_bc = BounceBackBC(grid=grid, indices=jnp.where(jnp.array(mesh['solid_mask'])))
        
        solver = LBMSolver(
            grid=grid,
            boundary_conditions=[inlet_bc, outlet_bc, solid_bc],
            tau=self.tau,
            equilibrium=QuadraticEquilibrium()
        )
        
        return {'solver': solver, 'grid': grid}
    
    def run_simulation(self, config: dict, n_steps: int = 10000) -> dict:
        solver = config['solver']
        f = solver.initialize_equilibrium()
        
        for step in range(n_steps):
            f = solver.collision(f)
            f = solver.streaming(f)
            f = solver.apply_boundary_conditions(f)
            
            if step % 100 == 0:
                macroscopic = solver.compute_macroscopic(f)
                residual = solver.compute_residual(macroscopic)
                if residual < 1e-6:
                    break
        
        macroscopic = solver.compute_macroscopic(f)
        forces = self._compute_surface_forces(solver, f, config['solid_mask'])
        
        return {
            'velocity': np.array(macroscopic['velocity']),
            'pressure': np.array(macroscopic['density']),
            'forces': forces
        }
    
    def _compute_surface_forces(self, solver, f, solid_mask):
        force_density = solver.compute_boundary_force(f, solid_mask)
        total_force = jnp.sum(force_density, axis=(0, 1, 2))
        
        drag = float(total_force[0])
        lift = float(total_force[1])
        
        q_inf = 0.5 * 1.0 * self.lattice_velocity**2
        ref_area = 1.0
        
        return {
            'lift_coefficient': lift / (q_inf * ref_area),
            'drag_coefficient': drag / (q_inf * ref_area),
            'moment_coefficient': 0.0
        }
    
    def compute_aerodynamic_forces(self, results: dict, reference_area: float = 1.0):
        forces = results['forces']
        cl = forces['lift_coefficient']
        cd = forces['drag_coefficient']
        cm = forces['moment_coefficient']
        
        q_inf = 0.5 * 1.225 * 1.0**2
        lift = cl * q_inf * reference_area
        drag = cd * q_inf * reference_area
        ld_ratio = cl / cd if cd > 0 else 0.0
        stability = -cm
        fitness = ld_ratio * 10.0 + stability * 5.0 - cd * 20.0
        
        return AeroResults(lift, drag, cm, ld_ratio, fitness)
'''
"""
