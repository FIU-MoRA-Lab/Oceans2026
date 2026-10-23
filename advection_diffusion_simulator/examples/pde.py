import os
import sys

import dolfin as df
import matplotlib.pyplot as plt
import numpy as np
import tqdm

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from advection_diffusion_simulator.utils import MyVariableExpression

N_POINTS_P_AXIS = 20
TIME = 1.0


# Define forcing term
def forcing_term_test(x, y):
    return 0.0


# Define Velocity field
def velocity_field_test(x, y):
    return 0.7 * np.array([1.0, 1.0])


# Define Initial condition
def initial_condition_test(x, y):
    return 30 * np.exp(-(np.sqrt((x - 0.4) ** 2 + (y - 0.2) ** 2) / 0.01))


# Define boundary conditions
def boundary_condition_test(x, y):
    return 0.0


def solve_pde(
    mesh=df.UnitSquareMesh(
        N_POINTS_P_AXIS, N_POINTS_P_AXIS
    ),  # Unit square mesh [0, 1]x[0, 1]
    time=1.0,
    dt=0.1,
    d=0.005,
    forcing_term=forcing_term_test,
    velocity_field=velocity_field_test,
    initial_condition=initial_condition_test,
    boundary_condition=boundary_condition_test,
):
    dts = np.arange(dt, time, dt)
    # Mesh and Finite Element Discretization

    element_degree = 2  # Specify the degree of the element
    element_family = "P"  # Specify the element family

    # Create the Lagrange element
    element = df.FiniteElement(element_family, mesh.ufl_cell(), element_degree)

    # Define a function space using the Lagrange element
    function_space = df.FunctionSpace(mesh, element)
    vector_function_space = df.VectorFunctionSpace(mesh, element_family, element_degree)

    bc = MyVariableExpression(boundary_condition)
    bc = df.interpolate(bc, function_space)  # BC written as a python function

    def u0_boundary(x, on_boundary):
        return on_boundary

    bc = df.DirichletBC(function_space, bc, u0_boundary)

    forcing = MyVariableExpression(forcing_term)
    forcing = df.interpolate(
        forcing, function_space
    )  # Forcing term written as a python function

    velocity_f = MyVariableExpression(velocity_field, dimension=2)
    velocity_f = df.interpolate(
        velocity_f, vector_function_space
    )  # Velocity field written as a python function

    u0 = MyVariableExpression(initial_condition)
    u = df.interpolate(u0, function_space)  # BC written as a python function

    dt, d = df.Constant(dt), df.Constant(d)
    # Trial and Test Functions
    u_trial = df.TrialFunction(function_space)
    v_test = df.TestFunction(function_space)

    solutions = [df.interpolate(u0, function_space)]
    gradients = [df.grad(solutions[0])]

    print(
        "This code solves the advection diffusion equation as described in https://en.wikipedia.org/wiki/Convection%E2%80%93diffusion_equation \n \
          The domain is a square given certain diffusivity coefficient, velocity field, forcing term and initial condition with Dirichlet boundary conditions."
    )
    # time.sleep(1.0)

    for t in tqdm.tqdm(dts):
        # Creates the weak formulation

        u_sol = df.Function(function_space)
        a = (
            u_trial * v_test * df.dx
            + dt
            * df.dot(d * df.grad(u_trial) - u_trial * velocity_f, df.grad(v_test))
            * df.dx
        )
        L = (dt * forcing + u) * v_test * df.dx

        # Solving the variational formulation
        df.solve(a == L, u_sol, bc)
        u.assign(u_sol)
        solutions.append(u_sol)
        gradients.append(df.project(df.grad(u_sol), vector_function_space))

    return solutions, gradients, mesh


if __name__ == "__main__":
    solutions, gradients, _ = solve_pde()

    print(len(solutions))
    solution = lambda x, n: np.array(solutions[n](x))
    gradient = lambda x, n: np.array(gradients[n](x))
    print(solution([0.4, 0.2], 0))
    print(gradient([0.5, 0.5], 5))
    df.plot(solutions[-1])
    plt.show()
