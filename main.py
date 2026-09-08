from functions.mesh import Mesh
from functions.initialize import *
from functions.discretize import build_momentum_matrix
from functions.SIMPLE import *
from functions.helpers import *

import numpy as np

#===========================================#
#                  INPUTS                   #
#===========================================#

MESH_FILE = "fine_CMesh_NACA_2412.msh"
XFOIL_DATA = "XFOIL_NACA_2412_AoA3.dat"

ITERATIONS = 2000

SAVE_DIR = "results/fine_NACA_2412_AoA3"
SAVE_FREQ = 10

PLOT_MESH = False
PLOT_ICS = False

BOUNDARY_CONDITIONS = {
    "inlet u": 34.03 * np.cos(np.deg2rad(3)),    # [m/s]
    "inlet v": 34.03 * np.sin(np.deg2rad(3)),    # [m/s]
    "outlet p": 101325,                          # [Pa]
}

INITIAL_CONDITIONS = {
    "u": BOUNDARY_CONDITIONS["inlet u"],   # [m/s]
    "v": BOUNDARY_CONDITIONS["inlet v"],   # [m/s]
    "p": BOUNDARY_CONDITIONS["outlet p"],  # [Pa]
}

TRANSPORT_PROPERTIES ={
    "mu": 1.79e-5   # [kg/(m-s)]
}

URF = {
    # name: [start iter, end iter, start value, end value]
    "alpha u": [0, 1000, 0.1,  0.3],
    "alpha v": [0, 1000, 0.1,  0.3],
    "alpha p": [0, 1000, 0.05, 0.15]
}


#===========================================#
#                   SETUP                   #
#===========================================#

# read XFOIL data (used for pressure distribution comparison)
x_xfoil, Cp_xfoil = read_XFOIL_data(XFOIL_DATA)

# process mesh and cell metrics
mesh = Mesh(filename=MESH_FILE, plot=PLOT_MESH)

# initialize solution
u, v, p = initialize_soln(ICs=INITIAL_CONDITIONS, mesh=mesh, plot=PLOT_ICS)

# # load previous solution
# u, v, p = load_soln(file="results/fine_NACA_2412_AoA0/solution_iter_001000.npz", plot=PLOT_ICS)

#===========================================#
#                MAIN LOOP                  #
#===========================================#

# loop through SIMPLE algorithm iterations
for iter in range(ITERATIONS):

    print(f"------------------- Iteration {iter+1} -------------------")

    # discretize the incompressible N-S eqns
    M_u, M_v, b_u, b_v = build_momentum_matrix(mesh, u, v, p, TRANSPORT_PROPERTIES, BOUNDARY_CONDITIONS)

    # assemble matrices for SIMPLE algorithm
    A_u_inv, A_v_inv, H_u, H_v = assemble_matrices(M_u, M_v, u, v)

    # [Step 1] solve for intermediate velocity (U_star)
    U_star_u, U_star_v = predicted_velocity(
        A_u_inv, 
        A_v_inv, 
        H_u, 
        H_v, 
        b_u, 
        b_v
    )

    # [Step 2] solve for the pressure correction (p_prime)
    p_prime = pressure_correction(
        mesh,
        A_u_inv,
        A_v_inv,
        U_star_u,
        U_star_v,
        BCs=BOUNDARY_CONDITIONS
    )

    # [Step 3] correct the velocity and pressure
    u_new, v_new, p_new = correct_v_and_p(
        iter,
        mesh,
        u,
        v,
        p,
        U_star_u,
        U_star_v,
        A_u_inv,
        A_v_inv,
        p_prime,
        URFs=URF
    )

    # compute and print the residual for the continuity equation
    residual = continuity_residual(
        mesh,
        u_new,
        v_new,
        BCs=BOUNDARY_CONDITIONS
    )

    # [Step 4] update values for next iteration
    p = p_new
    u = u_new
    v = v_new

    # save and plot solution
    # if (iter + 1) % 500 == 0:
    #     plot_continuity_residual(mesh, residual)
    if (iter + 1) % SAVE_FREQ == 0:
        save_checkpoint(SAVE_DIR, iter + 1, mesh, u, v, p, BOUNDARY_CONDITIONS, x_xfoil, Cp_xfoil)


# plot final solution
plot_continuity_residual(mesh, residual)
plot_soln(mesh, u, v, p, title="Final Solution")
plot_pressure_distribution(mesh, p, BOUNDARY_CONDITIONS, x_xfoil, Cp_xfoil, c=1.0)

