from src.boundaries import *
from src.mesh import Mesh
from src.solution import *
from src.helpers import *

import numpy as np

# ============================================================ #
#                           INPUTS                             #
# ============================================================ #

MESH_FILE = "data\\meshes\\fine_CMesh_NACA_2412.msh"
XFOIL_FILE = "data\\xfoil\\XFOIL_NACA_2412_AoA3.dat"

ITERS = 1000
NON_ORTHO_CORRECTION_ITERS = 1

SAVE_DIR = "results\\final_test_2"
SAVE_FREQ = 10

PLOT_MESH = False
PLOT_ICS  = False

BOUNDARY_CONDITIONS = {
    "velocity-inlet": {
        "u": 34.3 * np.cos(np.deg2rad(3)),  # [m/s]
        "v": 34.3 * np.sin(np.deg2rad(3))   # [m/s]
    },
    "pressure-outlet": {
        "p": 101325.0                       # [Pa]
    },
    "slip-wall": {
    }
}

INITIAL_CONDITIONS = {
    "u": 34.3 * np.cos(np.deg2rad(3)),      # [m/s]
    "v": 34.3 * np.sin(np.deg2rad(3)),       # [m/s]
    "p": 101325.0   # [Pa]
    # "rho": 1.225    # [kg/m^3] (CONSTANT rho = 1 is used for now...)
}

TRANS_PROPS = {
    "mu": 1.79e-5,  # [kg/(m·s)]
}

URF = {
    # name: [start iter, end iter, start value, end value]
    "alpha-u": [0, 1000, 0.4, 0.6],
    "alpha-v": [0, 1000, 0.4, 0.6],
    "alpha-p": [0, 1000, 0.2, 0.3],
}


# ============================================================ #
#                            SETUP                             #
# ============================================================ #

# XFOIL reference data for the Cp plot
xfoil_data = read_XFOIL_data(XFOIL_FILE)

# mesh + geometric quantities
mesh = Mesh(filename=MESH_FILE, plot=PLOT_MESH)

# processed boundary-condition objects (dict: name -> BoundaryCondition)
BCs = process_boundaries(BOUNDARY_CONDITIONS)

# initial condition
soln = initialize_soln(ICs=INITIAL_CONDITIONS, mesh=mesh, plot=PLOT_ICS)


# ============================================================ #
#                          MAIN LOOP                           #
# ============================================================ #

residual_hist = np.empty((0, 4))

for iter in range(ITERS):

    print(f"------------------- Iteration {iter + 1} -------------------")

    # [STEP 0] discretize the incompressible N-S eqns
    M_u, M_v, b_u, b_v = build_momentum_matrix(
        iter,
        mesh,
        soln,
        TRANS_PROPS["mu"],
        BCs,
        A_u_inv if iter > 0 else None,
        A_v_inv if iter > 0 else None,
    )

    # [STEP 1a] split M into A (diagonal) and H (off-diagonal)
    A_u_inv, A_v_inv, H_u, H_v = assemble_matrices(M_u, M_v, soln)

    # [STEP 1b] intermediate velocity U* = A⁻¹ (H + b)
    U_star_u, U_star_v = predicted_velocity(
        A_u_inv, A_v_inv, H_u, H_v, b_u, b_v
    )

    # [STEP 2] pressure correction p' via ∇⋅(A⁻¹ ∇p') = ∇⋅U*
    p_prime = pressure_correction(
        mesh,
        A_u_inv, A_v_inv,
        soln.p,
        U_star_u, U_star_v,
        BCs=BCs,
        corr_iters=NON_ORTHO_CORRECTION_ITERS,
    )

    # [STEP 3] correct velocity and pressure
    u_new, v_new, p_new = correct_v_and_p(
        iter,
        mesh,
        soln.u, soln.v, soln.p,
        U_star_u, U_star_v,
        A_u_inv, A_v_inv,
        p_prime,
        URFs=URF,
        BCs=BCs,
    )

    # [STEP 4] update the solution state
    soln = Solution(u=u_new, v=v_new, p=p_new)

    # continuity residual for the new solution
    residual, results = continuity_residual(
        mesh, 
        soln,
        A_u_inv, A_v_inv,
        BCs=BCs,
    )
    residual_hist = np.vstack((residual_hist, [iter, *results]))


    # if (iter + 1) % 50 == 0:
    #     plot_continuity_residual(mesh, residual)
    #     plot_soln(mesh, soln)

    # save checkpoint (plots + .npz)
    if (iter + 1) % SAVE_FREQ == 0:
        save_checkpoint(
            SAVE_DIR, iter + 1,
            mesh, 
            soln,
            residual,
            residual_hist,
            BOUNDARY_CONDITIONS,
            xfoil_data,
        )


# ============================================================ #
#                        FINAL PLOTS                           #
# ============================================================ #

plot_continuity_residual(mesh, residual)
plot_soln(mesh, soln, title="Final Solution")
plot_pressure_distribution(mesh, soln.p, BOUNDARY_CONDITIONS, xfoil_data, c=1.0)