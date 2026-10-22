from src.boundaries import *
from src.mesh import Mesh
from src.solution import *
from src.helpers import *

import numpy as np

# ============================================================ #
#                           INPUTS                             #
# ============================================================ #

MESH_FILE = "data\\meshes\\fine_flat_plate.msh"

ITERS = 2500
NON_ORTHO_CORRECTION_ITERS = 0

SAVE_DIR = "results\\blasius_validation"
SAVE_FREQ = 10

PLOT_MESH = False
PLOT_ICS  = False

BOUNDARY_CONDITIONS = {
    "velocity-inlet": {
        "u": 34.3 * np.cos(np.deg2rad(0)),  # [m/s]
        "v": 34.3 * np.sin(np.deg2rad(0))   # [m/s]
    },
    "pressure-outlet": {
        "p": 101325.0                       # [Pa]
    },
    "slip-wall": {
    },
    "no-slip-wall":{
    }
}

INITIAL_CONDITIONS = {
    "u": 34.3 * np.cos(np.deg2rad(0)),      # [m/s]
    "v": 34.3 * np.sin(np.deg2rad(0)),      # [m/s]
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

# mesh + geometric quantities
mesh = Mesh(filename=MESH_FILE, plot=PLOT_MESH)

# processed boundary-condition objects (dict: name -> BoundaryCondition)
BCs = process_boundaries(BOUNDARY_CONDITIONS)

# initial condition
soln = initialize_soln(ICs=INITIAL_CONDITIONS, mesh=mesh, plot=PLOT_ICS)

# soln_file = "\\results\\blasius_validation\\{SOLUTION_FILE}.npz"
# soln = load_soln(soln_file, mesh, plot=PLOT_ICS)


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

    # continuity residual for the updated solution
    residual, results = continuity_residual(
        mesh, 
        soln,
        A_u_inv, A_v_inv,
        BCs=BCs,
    )
    residual_hist = np.vstack((residual_hist, [iter, *results]))

    # save checkpoint (plots + .npz)
    if (iter + 1) % SAVE_FREQ == 0:
        save_checkpoint(
            SAVE_DIR, iter + 1,
            mesh,
            soln,
            residual_hist,
            BOUNDARY_CONDITIONS,
            blasius_file=".\\data\\blasius\\blasius_table.csv",    
            x_eval=0.1,                               
            y_eval=0.0025,                        
            nu=TRANS_PROPS["mu"], # considering rho=1
        )


# ============================================================ #
#                        FINAL PLOT                            #
# ============================================================ #

plot_blasius(
    mesh, 
    soln, 
    x_eval=0.1, 
    y_eval=0.0025,
    U_infty = BOUNDARY_CONDITIONS["velocity-inlet"]["u"],
    nu = TRANS_PROPS["mu"], # considering rho=1
    blasius_file=".\\data\\blasius\\blasius_table.csv"
)