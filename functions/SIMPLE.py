from scipy.sparse import lil_matrix
from scipy.sparse.linalg import spsolve

import numpy as np

def assemble_matrices(M_u, M_v, u, v):
    '''
    Assemble the matrices A⁻¹ and H from MU = AU - H = -∇p 
    - A: diagonal matrix of M
    - A⁻¹: inverse of A
    - H: off-diagonal terms
    '''

    # extract diagonal terms
    A_u = M_u.diagonal()
    A_v = M_v.diagonal()

    # get inverse of diagonal matrix (element‑wise reciprocal)
    A_u_inv = 1 / A_u
    A_v_inv = 1 / A_v

    # compute residual of the diagonal part (i.e. off-diagonal terms)
    H_u = A_u * u - M_u @ u
    H_v = A_v * v - M_v @ v

    # return the assembled matrices
    return A_u_inv, A_v_inv, H_u, H_v


def predicted_velocity(A_u_inv, A_v_inv, H_u, H_v, b_u, b_v):
    '''
    Solve for the intermediate velocity via the following equation:
     
    U* = A⁻¹(H + -∇p)
    '''

    # evaluate the equation for the intermediate velocity satisfying momentum eqns
    U_star_u = A_u_inv * (H_u + b_u)
    U_star_v = A_v_inv * (H_v + b_v)

    # return computed intermediate velocities
    return U_star_u, U_star_v


def pressure_correction(mesh, A_u_inv, A_v_inv, U_star_u, U_star_v, BCs):
    '''
    Solve for the pressure correction via the following equation:
    
    ∇⋅(A⁻¹∇p') = ∇⋅U*
    '''

    # initialize pressure correction matrix (L) and RHS (b_p)
    n_cells = mesh.n_cells

    L = lil_matrix((n_cells, n_cells))
    b_p = np.zeros(n_cells)


    # ================== INTERNAL FACES ================== #
    '''
    For the internal faces, determine flux contributions from the pressure correction term and the 
    intermediate velocity term
    '''

    # loop through each internal face (i.e. a cell on each side of the face)
    for internal_face in mesh.internal_faces:

        # extract face details
        node_1, node_2, cell_P, cell_N = internal_face
        key = (node_1, node_2, cell_P, cell_N)

        # extract face geometry
        n_x, n_y = mesh.face_normals[key]
        face_length = mesh.face_lengths[key]
        face_centroid = mesh.face_centroids[key]

        # compute distances between face and cells
        d_P_to_f = np.linalg.norm(face_centroid - mesh.cell_centroids[cell_P])
        d_N_to_f = np.linalg.norm(face_centroid - mesh.cell_centroids[cell_N])
        d_P_to_N = np.linalg.norm(mesh.cell_centroids[cell_N] - mesh.cell_centroids[cell_P])


        # ---------------- Pressure Correction Flux ---------------- #

        # linear interpolation for face value of A⁻¹
        A_u_inv_face = (d_N_to_f * A_u_inv[cell_P] + d_P_to_f * A_u_inv[cell_N]) / d_P_to_N
        A_v_inv_face = (d_N_to_f * A_v_inv[cell_P] + d_P_to_f * A_v_inv[cell_N]) / d_P_to_N

        # assemble the coefficient for the operation
        D = (face_length / d_P_to_N) * (A_u_inv_face * n_x**2 + A_v_inv_face * n_y**2)

        # pressure correction flux contribution of cell P for pressure correction term
        L[cell_P, cell_P] += -D 
        L[cell_P, cell_N] +=  D

        # pressure correction flux contribution of cell N for pressure correction term
        L[cell_N, cell_N] += -D 
        L[cell_N, cell_P] +=  D 


        # ---------------- Intermediate Velocity Flux ---------------- #

        # linear interpolation for intermediate velocities
        U_star_u_f = (d_N_to_f * U_star_u[cell_P] + d_P_to_f * U_star_u[cell_N]) / d_P_to_N
        U_star_v_f = (d_N_to_f * U_star_v[cell_P] + d_P_to_f * U_star_v[cell_N]) / d_P_to_N

        # intermediate velocity flux contribution through face from cell P to N
        flux_star = U_star_u_f * (n_x * face_length) + U_star_v_f * (n_y * face_length)

        # add the flux contributions to the RHS source term 
        b_p[cell_P] += flux_star
        b_p[cell_N] -= flux_star


    # ================== BOUNDARY FACES ================== #
    '''
    For the boundary faces, determine flux contributions from the pressure correction term and the 
    intermediate velocity term
    '''

    # loop through each boundary face (i.e. a cell on only one side of the face)
    for boundary_face in mesh.boundary_faces:

        # extract face details
        node_1, node_2, cell_P, name = boundary_face
        key = (node_1, node_2, cell_P, name)

        # extract face geometry
        n_x, n_y = mesh.face_normals[key]
        face_length = mesh.face_lengths[key]
        face_centroid = mesh.face_centroids[key]

        # compute distances between face and cells
        d_P_to_f = np.linalg.norm(face_centroid - mesh.cell_centroids[cell_P])

        # pick outlet cell to pin the pressure correction (needed to avoid null space)
        outlet_pin_cell = None
        outlet_cells = []

        # get all outlet boundary faces
        for boundary_face in mesh.boundary_faces:
            if boundary_face[3] == "outlet": 
                outlet_cells.append(boundary_face[2])

        # select out of the middle of the list (only need to pin one correction as the reference)
        if outlet_cells:
            outlet_pin_cell = outlet_cells[len(outlet_cells)//2]

        # ---------------- Inlet ---------------- #
        if name == "inlet":

            # ~~~~~ Pressure Correction Flux ~~~~~ #

            '''
            NO CONTRIBUTION from the pressure correction flux through the inlet face since there is
            a zero-gradient pressure condition.
            '''

            # ~~~~~ Intermediate Velocity Flux ~~~~~ #

            # prescribed velocity at the inlet
            u_f = BCs["inlet u"]
            v_f = BCs["inlet v"]

            # mass flux based on prescribed velocity
            flux_star = (u_f * n_x + v_f * n_y)

            # add intermediate velocity flux contribution to RHS source term
            b_p[cell_P] += flux_star * face_length


        # ---------------- Outlet ---------------- #
        elif name == "outlet":

            # # Then in the boundary loop:
            # if cell_P == outlet_pin_cell:
            #     L[cell_P, :] = 0
            #     L[cell_P, cell_P] = 1
            #     b_p[cell_P] = 0

            #     continue

            # # ~~~~~ Pressure Correction Flux ~~~~~ #

            '''
            NO CONTRIBUTION from the pressure correction flux through the outlet face since the correct
            pressure value has already been enforced.
            '''

            # assemble diffusion coefficient
            D = (face_length / d_P_to_f) * (A_u_inv[cell_P] * n_x**2 + A_v_inv[cell_P] * n_y**2)

            # add contribution of outlet face to pressure correction flux
            L[cell_P, cell_P] += -D

            # ~~~~~ Intermediate Velocity Flux ~~~~~ #
            
            # zero-gradient velocity (use cell centroid values)
            u_f = U_star_u[cell_P]
            v_f = U_star_v[cell_P]

            # mass flux based on prescribed velocity
            flux_star = (u_f * n_x + v_f * n_y)

            # add intermediate velocity flux contribution to RHS source term
            b_p[cell_P] += flux_star * face_length


        # ---------------- Slip Wall ---------------- #
        elif name == "slip wall": 
            pass

            # ~~~~~ Pressure Correction Flux ~~~~~ #

            '''
            NO CONTRIBUTION from the pressure correction flux through the slip-wall face since there is
            a zero-gradient pressure normal to the slip wall
            '''

            # ~~~~~ Intermediate Velocity Flux ~~~~~ #

            '''
            NO CONTRIBUTION from the intermediate velocity flux through the slip-wall face since there is no 
            component of velocity normal to the face i.e. mass flux = 0.
            '''
            

    # solve pressure correction equation
    L = L.tocsr()
    p_prime = spsolve(L, b_p)

    residual = L @ p_prime - b_p
    print(f"Pressure correction residual: {np.linalg.norm(residual):.6e}")

    # return the pressure correction vector
    return p_prime


def correct_v_and_p(iter, mesh, u, v, p, U_star_u, U_star_v, A_u_inv, A_v_inv, p_prime, URFs):
    '''
    Correct the velocity and the pressure via the following update equations:
    
    pᵏ⁺¹ = pᵏ + αₚp'
    
    Uᵏ⁺¹ = Uᵏ + αᵤ[(U* - A⁻¹∇p) - Uᵏ]
    '''

    # initialize pressure gradient vectors
    n_cells = mesh.n_cells
    grad_p_x = np.zeros(n_cells)
    grad_p_y = np.zeros(n_cells)


    # ================== INTERNAL FACES ================== #
    '''
    For the internal faces, compute the pressure gradient contributions
    '''

    # loop through each internal face (i.e. a cell on each side of the face)
    for internal_face in mesh.internal_faces:

        # extract face details
        node_1, node_2, cell_P, cell_N = internal_face
        key = (node_1, node_2, cell_P, cell_N)

        # extract face geometry
        n_x, n_y = mesh.face_normals[key]
        face_length = mesh.face_lengths[key]
        face_centroid = mesh.face_centroids[key]

        cell_P_area = mesh.cell_areas[cell_P]
        cell_N_area = mesh.cell_areas[cell_N]

        # compute distances between face and cells
        d_P_to_f = np.linalg.norm(face_centroid - mesh.cell_centroids[cell_P])
        d_N_to_f = np.linalg.norm(face_centroid - mesh.cell_centroids[cell_N])
        d_P_to_N = np.linalg.norm(mesh.cell_centroids[cell_N] - mesh.cell_centroids[cell_P])

        # linear interpolation for face value of p'
        p_prime_f = (d_N_to_f * p_prime[cell_P] + d_P_to_f * p_prime[cell_N]) / d_P_to_N

        # contribution to pressure gradient at cell P
        grad_p_x[cell_P] += 1/cell_P_area * p_prime_f * (n_x * face_length) 
        grad_p_y[cell_P] += 1/cell_P_area * p_prime_f * (n_y * face_length)

        # contribution to pressure gradient at cell N
        grad_p_x[cell_N] -= 1/cell_N_area * p_prime_f * (n_x * face_length)
        grad_p_y[cell_N] -= 1/cell_N_area * p_prime_f * (n_y * face_length) 


    # ================== BOUNDARY FACES ================== #
    '''
    For the boundary faces, compute the pressure gradient contributions
    '''

    # loop through each boundary face (i.e. a cell on only one side of the face)
    for boundary_face in mesh.boundary_faces:

        # extract face details
        node_1, node_2, cell_P, name = boundary_face
        key = (node_1, node_2, cell_P, name)

        # extract face geometry
        n_x, n_y = mesh.face_normals[key]
        face_length = mesh.face_lengths[key]
        face_centroid = mesh.face_centroids[key]

        cell_P_area = mesh.cell_areas[cell_P]

        # compute distances between face and cells
        d_P_to_f = np.linalg.norm(face_centroid - mesh.cell_centroids[cell_P])

        # consider contributions to each boundary
        if "inlet" in name:

            # zero-gradient pressure (use cell centroid value)
            p_prime_f = p_prime[cell_P]

        elif name == "outlet":

            # pressure is prescribed at the outlet, pressure does not need correction at outlet (p' -> 0)
            p_prime_f = 0.0

        elif name == "slip wall":

            # zero-gradient pressure (use cell centroid value)
            p_prime_f = p_prime[cell_P]

        else:
            continue

        # Add boundary contribution
        grad_p_x[cell_P] += 1/cell_P_area * p_prime_f * (n_x * face_length)
        grad_p_y[cell_P] += 1/cell_P_area * p_prime_f * (n_y * face_length)

    # ================== UNDER RELAXATION FACTORS ================== #

    # unpack the URF dictionary
    alpha_u = URFs["alpha u"]
    alpha_v = URFs["alpha v"]
    alpha_p = URFs["alpha p"]

    # linear URF for u
    if alpha_u[0] <= iter <= alpha_u[1]:
        alpha_u_iter = alpha_u[2] + (iter - alpha_u[0]) * (alpha_u[3] - alpha_u[2]) / (alpha_u[1] - alpha_u[0])
    else:
        alpha_u_iter = alpha_u[3]

    # linear URF for v
    if alpha_v[0] <= iter <= alpha_v[1]:
        alpha_v_iter = alpha_v[2] + (iter - alpha_v[0]) * (alpha_v[3] - alpha_v[2]) / (alpha_v[1] - alpha_v[0])
    else:
        alpha_v_iter = alpha_v[3]
        
    # linear URF for p
    if alpha_p[0] <= iter <= alpha_p[1]:
        alpha_p_iter = alpha_p[2] + (iter - alpha_p[0]) * (alpha_p[3] - alpha_p[2]) / (alpha_p[1] - alpha_p[0])
    else:
        alpha_p_iter = alpha_p[3]


    # ================== SOLUTION UPDATE ================== #

    # evaluate the solution update equations for velocity and pressure
    u_new = u + alpha_u_iter * ((U_star_u - A_u_inv * grad_p_x) - u)
    v_new = v + alpha_v_iter * ((U_star_v - A_v_inv * grad_p_y) - v)
    p_new = p + alpha_p_iter * p_prime

    # return the updated velocities and pressure
    return u_new, v_new, p_new
