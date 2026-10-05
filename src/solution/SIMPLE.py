from scipy.sparse import lil_matrix
from scipy.sparse.linalg import spsolve

from src.numerics import (
    linear_interp,
    compute_grad,
    rhie_and_chow_interp,
    non_ortho_corr_flux_p_prime,
)

import numpy as np


# ============================================================ #
#                       MOMENTUM SPLIT                         #
# ============================================================ #

def assemble_matrices(M_u, M_v, soln):
    """
    Split M into  A (diagonal)  and  H (off-diagonal contribution):
        M·U = A·U - H
    """

    # diagonalization of momentum matrix: A = diag(M)
    A_u = M_u.diagonal()
    A_v = M_v.diagonal()

    # inverse of diagonal: A⁻¹ = 1/A (note: A is diagonal)
    A_u_inv = 1.0 / A_u
    A_v_inv = 1.0 / A_v

    # off-diagonal contributions: H = A·U - M·U
    H_u = A_u * soln.u - M_u @ soln.u
    H_v = A_v * soln.v - M_v @ soln.v

    return A_u_inv, A_v_inv, H_u, H_v


def predicted_velocity(A_u_inv, A_v_inv, H_u, H_v, b_u, b_v):
    """
    Intermediate velocity:  U* = A⁻¹ (H + b)
    """

    U_star_u = A_u_inv * (H_u + b_u)
    U_star_v = A_v_inv * (H_v + b_v)

    return U_star_u, U_star_v


# ============================================================ #
#                   PRESSURE CORRECTION SOLVE                  #
# ============================================================ #

def pressure_correction(mesh, A_u_inv, A_v_inv, p, U_star_u, U_star_v, BCs, corr_iters):
    """
    Solve the pressure-correction Poisson equation
        ∇⋅(A⁻¹ ∇p') = ∇⋅U*

    Note: 
    - The non-orthogonal correction is applied explicitly (to the RHS) and the orthogonal part
      is solved implicitly
    - The non-orthogonal correction is applied iteratively as specified by user
    - An over-relaxed approach is used for the decomposition of the face normal into orthogonal
      and non-orthogonal components
    """

    n_cells = mesh.n_cells

    # Poisson matrix and RHS
    L   = lil_matrix((n_cells, n_cells))
    b_p = np.zeros(n_cells)

    # cell-centered pressure gradient (needed for Rhie–Chow)
    grad_p_x, grad_p_y = compute_grad(mesh, None, None, p, BCs, field="p")


    # ================== INTERNAL FACES ================== #
    for internal_face in mesh.internal_faces:

        node_1, node_2, cell_P, cell_N = internal_face
        key = (node_1, node_2, cell_P, cell_N)

        n_x, n_y      = mesh.face_normals[key]
        face_length   = mesh.face_lengths[key]
        face_centroid = mesh.face_centroids[key]

        d_P_to_f = np.linalg.norm(face_centroid - mesh.cell_centroids[cell_P])
        d_N_to_f = np.linalg.norm(face_centroid - mesh.cell_centroids[cell_N])

        # face-interpolated A⁻¹
        A_u_inv_face = linear_interp(A_u_inv[cell_P], A_u_inv[cell_N], d_P_to_f, d_N_to_f)
        A_v_inv_face = linear_interp(A_v_inv[cell_P], A_v_inv[cell_N], d_P_to_f, d_N_to_f)

        # implicit orthogonal part of the p' Poisson equation: Dp' = ∇⋅U*
        d_P_to_N_vec = mesh.cell_centroids[cell_N] - mesh.cell_centroids[cell_P]
        d_P_to_N_sq = np.dot(d_P_to_N_vec, d_P_to_N_vec)

        n_sq = n_x**2 + n_y**2
        n_dot_d  = n_x * d_P_to_N_vec[0] + n_y * d_P_to_N_vec[1]

        D = face_length * (
            n_sq / (n_dot_d * d_P_to_N_sq)
        ) * (
            A_u_inv_face * d_P_to_N_vec[0]**2
            + A_v_inv_face * d_P_to_N_vec[1]**2
        )     

        L[cell_P, cell_P] += -D
        L[cell_P, cell_N] +=  D
        L[cell_N, cell_N] += -D
        L[cell_N, cell_P] +=  D

        # Rhie–Chow interpolation for intermediate-velocity flux
        U_star_u_f, U_star_v_f = rhie_and_chow_interp(
            mesh,
            cell_P, cell_N,
            n_x, n_y,
            d_P_to_f, d_N_to_f,
            U_star_u[cell_P], U_star_u[cell_N],
            U_star_v[cell_P], U_star_v[cell_N],
            p[cell_P], p[cell_N],
            grad_p_x[cell_P], grad_p_x[cell_N],
            grad_p_y[cell_P], grad_p_y[cell_N],
            A_u_inv_face, A_v_inv_face,
        )

        # compute the flux contribution to RHS using Rhie-Chow interpolated velocity
        flux_star = (U_star_u_f * n_x + U_star_v_f * n_y) * face_length
        b_p[cell_P] += flux_star
        b_p[cell_N] -= flux_star


    # ================== BOUNDARY FACES ================== #
    for boundary_face in mesh.boundary_faces:

        node_1, node_2, cell, name = boundary_face
        key = (node_1, node_2, cell, name)

        n_x, n_y      = mesh.face_normals[key]
        face_length   = mesh.face_lengths[key]
        face_centroid = mesh.face_centroids[key]

        d_P_to_f = np.linalg.norm(face_centroid - mesh.cell_centroids[cell])

        # 'pressure_correction_face_flux' supplies the diagonal contribution for the BC
        D = BCs[name].pressure_correction_face_flux(
            A_u_inv[cell], A_v_inv[cell],
            n_x, n_y, d_P_to_f, face_length
        )
        L[cell, cell] += D

        # 'intermediate_velocity_face_flux' supplies the RHS contribution for the BC
        b_p[cell] += BCs[name].intermediate_velocity_face_flux(
            U_star_u[cell], U_star_v[cell],
            n_x, n_y, face_length
        )


    # ================== NON-ORTHOGONAL CORRECTION ================== #
    L = L.tocsr()

    for corrector in range(corr_iters + 1):

        # solve the implicit (orthogonal) system with the current RHS
        p_prime = spsolve(L, b_p)

        # last pass: return without recomputing the explicit correction
        if corrector == corr_iters:
            break

        # cell-centered gradient of p' 
        grad_p_prime_x, grad_p_prime_y = compute_grad(mesh, None, None, p_prime, BCs, field="p_prime")

        # explicit non-orthogonal flux correction
        b_p_prime_non_ortho = np.zeros(n_cells)

        # ================== INTERNAL FACES ================== #
        for internal_face in mesh.internal_faces:

            node_1, node_2, cell_P, cell_N = internal_face
            key = (node_1, node_2, cell_P, cell_N)

            n_x, n_y      = mesh.face_normals[key]
            face_length   = mesh.face_lengths[key]
            face_centroid = mesh.face_centroids[key]

            cell_centroid_P = mesh.cell_centroids[cell_P]
            cell_centroid_N = mesh.cell_centroids[cell_N]

            d_P_to_f = np.linalg.norm(face_centroid - cell_centroid_P)
            d_N_to_f = np.linalg.norm(face_centroid - cell_centroid_N)

            A_u_inv_face = linear_interp(A_u_inv[cell_P], A_u_inv[cell_N], d_P_to_f, d_N_to_f)
            A_v_inv_face = linear_interp(A_v_inv[cell_P], A_v_inv[cell_N],d_P_to_f, d_N_to_f)

            flux_non = non_ortho_corr_flux_p_prime(
                n_x, n_y,
                face_centroid, face_length,
                cell_centroid_P, cell_centroid_N,
                grad_p_prime_x[cell_P], grad_p_prime_x[cell_N],
                grad_p_prime_y[cell_P], grad_p_prime_y[cell_N],
                A_u_inv_face, A_v_inv_face,
            )

            b_p_prime_non_ortho[cell_P] -= flux_non
            b_p_prime_non_ortho[cell_N] += flux_non

        # ================== BOUNDARY FACES ================== #
        for boundary_face in mesh.boundary_faces:

            node_1, node_2, cell, name = boundary_face
            key = (node_1, node_2, cell, name)

            n_x, n_y      = mesh.face_normals[key]
            face_length   = mesh.face_lengths[key]
            face_centroid = mesh.face_centroids[key]

            cell_centroid_P = mesh.cell_centroids[cell]

            d_P_to_f = np.linalg.norm(face_centroid - cell_centroid_P)

            A_u_inv_face = A_u_inv[cell] # <- boundary 'trick'
            A_v_inv_face = A_v_inv[cell] # <- boundary 'trick'

            flux_non = non_ortho_corr_flux_p_prime(
                n_x, n_y,
                face_centroid, face_length,
                cell_centroid_P, face_centroid,                 # <- boundary "trick"
                grad_p_prime_x[cell_P], grad_p_prime_x[cell_P], # <- boundary 'trick'
                grad_p_prime_y[cell_P], grad_p_prime_y[cell_P], # <- boundary 'trick'
                A_u_inv_face, A_v_inv_face, 
            )

            b_p_prime_non_ortho[cell] -= flux_non

        # add the non-orthogonal correction to the RHS for the next iteration
        b_p += b_p_prime_non_ortho

    return p_prime


# ============================================================ #
#                       SOLUTION UPDATE                        #
# ============================================================ #

def correct_v_and_p(iter, mesh, u, v, p,
                    U_star_u, U_star_v,
                    A_u_inv, A_v_inv, p_prime,
                    URFs, BCs):
    """
    Update the velocity and pressure:
        pᵏ⁺¹ = pᵏ + α_p · p'
        Uᵏ⁺¹ = Uᵏ + α_U [ (U* - A⁻¹ ∇p') - Uᵏ ]
    """
    
    # compute gradients of p' for the velocity update
    grad_p_prime_x, grad_p_prime_y = compute_grad(
        mesh, None, None, p_prime, BCs, field="p_prime"
    )

    # extract under-relaxation factors for this iterations
    alpha_u_iter = _linear_urf(iter, URFs["alpha-u"])
    alpha_v_iter = _linear_urf(iter, URFs["alpha-v"])
    alpha_p_iter = _linear_urf(iter, URFs["alpha-p"])

    # update the solution
    u_new = u + alpha_u_iter * ((U_star_u - A_u_inv * grad_p_prime_x) - u)
    v_new = v + alpha_v_iter * ((U_star_v - A_v_inv * grad_p_prime_y) - v)
    p_new = p + alpha_p_iter * p_prime

    return u_new, v_new, p_new


def _linear_urf(iter, spec):
    """
    Linear ramp for an under-relaxation factor.

    spec = [start_iter, end_iter, start_value, end_value]
    """

    start, end, urf_start, urf_end = spec
    
    if start <= iter <= end:
        return urf_start + (iter - start) * (urf_end - urf_start) / (end - start)
    
    elif iter > end:
        return urf_end
