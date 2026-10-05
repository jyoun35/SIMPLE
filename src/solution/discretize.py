from scipy.sparse import lil_matrix, diags
from src.numerics import compute_grad, linear_interp, rhie_and_chow_interp, non_ortho_corr_flux

import numpy as np


def build_momentum_matrix(iter, mesh, soln, mu, BCs, A_u_inv, A_v_inv):
    """
    Build the spatial operator of the discretized incompressible N-S eqns,
    i.e. the M in M·U = -∇p.
    """

    # extract current solution
    u = soln.u
    v = soln.v
    p = soln.p

    # initialize spatial operator matrices
    M_u = lil_matrix((mesh.n_cells, mesh.n_cells))
    M_v = lil_matrix((mesh.n_cells, mesh.n_cells))

    # pressure gradient source (integral form; divide by area at the end)
    grad_p_x, grad_p_y = compute_grad(mesh, u, v, p, BCs, field="p")
    b_u = -grad_p_x * mesh.cell_areas
    b_v = -grad_p_y * mesh.cell_areas

    # velocity gradients for the explicit non-orthogonal correction in diffusive flux
    grad_u_x, grad_u_y = compute_grad(mesh, u, v, p, BCs, field="u")
    grad_v_x, grad_v_y = compute_grad(mesh, u, v, p, BCs, field="v")


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

        # ---------------- Convective Flux (upwind) ---------------- #
        # linear inteprolation for first iteration
        if iter == 0:
            u_f = linear_interp(u[cell_P], u[cell_N], d_P_to_f, d_N_to_f)
            v_f = linear_interp(v[cell_P], v[cell_N], d_P_to_f, d_N_to_f)

        # Rhie and Chow interpolation for subsequent iterations
        else:
            # face-interpolated A⁻¹
            A_u_inv_face = linear_interp(A_u_inv[cell_P], A_u_inv[cell_N], d_P_to_f, d_N_to_f)
            A_v_inv_face = linear_interp(A_v_inv[cell_P], A_v_inv[cell_N], d_P_to_f, d_N_to_f)

            u_f, v_f = rhie_and_chow_interp(
                        mesh,
                        cell_P, cell_N,
                        n_x, n_y,
                        d_P_to_f, d_N_to_f,
                        u[cell_P], u[cell_N], 
                        v[cell_P], v[cell_N], 
                        p[cell_P], p[cell_N], 
                        grad_p_x[cell_P], grad_p_x[cell_N],
                        grad_p_y[cell_P], grad_p_y[cell_N],
                        A_u_inv_face, A_v_inv_face
                    )

        # compute mass flux at the face
        m_dot = u_f * n_x + v_f * n_y

        # u-momentum
        M_u[cell_P, cell_P] += max( m_dot, 0.0) * face_length
        M_u[cell_P, cell_N] += min( m_dot, 0.0) * face_length
        M_u[cell_N, cell_N] += max(-m_dot, 0.0) * face_length
        M_u[cell_N, cell_P] += min(-m_dot, 0.0) * face_length

        # v-momentum
        M_v[cell_P, cell_P] += max( m_dot, 0.0) * face_length
        M_v[cell_P, cell_N] += min( m_dot, 0.0) * face_length
        M_v[cell_N, cell_N] += max(-m_dot, 0.0) * face_length
        M_v[cell_N, cell_P] += min(-m_dot, 0.0) * face_length


        # ---------------- Diffusive Flux (orthogonal + non-orthogonal) ---------------- #
        coeff_ortho_u, explicit_non_ortho_u = non_ortho_corr_flux(
            n_x, n_y,
            face_centroid, face_length,
            cell_centroid_P, cell_centroid_N,
            grad_u_x[cell_P], grad_u_x[cell_N],
            grad_u_y[cell_P], grad_u_y[cell_N],
        )

        coeff_ortho_v, explicit_non_ortho_v = non_ortho_corr_flux(
            n_x, n_y,
            face_centroid, face_length,
            cell_centroid_P, cell_centroid_N,
            grad_v_x[cell_P], grad_v_x[cell_N],
            grad_v_y[cell_P], grad_v_y[cell_N],
        )

        # u-momentum — implicit orthogonal part
        M_u[cell_P, cell_P] +=  mu * coeff_ortho_u
        M_u[cell_P, cell_N] += -mu * coeff_ortho_u
        M_u[cell_N, cell_N] +=  mu * coeff_ortho_u
        M_u[cell_N, cell_P] += -mu * coeff_ortho_u

        # u-momentum — explicit non-orthogonal part
        b_u[cell_P] +=  mu * explicit_non_ortho_u
        b_u[cell_N] += -mu * explicit_non_ortho_u

        # v-momentum — implicit orthogonal part
        M_v[cell_P, cell_P] +=  mu * coeff_ortho_v
        M_v[cell_P, cell_N] += -mu * coeff_ortho_v
        M_v[cell_N, cell_N] +=  mu * coeff_ortho_v
        M_v[cell_N, cell_P] += -mu * coeff_ortho_v

        # v-momentum — explicit non-orthogonal part
        b_v[cell_P] +=  mu * explicit_non_ortho_v
        b_v[cell_N] += -mu * explicit_non_ortho_v


    # ================== BOUNDARY FACES ================== #
    for boundary_face in mesh.boundary_faces:

        node_1, node_2, cell, name = boundary_face
        key = (node_1, node_2, cell, name)

        n_x, n_y      = mesh.face_normals[key]
        face_length   = mesh.face_lengths[key]
        face_centroid = mesh.face_centroids[key]

        cell_centroid_P = mesh.cell_centroids[cell]

        # cell-centered fields at this boundary cell
        u_cell = u[cell]
        v_cell = v[cell]

        grad_u_x_P = grad_u_x[cell]
        grad_u_y_P = grad_u_y[cell]
        grad_v_x_P = grad_v_x[cell]
        grad_v_y_P = grad_v_y[cell]

        # ---------------- Convective Flux ---------------- #
        M_u_conv, M_v_conv, b_u_conv, b_v_conv = BCs[name].convective_face_flux(
            u_cell, v_cell, n_x, n_y, face_length
        )

        # ---------------- Diffusive Flux ---------------- #
        M_u_diff, M_v_diff, b_u_diff, b_v_diff = BCs[name].diffusive_face_flux(
            mu,
            u_cell, v_cell,
            n_x, n_y,
            face_centroid, face_length,
            cell_centroid_P,
            grad_u_x_P, grad_u_y_P,
            grad_v_x_P, grad_v_y_P,
        )

        # accumulate contributions from convective and diffusive fluxes into momentum matrix and source term
        M_u[cell, cell] += M_u_conv + M_u_diff
        M_v[cell, cell] += M_v_conv + M_v_diff

        b_u[cell] += b_u_conv + b_u_diff
        b_v[cell] += b_v_conv + b_v_diff


    # ================== SCALE BY CELL AREA ================== #
    # (integral form  →  per-unit-volume form)
    M_u = diags(1.0 / mesh.cell_areas) @ M_u
    M_v = diags(1.0 / mesh.cell_areas) @ M_v

    b_u = b_u / mesh.cell_areas
    b_v = b_v / mesh.cell_areas

    # convert to CSR for efficient solving
    M_u = M_u.tocsr()
    M_v = M_v.tocsr()

    return M_u, M_v, b_u, b_v