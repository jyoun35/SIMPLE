from scipy.sparse import lil_matrix, diags, csr_matrix

import numpy as np

def build_momentum_matrix(mesh, u, v, p, transport_properties, BCs):
    '''
    Builds the spatial operator of the discretized incompressible N-S eqns
    i.e. the M in MU = -∇p
    '''

    # initialize spatial operator matrices using lil_matrix ('list of lists' sparse matrix format)
    M_u = lil_matrix((mesh.n_cells, mesh.n_cells))
    M_v = lil_matrix((mesh.n_cells, mesh.n_cells))

    # initialize the vectors for the source term (i.e. pressure gradient (-∇p))
    b_u = np.zeros(mesh.n_cells)
    b_v = np.zeros(mesh.n_cells)

    # ================== INTERNAL FACES ================== #
    '''
    For the internal faces, determine flux contributions from convection, diffusion, and pressure gradient
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


        # ---------------- Convective Flux (upwind) ---------------- #
        
        # linear interpolation for face velocity
        u_face = (d_N_to_f * u[cell_P] + d_P_to_f * u[cell_N]) / d_P_to_N
        v_face = (d_N_to_f * v[cell_P] + d_P_to_f * v[cell_N]) / d_P_to_N

        # mass flux (outward from P)
        m_dot = (u_face * n_x + v_face * n_y)

        # convective flux contribution of cell P for u-velocity
        M_u[cell_P, cell_P] += max(m_dot, 0.0) * face_length
        M_u[cell_P, cell_N] += min(m_dot, 0.0) * face_length

        # convective flux contribution of cell N for u-velocity
        M_u[cell_N, cell_N] += max(-m_dot, 0.0) * face_length
        M_u[cell_N, cell_P] += min(-m_dot, 0.0) * face_length

        # convective flux contribution of cell N for u-velocity
        M_v[cell_P, cell_P] += max(m_dot, 0.0) * face_length
        M_v[cell_P, cell_N] += min(m_dot, 0.0) * face_length

        # convective flux contribution of cell N for v-velocity
        M_v[cell_N, cell_N] += max(-m_dot, 0.0) * face_length
        M_v[cell_N, cell_P] += min(-m_dot, 0.0) * face_length


        # ---------------- Diffusive Flux (orthogonal) ---------------- #

        # assemble the diffusion coefficient
        mu = transport_properties["mu"]
        diffusion_coeff = mu * face_length / d_P_to_N

        # diffusive flux contribution of cell P for u-velocity
        M_u[cell_P, cell_P] += diffusion_coeff
        M_u[cell_P, cell_N] += -diffusion_coeff

        # diffusive flux contribution of cell N for u-velocity
        M_u[cell_N, cell_N] += diffusion_coeff
        M_u[cell_N, cell_P] += -diffusion_coeff

        # diffusive flux contribution of cell P for v-velocity
        M_v[cell_P, cell_P] += diffusion_coeff
        M_v[cell_P, cell_N] += -diffusion_coeff

        # diffusive flux contribution of cell N for v-velocity
        M_v[cell_N, cell_N] += diffusion_coeff
        M_v[cell_N, cell_P] += -diffusion_coeff


        # ---------------- Pressure Gradient Source ---------------- #

        # linear interpolation for face pressure
        p_f = (d_N_to_f * p[cell_P] + d_P_to_f * p[cell_N]) / (d_P_to_f + d_N_to_f)

        # pressure gradient source contribution of cell P for u-velocity
        b_u[cell_P] += -p_f * n_x*face_length

        # pressure gradient source contribution of cell N for u-velocity
        b_u[cell_N] +=  p_f * n_x*face_length   

        # pressure gradient source contribution of cell P for v-velocity
        b_v[cell_P] += -p_f * n_y*face_length

        # pressure gradient source contribution of cell N for v-velocity
        b_v[cell_N] +=  p_f * n_y*face_length

    
    # ================== BOUNDARY FACES ================== #
    '''
    For the boundary faces, determine flux contributions from convection, diffusion, and pressure gradient
    '''

    # loop through each boundary face (i.e. a cell on only one side of the face)
    for boundary_face in mesh.boundary_faces:

        # extract face details
        node_1, node_2, cell, name = boundary_face
        key = (node_1, node_2, cell, name)

        # extract face geometry
        n_x, n_y = mesh.face_normals[key]
        face_length = mesh.face_lengths[key]
        face_centroid = mesh.face_centroids[key]


        # ---------------- Inlet ---------------- #
        if name == "inlet":

            # ~~~~~ Convective Flux ~~~~~ #
            
            # prescribed velocity at the inlet
            u_f = BCs["inlet u"]
            v_f = BCs["inlet v"]

            # mass flux based on prescribed velocity
            m_dot = (u_f * n_x + v_f * n_y)

            # add explicit convective flux contribution to RHS source term
            b_u[cell] -= u_f * m_dot * face_length
            b_v[cell] -= v_f * m_dot * face_length


            # ~~~~~ Diffusive Flux (orthogonal) ~~~~~ #

            # assemble the diffusion coefficient
            d_P_to_f = np.linalg.norm(face_centroid - mesh.cell_centroids[cell])
            diffusion_coeff = mu * face_length / d_P_to_f

            # diffusive flux contibutions through inlet face
            M_u[cell, cell] += diffusion_coeff    
            M_v[cell, cell] += diffusion_coeff

            # add explicit diffusive flux contribution to RHS source term
            b_u[cell] += u_f * diffusion_coeff
            b_v[cell] += v_f * diffusion_coeff


            # ~~~~~ Pressure Gradient Source ~~~~~ #

            # pressure at boundary face (zero-gradient normal pressure)
            p_f = p[cell]

            # add explicit pressure gradient source contribution
            b_u[cell] += -p_f * n_x * face_length
            b_v[cell] += -p_f * n_y * face_length


        # ---------------- Outlet ---------------- #
        elif name == "outlet":

            # ~~~~~ Convective Flux ~~~~~ #

            # zero-gradient velocity (use cell centroid values)
            u_f = u[cell]
            v_f = v[cell]

            # mass flux through outlet
            m_dot = (u_f * n_x + v_f * n_y)

            if m_dot >= 0: # flow out the domain

                # convective flux contributions through the outlet face
                M_u[cell, cell] += m_dot * face_length
                M_v[cell, cell] += m_dot * face_length

            elif m_dot < 0: # backflow scenario (do nothing)
                M_u[cell, cell] += 0
                M_v[cell, cell] += 0


            # ~~~~~ Diffusive Flux ~~~~~ #

            '''
            NO CONTRIBUTION from diffusive flux through the outlet face since there is a zero-
            gradient velocity condition.
            '''

            # ~~~~~ Pressure Gradient Source ~~~~~ #

            # prescribed pressure at the outlet
            p_f = BCs["outlet p"]

            # add explicit pressure gradient source contribution
            b_u[cell] += -p_f * n_x * face_length
            b_v[cell] += -p_f * n_y * face_length


        # ---------------- Slip Wall ---------------- #
        elif name == "slip wall":
            '''
            Enforcing no-penetration (u ⋅ n = 0) and slip (∂u/∂n = 0)
            '''

            # ~~~~~ Convective Flux ~~~~~ #

            '''
            NO CONTRIBUTION from convective flux through the slip wall face since there is no 
            component of velocity normal to the face i.e. mass flux = 0.
            
            Therefore, no flow is going into or out of the slip wall face.
            '''

            # ~~~~~ Diffusive Flux ~~~~~ #

            '''
            NO CONTRIBUTION from diffusive flux through the outlet face since there is no gradient
            of velocity in the normal direction since the slip enforces that ∂u/∂n = 0
            '''

            # ~~~~~ Pressure Gradient Source ~~~~~ #

            # pressure at boundary face (zero-gradient normal pressure)
            p_f = p[cell]

            # add explicit pressure gradient source contribution
            b_u[cell] += -p_f * n_x * face_length
            b_v[cell] += -p_f * n_y * face_length


    # divide by cell areas under assumption of constant pressure gradient across cells (check formulation for details)
    M_u = diags(1 / mesh.cell_areas) @ M_u
    M_v = diags(1 / mesh.cell_areas) @ M_v

    b_u = b_u / mesh.cell_areas
    b_v = b_v / mesh.cell_areas

    # convert lil_matrix to csr_matrix ('compressed sparse row' sparse matrix format) for efficient solving
    M_u = M_u.tocsr()
    M_v = M_v.tocsr()

    # return spatial operator matrix and source term vector of the discretized incompressible N-S eqns
    return M_u, M_v, b_u, b_v
