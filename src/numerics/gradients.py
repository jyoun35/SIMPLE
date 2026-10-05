import numpy as np


def _compute_nodal_value(mesh, phi, node):
    """
    Inverse-distance-weighted average of phi over the cells touching `node`.
    """

    phi_node = 0.0
    sum_of_weights = 0.0

    # add the distance-weighted contributions from all cells touching the node
    for node_cell in mesh.node_cells[node]:
        dist = np.linalg.norm(mesh.node_coords[node] - mesh.cell_centroids[node_cell])
        weight = 1.0 / dist

        phi_node += phi[node_cell] * weight
        sum_of_weights += weight

    # compute nodal distance-weighted average
    nodal_avg = phi_node / sum_of_weights

    return nodal_avg


def compute_grad(mesh, u, v, p, BCs, field):
    """
    Cell-centered Green–Gauss gradients using nodal face value reconstruction.
    """

    # determine field to compute gradient for (needed for BCs)
    if field == "u":
        phi = u
    elif field == "v":
        phi = v
    elif field == "p" or field == "p_prime":
        phi = p
    else:
        raise ValueError(f"Unknown field: {field}")

    grad_phi_x = np.zeros(mesh.n_cells)
    grad_phi_y = np.zeros(mesh.n_cells)

    # ================== INTERNAL FACES ================== #
    for internal_face in mesh.internal_faces:

        node_1, node_2, cell_P, cell_N = internal_face
        key = (node_1, node_2, cell_P, cell_N)

        n_x, n_y    = mesh.face_normals[key]
        face_length = mesh.face_lengths[key]

        phi_node_1 = _compute_nodal_value(mesh, phi, node_1)
        phi_node_2 = _compute_nodal_value(mesh, phi, node_2)
        
        phi_f = (1/2) * (phi_node_1 + phi_node_2)

        # Gauss contributions ('n' points outward from cell_P, inward toward cell_N)
        grad_phi_x[cell_P] += phi_f * n_x * face_length
        grad_phi_y[cell_P] += phi_f * n_y * face_length

        grad_phi_x[cell_N] -= phi_f * n_x * face_length
        grad_phi_y[cell_N] -= phi_f * n_y * face_length


    # ================== BOUNDARY FACES ================== #
    for boundary_face in mesh.boundary_faces:

        node_1, node_2, cell, name = boundary_face
        key = (node_1, node_2, cell, name)

        n_x, n_y    = mesh.face_normals[key]
        face_length = mesh.face_lengths[key]

        # compute face value based on boundary condition type
        if field == "u":
            u_cell = u[cell]
            v_cell = v[cell]
            phi_f, _ = BCs[name].get_face_velocity(u_cell, v_cell, n_x, n_y)

        elif field == "v":
            u_cell = u[cell]
            v_cell = v[cell]
            _, phi_f = BCs[name].get_face_velocity(u_cell, v_cell, n_x, n_y)

        elif field == "p":
            p_cell = p[cell]
            phi_f = BCs[name].get_face_pressure(p_cell, n_x, n_y)

        elif field == "p_prime":
            p_prime_cell = p[cell]
            phi_f = BCs[name].get_face_pressure_correction(p_prime_cell)

        else:
            raise ValueError(f"Unknown field: {field}")

        # Gauss contributions ('n' points outward from cell_P, inward toward boundary)
        grad_phi_x[cell] += phi_f * n_x * face_length
        grad_phi_y[cell] += phi_f * n_y * face_length

    # divide by cell area (integral -> cell-average)
    grad_phi_x /= mesh.cell_areas
    grad_phi_y /= mesh.cell_areas

    return grad_phi_x, grad_phi_y