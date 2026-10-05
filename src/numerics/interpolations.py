import numpy as np

def linear_interp(phi_P, phi_N, d_P_to_f, d_N_to_f):
    """
    Linearly interpolate for the face value based on cell centers.
    """

    phi_f = (d_N_to_f * phi_P + d_P_to_f * phi_N) / (d_P_to_f + d_N_to_f) 

    return phi_f


def rhie_and_chow_interp(mesh, 
                         cell_P, cell_N,
                         n_x, n_y,
                         d_P_to_f, d_N_to_f,
                         u_P, u_N, 
                         v_P, v_N, 
                         p_P, p_N,
                         grad_p_x_P, grad_p_x_N,
                         grad_p_y_P, grad_p_y_N,
                         A_u_inv_face, A_v_inv_face):
    """
    Rhie and Chow interpolation for face velocity based on cell-centered values.
    This method is used to avoid odd-even decoupling in collocated grids.
    """

    # linear interpolation for face velocity
    u_f_linear_interp = linear_interp(u_P, u_N, d_P_to_f, d_N_to_f)
    v_f_linear_interp = linear_interp(v_P, v_N, d_P_to_f, d_N_to_f)

    # linear inteprolation for face pressure gradient
    grad_p_x_f_interp = linear_interp(grad_p_x_P, grad_p_x_N, d_P_to_f, d_N_to_f)
    grad_p_y_f_interp = linear_interp(grad_p_y_P, grad_p_y_N, d_P_to_f, d_N_to_f)

    # distance and directions between current and neighbor cell
    d_P_to_N = mesh.cell_centroids[cell_N] - mesh.cell_centroids[cell_P]
    d_P_to_N_mag = np.linalg.norm(d_P_to_N)

    d_P_to_N_x = d_P_to_N[0]
    d_P_to_N_y = d_P_to_N[1]

    d_normal = (d_P_to_N_x*n_x) + (d_P_to_N_y*n_y) # dot product between d_P_to_N and normal

    # compute interpolated pressure gradient
    grad_p_P_to_N_interp = (
                            grad_p_x_f_interp * (d_P_to_N_x / d_P_to_N_mag) 
                            + grad_p_y_f_interp * (d_P_to_N_y / d_P_to_N_mag)
                           )

    # compute direct pressure gradient
    grad_p_P_to_N_direct = (p_N - p_P) / d_normal

    # apply rhie and chow correction
    correction = grad_p_P_to_N_direct - grad_p_P_to_N_interp

    u_f = u_f_linear_interp - (A_u_inv_face * correction * n_x)
    v_f = v_f_linear_interp - (A_v_inv_face * correction * n_y)

    # return rhie and chow interpolated face value
    return u_f, v_f
