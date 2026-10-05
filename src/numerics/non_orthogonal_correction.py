import numpy as np

from .interpolations import linear_interp


def non_ortho_corr_flux(
                   n_x, n_y,
                   face_centroid, face_length,
                   cell_centroid_P, cell_centroid_N,
                   grad_phi_x_P, grad_phi_x_N,
                   grad_phi_y_P, grad_phi_y_N
                  ):
    """
    Split the face-normal diffusion flux into an implicit orthogonal part
    and an explicit non-orthogonal correction.
    """

    # center-to-center vector
    d_P_to_N_vec = cell_centroid_N - cell_centroid_P
    d_P_to_N_mag = np.linalg.norm(d_P_to_N_vec)

    # face-to-cell distances (for linear interpolation)
    d_P_to_f = np.linalg.norm(face_centroid - cell_centroid_P)
    d_N_to_f = np.linalg.norm(face_centroid - cell_centroid_N)

    # orthogonal / non-orthogonal decomposition of the face normal (using over-relaxed approach)
    n_f_mag_sq = n_x**2 + n_y**2
    n_dot_d    = n_x * d_P_to_N_vec[0] + n_y * d_P_to_N_vec[1]

    n_ortho_x = d_P_to_N_vec[0] * (n_f_mag_sq / n_dot_d)
    n_ortho_y = d_P_to_N_vec[1] * (n_f_mag_sq / n_dot_d)

    n_non_ortho_x = n_x - n_ortho_x
    n_non_ortho_y = n_y - n_ortho_y

    # ---- implicit orthogonal coefficient ---- #
    d_hat_x = d_P_to_N_vec[0] / d_P_to_N_mag
    d_hat_y = d_P_to_N_vec[1] / d_P_to_N_mag

    coeff_ortho = (face_length / d_P_to_N_mag) * (
        n_ortho_x * d_hat_x + n_ortho_y * d_hat_y
    )

    # ---- explicit non-orthogonal correction ---- #
    grad_phi_x_face = linear_interp(grad_phi_x_P, grad_phi_x_N, d_P_to_f, d_N_to_f)
    grad_phi_y_face = linear_interp(grad_phi_y_P, grad_phi_y_N, d_P_to_f, d_N_to_f)

    explicit_non_ortho = face_length * (
        grad_phi_x_face * n_non_ortho_x
        + grad_phi_y_face * n_non_ortho_y
    )

    # return implicit orthogonal coefficient and explicit and non-orthogonal correction
    return coeff_ortho, explicit_non_ortho


def non_ortho_corr_flux_p_prime(
                   n_x, n_y,
                   face_centroid, face_length,
                   cell_centroid_P, cell_centroid_N,
                   grad_pp_x_P, grad_pp_x_N,
                   grad_pp_y_P, grad_pp_y_N,
                   A_u_inv_face, A_v_inv_face
                  ):
    """
    Explicit non-orthogonal correction for the pressure-correction flux.

    Different than 'non_ortho_corr_flux' because the pressure-correction flux is 
    weighted by the inverse of the anisotropic A matrix.

    The implicit orthogonal part is already handled by the D coefficient
    inside `pressure_correction`. Only the explicit correction is returned
    here with the anisotropic A⁻¹ accounted for.
    """

    # center-to-center vector
    d_P_to_N_vec = cell_centroid_N - cell_centroid_P

    # face-to-cell distances (for linear interpolation)
    d_P_to_f = np.linalg.norm(face_centroid - cell_centroid_P)
    d_N_to_f = np.linalg.norm(face_centroid - cell_centroid_N)

    # non-orthogonal decomposition of the face normal (using over-relaxed approach)
    n_f_mag_sq = n_x**2 + n_y**2
    n_dot_d    = n_x * d_P_to_N_vec[0] + n_y * d_P_to_N_vec[1]

    n_ortho_x = d_P_to_N_vec[0] * (n_f_mag_sq / n_dot_d)
    n_ortho_y = d_P_to_N_vec[1] * (n_f_mag_sq / n_dot_d)

    n_non_ortho_x = n_x - n_ortho_x
    n_non_ortho_y = n_y - n_ortho_y

    # interpolate ∇p' to the face 
    grad_pp_x_face = linear_interp(grad_pp_x_P, grad_pp_x_N, d_P_to_f, d_N_to_f)
    grad_pp_y_face = linear_interp(grad_pp_y_P, grad_pp_y_N, d_P_to_f, d_N_to_f)

    # explicit correction --> anisotropic A⁻¹-weighted non-orthogonal flux
    flux_non = face_length * (
        A_u_inv_face * grad_pp_x_face * n_non_ortho_x
        + A_v_inv_face * grad_pp_y_face * n_non_ortho_y
    )

    return flux_non