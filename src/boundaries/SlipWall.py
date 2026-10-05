from .boundaries import BoundaryCondition
from ..numerics import non_ortho_corr_flux

class SlipWall(BoundaryCondition):
    """No-penetration (u·n = 0), slip (∂u/∂n = 0) wall boundary condition."""

    # ---------------- Getters ---------------- #
    def get_face_velocity(self, u_cell, v_cell, n_x, n_y):

        # remove the normal component of the velocity at the face
        u_normal = u_cell * n_x + v_cell * n_y
        u_f = u_cell - u_normal * n_x
        v_f = v_cell - u_normal * n_y
        
        return u_f, v_f

    def get_face_pressure(self, p_cell, n_x, n_y):
        return p_cell  # zero-gradient pressure

    def get_face_pressure_correction(self, p_prime_cell):
        return p_prime_cell  # zero-gradient pressure correction

    # ---------------- Convective ---------------- #
    def convective_face_flux(self, u_cell, v_cell, n_x, n_y, face_length):
        
        # u·n = 0  =>  no convective flux
        return 0.0, 0.0, 0.0, 0.0

    # ---------------- Diffusive ---------------- #
    def diffusive_face_flux(self,
                            mu,
                            u_cell, v_cell,
                            n_x, n_y,
                            face_centroid, face_length,
                            cell_centroid_P,
                            grad_u_x_P, grad_u_y_P,
                            grad_v_x_P, grad_v_y_P):

        # implicit orthogonal coefficient + explicit non-orthgonal correction
        coeff_ortho_u, explicit_corr_u = non_ortho_corr_flux(
            n_x, n_y,
            face_centroid, face_length,
            cell_centroid_P, face_centroid,
            grad_u_x_P, grad_u_x_P,
            grad_u_y_P, grad_u_y_P,
        )

        coeff_ortho_v, explicit_corr_v = non_ortho_corr_flux(
            n_x, n_y,
            face_centroid, face_length,
            cell_centroid_P, face_centroid,
            grad_v_x_P, grad_v_x_P,
            grad_v_y_P, grad_v_y_P,
        )

        # u-momentum: implicit coeff n_x² on u_P, explicit coupling to v_P (since u_P is the unknown)
        M_u =  mu * coeff_ortho_u * n_x**2
        b_u = -mu * coeff_ortho_u * n_x * n_y * v_cell + (mu * explicit_corr_u)

        # v-momentum: implicit coeff n_y² on v_P, explicit coupling to u_P (since v_P is the unknown)
        M_v =  mu * coeff_ortho_v * n_y**2
        b_v = -mu * coeff_ortho_v * n_x * n_y * u_cell + (mu * explicit_corr_v)

        # NOTE: for u-momentum u_P is the unknown, so explicit coupling to v_P is added to the source term
        #       since we need both to get face velocity (vice versa for v-momentum)

        return M_u, M_v, b_u, b_v

    # ---------------- Pressure-Correction ---------------- #
    def pressure_correction_face_flux(self, A_u_inv, A_v_inv, n_x, n_y, d_P_to_f, face_length):
        
        # zero-gradient p' at the wall -> no contribution
        return 0.0

    def intermediate_velocity_face_flux(self, U_star_u_P, U_star_v_P, n_x, n_y, face_length):
        
        # u·n = 0 -> no flux
        return 0.0