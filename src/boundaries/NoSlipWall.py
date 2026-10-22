from .boundaries import BoundaryCondition
from ..numerics import non_ortho_corr_flux

class NoSlipWall(BoundaryCondition):
    """No-penetration (U·n = 0), and no-slip (u = 0, v = 0) wall boundary condition."""

    # ---------------- Getters ---------------- #
    def get_face_velocity(self, u_cell, v_cell, n_x, n_y):

        u_f = 0.0 # no-slip condition
        v_f = 0.0 # no-slip condition
        
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

        # RHS u_f term (wall contribution) vanishes since u_f=0 -> gradient is (u_P - 0) / d_pf
        M_u = mu * coeff_ortho_u
        b_u = mu * explicit_corr_u

        M_v = mu * coeff_ortho_v
        b_v = mu * explicit_corr_v

        return M_u, M_v, b_u, b_v

    # ---------------- Pressure-Correction ---------------- #
    def pressure_correction_face_flux(self, A_u_inv, A_v_inv, n_x, n_y, d_P_to_f, face_length):
        
        # zero-gradient p' at the wall -> no contribution
        return 0.0

    def intermediate_velocity_face_flux(self, U_star_u_P, U_star_v_P, n_x, n_y, face_length):
        
        # u·n = 0 -> no flux
        return 0.0