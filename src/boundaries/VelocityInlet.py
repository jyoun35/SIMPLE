from .boundaries import BoundaryCondition
from ..numerics import non_ortho_corr_flux

class VelocityInlet(BoundaryCondition):
    """Constant-velocity inlet boundary condition."""

    # ---------------- Initialize ---------------- #
    def __init__(self, u, v):
        self.u = u
        self.v = v

    # ---------------- Getters ---------------- #
    def get_face_velocity(self, u_cell, v_cell, n_x, n_y):
        return self.u, self.v # specified inlet velocity

    def get_face_pressure(self, p_cell, n_x, n_y):
        return p_cell  # zero-gradient pressure

    def get_face_pressure_correction(self, p_prime_cell):
        return p_prime_cell  # zero-gradient pressure correction (to maintain consistency with zero-gradient pressure)

    # ---------------- Convective ---------------- #
    def convective_face_flux(self, u_cell, v_cell, n_x, n_y, face_length):
        u_f, v_f = self.u, self.v
        m_dot = u_f * n_x + v_f * n_y # mass flux at face

        # explicit convective contribution to source (based on prescribed velocity)
        b_u = -u_f * m_dot * face_length
        b_v = -v_f * m_dot * face_length

        return 0.0, 0.0, b_u, b_v

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
            cell_centroid_P, face_centroid,  # <- boundary "trick"
            grad_u_x_P, grad_u_x_P,          # <- pass cell center gradient twice
            grad_u_y_P, grad_u_y_P,
        )

        coeff_ortho_v, explicit_corr_v = non_ortho_corr_flux(
            n_x, n_y,
            face_centroid, face_length,
            cell_centroid_P, face_centroid,
            grad_v_x_P, grad_v_x_P,
            grad_v_y_P, grad_v_y_P,
        )

        # prescribed Dirichlet values
        u_f, v_f = self.u, self.v

        # u-momentum: implicit orthogonal component + explicit non-orthogonal correction
        M_u = mu * coeff_ortho_u
        b_u = mu * (coeff_ortho_u * u_f + explicit_corr_u)

        # v-momentum: implicit orthogonal component + explicit non-orthogonal correction
        M_v = mu * coeff_ortho_v
        b_v = mu * (coeff_ortho_v * v_f + explicit_corr_v)

        return M_u, M_v, b_u, b_v

    # ---------------- Pressure-Correction ---------------- #
    def pressure_correction_face_flux(self, A_u_inv, A_v_inv, n_x, n_y, d_P_to_f, face_length):
        
        # zero-gradient pressure -> no contribution
        return 0.0

    def intermediate_velocity_face_flux(self, U_star_u_P, U_star_v_P, n_x, n_y, face_length):

        # compute flux based on prescribed inlet velocity
        u_f, v_f = self.u, self.v
        m_dot_star = u_f * n_x + v_f * n_y
        flux = m_dot_star * face_length

        return flux