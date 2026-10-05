from .boundaries import BoundaryCondition


class PressureOutlet(BoundaryCondition):
    """Constant-pressure outlet boundary condition."""

    # ---------------- Initialize ---------------- #
    def __init__(self, p):
        self.p = p

    # ---------------- Getters ---------------- #
    def get_face_velocity(self, u_cell, v_cell, n_x, n_y):
        return u_cell, v_cell  # zero-gradient velocity

    def get_face_pressure(self, p_cell, n_x, n_y):
        return self.p # specified outlet pressure

    def get_face_pressure_correction(self, p_prime_cell):
        return 0.0  # p_f' = 0 at the outlet

    # ---------------- Convective ---------------- #
    def convective_face_flux(self, u_cell, v_cell, n_x, n_y, face_length):
        u_f, v_f = u_cell, v_cell
        m_dot = u_f * n_x + v_f * n_y

        if m_dot >= 0.0:
            # outflow: upwind on interior value, implicit
            M_u = m_dot * face_length
            M_v = m_dot * face_length
            return M_u, M_v, 0.0, 0.0
        else:
            # backflow: upwind on interior value, explicit
            b_u = -m_dot * u_cell * face_length
            b_v = -m_dot * v_cell * face_length
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
        
        return 0.0, 0.0, 0.0, 0.0 # zero-gradient velocity -> no diffusive flux

    # ---------------- Pressure-Correction ---------------- #
    def pressure_correction_face_flux(self, A_u_inv, A_v_inv, n_x, n_y, d_P_to_f, face_length):

        # Dirichlet p_f' = 0 -> diagonal contribution is -D (only from interior)
        D = (face_length / d_P_to_f) * (A_u_inv * n_x**2 + A_v_inv * n_y**2)
        
        return -D

    def intermediate_velocity_face_flux(self, U_star_u_P, U_star_v_P, n_x, n_y, face_length):
        
        # zero-gradient velocity at the face: use cell values
        m_dot_star = U_star_u_P * n_x + U_star_v_P * n_y
        flux = m_dot_star * face_length

        return flux