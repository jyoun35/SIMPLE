from abc import ABC, abstractmethod


class BoundaryCondition(ABC):
    """
    Abstract base class for all boundary conditions.
    """

    @abstractmethod
    def get_face_velocity(self, u_cell, v_cell, n_x, n_y) -> tuple[float, float]:
        """
        Return the boundary-face velocity (u_f, v_f).
        """

    @abstractmethod
    def get_face_pressure(self, p_cell, n_x, n_y) -> float:
        """
        Return the boundary-face pressure value p_f.
        """

    @abstractmethod
    def get_face_pressure_correction(self, p_prime_cell) -> float:
        """
        Return the boundary-face pressure-correction value p'_f.
        """

    @abstractmethod
    def convective_face_flux(self, u_cell, v_cell, n_x, n_y, face_length):
        """
        Return the Convective flux contributions for a faceto the momentum operator and/or source term.
        """

    @abstractmethod
    def diffusive_face_flux(self,
                            mu,
                            u_cell, v_cell,
                            n_x, n_y,
                            face_centroid, face_length,
                            cell_centroid_P,
                            grad_u_x_P, grad_u_y_P,
                            grad_v_x_P, grad_v_y_P):
        """
        Return the diffusive flux contributions for a face to the momentum operator and/or source term.
        """

    @abstractmethod
    def pressure_correction_face_flux(self, A_u_inv, A_v_inv, n_x, n_y, d_P_to_f, face_length) -> float:
        """
        Return the contribution to the pressure-correction equation matrix for a face.
        """

    @abstractmethod
    def intermediate_velocity_face_flux(self, U_star_u_P, U_star_v_P, n_x, n_y, face_length) -> float:
        """
        Return the intermediate-velocity flux contribution to the RHS of the pressure Poisson equation for a face.
        """


def process_boundaries(BCs):
    """
    Build a dict of boundary-condition objects keyed by boundary name.

    Supported names: 
    - velocity-inlet
    - pressure-outlet
    - slip-wall
    """
    
    boundaries = {}

    # loop through BCs and create the appropriate boundary-condition object for each
    for name, details in BCs.items():
        if name == "velocity-inlet":
            boundaries[name] = VelocityInlet(u=details["u"], v=details["v"])
        elif name == "pressure-outlet":
            boundaries[name] = PressureOutlet(p=details["p"])
        elif name == "slip-wall":
            boundaries[name] = SlipWall()
        else:
            raise ValueError(f"Unknown or unimplemented boundary condition: {name}")

    return boundaries


# import boundary conditons at bottom due to circular import issues
from .VelocityInlet import VelocityInlet   
from .PressureOutlet import PressureOutlet
from .SlipWall import SlipWall         