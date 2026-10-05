from dataclasses import dataclass

from src.helpers import plot_soln

import numpy as np


@dataclass
class Solution:
    """Container for the cell-centered velocity and pressure fields."""
    u: np.ndarray
    v: np.ndarray
    p: np.ndarray


def initialize_soln(ICs, mesh, plot=False):
    """Initialize a uniform u, v, p field based on specified initial conditions."""
    n_cells = mesh.n_cells

    u = np.full(n_cells, ICs["u"])
    v = np.full(n_cells, ICs["v"])
    p = np.full(n_cells, ICs["p"])

    solution = Solution(
        u=u, 
        v=v, 
        p=p
    )

    # plot initial solution if specified
    if plot: plot_soln(mesh, solution, title="Initial Solution")

    return solution


def load_soln(file, mesh, plot=False):
    """Load a solution from a saved .npz checkpoint."""
    data = np.load(file)

    solution = Solution(
        u=data["u"],
        v=data["v"],
        p=data["p"],
    )

    # plot loaded solution if specified
    if plot: plot_soln(mesh, solution, title="Loaded Solution")

    return solution