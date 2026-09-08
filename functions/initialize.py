from functions.helpers import plot_soln

import numpy as np

def initialize_soln(ICs, mesh, plot=False):
    '''
    Initialize u, v, and p based on INITIAL_CONDITIONS
    '''
    n_cells = mesh.n_cells

    u = np.full(n_cells, ICs["u"])
    v = np.full(n_cells, ICs["v"])
    p = np.full(n_cells, ICs["p"])

    if plot: plot_soln(mesh, u, v, p, title="Initial Solution")

    return u, v, p

def load_soln(file, plot=False):
    '''
    Initialize u, v, and p based on saved solution (.npz file format)
    '''
    data = np.load(file)

    u = data["u"]
    v = data["v"]
    p = data["p"] 

    return u, v, p