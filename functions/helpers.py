import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import PolyCollection

def plot_soln(mesh, u, v, p, title="Solution"):
    """
    Plot the cell-centered scalar fields u, v, p on the mesh.
    """

    # create quads for each cell based on their nodes
    quads = [mesh.node_coords[cell] for cell in mesh.cells]

    # plot for u-velocity
    fig, ax = plt.subplots(figsize=(7, 5))

    collection = PolyCollection(quads, array=u, cmap="coolwarm", edgecolors="k", linewidths=0.2)
    collection.set_clim(np.min(u), np.max(u))

    ax.add_collection(collection)
    ax.autoscale()
    ax.set_title(f"{title} - u velocity")
    ax.set_aspect("equal")
    fig.colorbar(collection, ax=ax)

    # plot for v-velocity
    fig, ax = plt.subplots(figsize=(7, 5))

    collection = PolyCollection(quads, array=v, cmap="coolwarm", edgecolors="k", linewidths=0.2)
    collection.set_clim(np.min(v), np.max(v))

    ax.add_collection(collection)
    ax.autoscale()
    ax.set_title(f"{title} - v velocity")
    ax.set_aspect("equal")
    fig.colorbar(collection, ax=ax)

    # plot for pressure
    fig, ax = plt.subplots(figsize=(7, 5))

    collection = PolyCollection(quads, array=p, cmap="viridis", edgecolors="k", linewidths=0.2)
    collection.set_clim(np.min(p), np.max(p))

    ax.add_collection(collection)
    ax.autoscale()
    ax.set_title(f"{title} - Pressure")
    ax.set_aspect("equal")
    fig.colorbar(collection, ax=ax)

    # show plots
    plt.show()


def read_XFOIL_data(xfoil_data):

    # load the XFOIL data
    data = np.loadtxt(xfoil_data, comments="#", skiprows=2)

    # extract the x and Cp values
    x = data[:, 0]
    Cp = data[:, 2]

    # return x and Cp
    return x, Cp


def plot_pressure_distribution(mesh, p, BCs, x_xfoil, Cp_xfoil, c=1.0):
    '''
    Plot pressure coefficient distribution
    '''

    # extract freestream conditions
    rho = 1.0
    p_inf = BCs["outlet p"]
    u = BCs["inlet u"]
    v = BCs["inlet v"]
    V_inf = np.sqrt(u**2 + v**2)
    q_inf = 0.5 * rho * V_inf**2

    # Get all boundary faces belonging to the airfoil
    airfoil_faces = [face for face in mesh.boundary_faces if face[3] == "slip wall"]

    # collect x, y, and p for the airfoil surface
    x_surface = []
    y_surface = []
    p_surface = []

    for node_1, node_2, cell, boundary_name in airfoil_faces:

        x1 = mesh.node_coords[node_1]
        x2 = mesh.node_coords[node_2]

        face_center = 0.5 * (x1 + x2)

        x_surface.append(face_center[0])
        y_surface.append(face_center[1])

        p_surface.append(p[cell])

    x_surface = np.array(x_surface)
    y_surface = np.array(y_surface)
    p_surface = np.array(p_surface)

    # calculate pressure coefficient (Cp)
    Cp = (p_surface - p_inf) / q_inf

    # normalize x location by chord
    x_over_c = x_surface / c

    # seperate upper and lower surfaces
    upper = y_surface > 0
    lower = y_surface < 0

    # sort by x/c
    upper_order = np.argsort(x_over_c[upper])
    lower_order = np.argsort(x_over_c[lower])

    # plot the upper and lower pressure coefficient distribution
    fig_Cp, ax = plt.subplots(figsize=(5.5, 6.5))

    ax.plot(x_over_c[upper][upper_order], Cp[upper][upper_order], "-o", markersize=3, label="Upper")
    ax.plot(x_over_c[lower][lower_order], Cp[lower][lower_order], "-o", markersize=3, label="Lower")

    # plot the XFOIL pressure coefficient distribution
    ax.plot(x_xfoil, Cp_xfoil, "--", markersize=3, label="XFOIL Data")

    # set plot format
    ax.invert_yaxis()
    ax.set_xlabel(r"$x/c$")
    ax.set_ylabel(r"$C_p$")
    ax.set_title("Airfoil Pressure Coefficient Distribution")
    ax.grid(True)
    ax.legend()

    # # show plot
    # plt.tight_layout()
    # plt.show()

    # return figure handle
    return fig_Cp


def save_checkpoint(SAVE_DIR, iter, mesh, u, v, p, BCs, x_xfoil, Cp_xfoil):
    '''
    Save the solution matrices + contour plots + Cp plot
    '''

    # make directory to save results
    os.makedirs(SAVE_DIR, exist_ok=True)

    # save solution as .npz (numPy zipped archive file)
    np.savez(
        os.path.join(SAVE_DIR, f"solution_iter_{iter:06d}.npz"),
        u=u,
        v=v,
        p=p
    )

    # quantities for plotting
    V_mag = np.sqrt(u**2 + v**2)
    p_diff = p - BCs["outlet p"]

    # create list of nodes for each cell
    quads = [mesh.node_coords[cell] for cell in mesh.cells]

    # set plot limits
    xlim = (-0.1, 1.1)
    ylim = (-0.3, 0.3)

    Vmin = 0.0
    Vmax = 45.0

    pmin = 500
    pmax = -500
    
    # create figure
    fig_soln, axes = plt.subplots(2, 1, figsize=(9, 8))

    # plot velocity magnitude
    collection = PolyCollection(quads, array=V_mag, cmap="coolwarm", edgecolors="none")
    collection.set_clim(Vmin, Vmax)

    # velocity magnitude plot details
    axes[0].add_collection(collection)
    axes[0].set_xlim(*xlim)
    axes[0].set_ylim(*ylim)
    axes[0].set_aspect("equal")
    axes[0].set_title(f"Velocity Magnitude - Iteration {iter}")
    axes[0].set_xlabel("x")
    axes[0].set_ylabel("y")
    fig_soln.colorbar(collection, ax=axes[0], label="Velocity Magnitude [m/s]")

    # plot pressure difference
    collection = PolyCollection(quads, array=p_diff, cmap="viridis", edgecolors="none")
    collection.set_clim(pmin, pmax)

    # pressure difference plot details
    axes[1].add_collection(collection)
    axes[1].set_xlim(*xlim)
    axes[1].set_ylim(*ylim)
    axes[1].set_aspect("equal")
    axes[1].set_title(f"Pressure Relative to Freestream - Iteration {iter}")
    axes[1].set_xlabel("x")
    axes[1].set_ylabel("y")
    fig_soln.colorbar(collection, ax=axes[1], label="$p-p_\\infty$ [Pa]")

    # save contour plot
    plt.tight_layout()
    fig_soln.savefig(
        os.path.join(SAVE_DIR, f"soln_contours_iter_{iter:06d}.png"),
        dpi=300,
        bbox_inches="tight"
    )
    plt.close(fig_soln)

    # generate pressure coefficient distribution plot
    fig_Cp = plot_pressure_distribution(mesh, p, BCs, x_xfoil, Cp_xfoil, c=1.0)

    # save contour plot
    plt.tight_layout()
    fig_Cp.savefig(
        os.path.join(SAVE_DIR, f"Cp_plot_iter_{iter:06d}.png"),
        dpi=300,
        bbox_inches="tight"
    )
    plt.close(fig_Cp)

    # print confirmation
    print(f"Saved checkpoint at iteration {iter}\n")


def continuity_residual(mesh, u, v, BCs, rho=1.0):
    '''
    Compute the continuity residual for current solution
    '''

    # initialize continuity residual vector
    residual = np.zeros(mesh.n_cells)


    # ================== INTERNAL FACES ================== #
    for internal_face in mesh.internal_faces:

        # extract face details
        node_1, node_2, cell_P, cell_N = internal_face
        key = (node_1, node_2, cell_P, cell_N)

        # extract face geometry
        n_x, n_y = mesh.face_normals[key]
        face_length = mesh.face_lengths[key]
        face_centroid = mesh.face_centroids[key]

        # compute distances between face and cells
        d_P_to_f = np.linalg.norm(face_centroid - mesh.cell_centroids[cell_P])
        d_N_to_f = np.linalg.norm(face_centroid - mesh.cell_centroids[cell_N])
        d_P_to_N = np.linalg.norm(mesh.cell_centroids[cell_N] - mesh.cell_centroids[cell_P])

        # linear interpolation for face value of velocity
        u_f = (d_N_to_f * u[cell_P] + d_P_to_f * u[cell_N]) / d_P_to_N
        v_f = (d_N_to_f * v[cell_P] + d_P_to_f * v[cell_N]) / d_P_to_N

        # outward flux from cell P
        flux = rho * (u_f * n_x + v_f * n_y) * face_length

        # add boundary flux contribution to residual -> cell P has outward flux
        residual[cell_P] += flux

        # add boundary flux contribution to residual -> cell N has inward flux
        residual[cell_N] -= flux


    # ================== BOUNDARY FACES ================== #
    for boundary_face in mesh.boundary_faces:

        # extract face details
        node_1, node_2, cell, name = boundary_face
        key = (node_1, node_2, cell, name)

        # extract face geometry
        n_x, n_y = mesh.face_normals[key]
        face_length = mesh.face_lengths[key]

        # ---------------- Inlet ---------------- #
        if name == "inlet":

            # prescribed velocity through outlet face
            u_f = BCs["inlet u"]
            v_f = BCs["inlet v"]

        # ---------------- Outlet ---------------- #
        elif name == "outlet": 

            # zero-gradient velocity (use cell centroid values)
            u_f = u[cell]
            v_f = v[cell]

        # ---------------- Slip Wall ---------------- #
        elif name == "slip wall":

            # get cell centroid values
            u_f = u[cell]
            v_f = v[cell]

            # remove component of velocity normal to the wall
            u_normal = u_f * n_x + v_f * n_y

            u_f = u_f - u_normal * n_x
            v_f = v_f - u_normal * n_y

        # compute boundary flux
        flux = rho * (u_f * n_x + v_f * n_y) * face_length

        # add boundary flux contribution to residual
        residual[cell] += flux

    # print residual metrics
    print(
        f"Continuity:\n",
        f"max = {np.max(np.abs(residual))}\n",
        f"L1 = {np.sum(np.abs(residual))}\n",
        f"net = {np.sum(residual)}\n"
    )

    # return residual vector
    return residual


def plot_continuity_residual(mesh, residual):
    '''
    Plot the continuity residual for each cell in a scatter plot
    '''

    # extract cell centroids
    x = mesh.cell_centroids[:, 0]
    y = mesh.cell_centroids[:, 1]

    # scatter plot for cointinuity residual
    fig, ax = plt.subplots(figsize=(10, 6))
    sc = ax.scatter(x, y, c=residual, cmap="RdBu_r", s=20)

    # format the scatter plot
    ax.set_aspect("equal")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title("Continuity Residual by Cell")

    cbar = plt.colorbar(sc, ax=ax)
    cbar.set_label(r"$R_P = \sum_f \rho \mathbf{u}_f \cdot \mathbf{S}_f$")

    # show plot
    plt.show()