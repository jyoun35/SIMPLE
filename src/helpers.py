import os
import numpy as np
import matplotlib.pyplot as plt

from matplotlib.collections import PolyCollection
from dataclasses import fields

from src.numerics import compute_grad, linear_interp, rhie_and_chow_interp


# ============================================================ #
#                            PLOTS                             #
# ============================================================ #

def plot_mesh(mesh):
    """
    Plot the mesh with labels for boundaries.
    """

    # initialize plot
    fig, ax = plt.subplots(figsize=(8,6))

    # plot the nodes of each of the quad cells
    for cell in mesh.cells:
        pts = mesh.node_coords[cell]
        pts = np.vstack([pts, pts[0]])
        ax.plot(pts[:,0], pts[:,1], 'k-', linewidth=0.5)

    # plot and label the boundary faces
    if mesh.boundary_faces:

        # extract boundary name
        boundary_name = list(set(boundary_face[3] for boundary_face in mesh.boundary_faces))

        # assign colors to boundaries
        color_cycle = ['red','blue','green','orange','purple','cyan','magenta']
        colour_map = {name: color_cycle[i % len(color_cycle)] for i, name in enumerate(boundary_name)}

        # plot each boundary face
        for face in mesh.boundary_faces:
            node_1, node_2, _, name = face
            pts = mesh.node_coords[[node_1, node_2]]
            ax.plot(pts[:,0], pts[:,1], color=colour_map.get(name, 'black'), linewidth=2.5, label=name)

        # remove duplicate legend labels
        handles, labels = ax.get_legend_handles_labels()
        unique_labels = dict(zip(labels, handles))

        # show unique legend labels
        if unique_labels: ax.legend(unique_labels.values(), unique_labels.keys())

    # set up the plot
    ax.set_aspect('equal')
    ax.set_xlabel('x'); ax.set_ylabel('y')
    ax.set_title(f'Mesh: {mesh.filename}')
    plt.tight_layout()
    plt.show()


def plot_soln(mesh, solution, title="Solution"):
    """
    Plot each field in the solution dataclass as a filled contour.
    """

    quads = [mesh.node_coords[cell] for cell in mesh.cells]

    for field in fields(solution):

        name   = field.name
        values = getattr(solution, name)

        fig, ax = plt.subplots(figsize=(7, 5))

        collection = PolyCollection(
            quads, array=values,
            cmap="coolwarm", edgecolors="k", linewidths=0.2,
        )
        collection.set_clim(np.min(values), np.max(values))

        ax.add_collection(collection)
        ax.autoscale()
        ax.set_title(f"{title} - {name}")
        ax.set_aspect("equal")
        fig.colorbar(collection, ax=ax)

    plt.show()


def plot_continuity_residual(mesh, residual):
    """
    Plot the cell-centered continuity residual by cell.
    """

    quads = [mesh.node_coords[cell] for cell in mesh.cells]

    fig, ax = plt.subplots(figsize=(7, 5))

    collection = PolyCollection(
        quads, array=residual,
        cmap="RdBu_r", edgecolors="k", linewidths=0.2,
    )
    collection.set_clim(np.min(residual), np.max(residual))

    ax.add_collection(collection)
    ax.autoscale()
    ax.set_title("Continuity Residual by Cell")
    ax.set_aspect("equal")

    cbar = fig.colorbar(collection, ax=ax)
    cbar.set_label(r"$R_P = \sum_f \rho \mathbf{u}_f \cdot \mathbf{S}_f$")

    plt.show()


# ============================================================ #
#                           XFOIL                              #
# ============================================================ #

def read_XFOIL_data(xfoil_file):
    """
    Read an XFOIL polar/CP file then return dictionary {'x':..., 'Cp':...}.
    """

    data = np.loadtxt(xfoil_file, comments="#", skiprows=2)

    x  = data[:, 0]
    Cp = data[:, 2]

    return {"x": x, "Cp": Cp}


def plot_pressure_distribution(mesh, p, raw_BCs, xfoil_data, c=1.0):
    """
    Plot the pressure-coefficient distribution around the airfoil and overlay XFOIL data.
    """

    # freestream conditions from "raw" (i.e. dictionary) BCs
    rho   = 1.0 # fixed for now...
    p_inf = raw_BCs["pressure-outlet"]["p"]
    u_inf = raw_BCs["velocity-inlet"]["u"]
    v_inf = raw_BCs["velocity-inlet"]["v"]
    V_inf = np.sqrt(u_inf**2 + v_inf**2)
    q_inf = 0.5 * rho * V_inf**2

    # collect data from the surface (i.e. slip-wall) of the airfoil
    airfoil_faces = [f for f in mesh.boundary_faces if f[3] == "slip-wall"]

    x_surface = []
    y_surface = []
    p_surface = []

    for node_1, node_2, cell, _ in airfoil_faces:
        fc = (mesh.node_coords[node_1] + mesh.node_coords[node_2]) / 2
        x_surface.append(fc[0])
        y_surface.append(fc[1])
        p_surface.append(p[cell])

    x_surface = np.asarray(x_surface)
    y_surface = np.asarray(y_surface)
    p_surface = np.asarray(p_surface)

    # compute the pressure coefficient and normalized position
    Cp      = (p_surface - p_inf) / q_inf
    x_over_c = x_surface / c

    # split the upper/lower surface
    upper = y_surface > 0
    lower = y_surface < 0

    upper_ordered = np.argsort(x_over_c[upper])
    lower_ordered = np.argsort(x_over_c[lower])

    # plot the upper, lower, and XFOIL data
    fig, ax = plt.subplots(figsize=(5.5, 6.5))

    ax.plot(x_over_c[upper][upper_ordered], Cp[upper][upper_ordered], "-o", markersize=3, label="Upper")
    ax.plot(x_over_c[lower][lower_ordered], Cp[lower][lower_ordered], "-o", markersize=3, label="Lower")
    ax.plot(xfoil_data["x"], xfoil_data["Cp"], "--", markersize=3, label="XFOIL Data")

    ax.invert_yaxis()
    ax.set_xlabel(r"$x/c$")
    ax.set_ylabel(r"$C_p$")
    ax.set_title("Airfoil Pressure Coefficient Distribution")
    ax.grid(True)
    ax.legend()

    return fig


# ============================================================ #
#                         CHECKPOINTS                          #
# ============================================================ #

def save_checkpoint(SAVE_DIR, iter, mesh, soln, residual, residual_hist, raw_BCs, xfoil_data):
    """
    Save .npz + solution contour + Cp plot for the current iteration.
    """

    os.makedirs(SAVE_DIR, exist_ok=True)

    # save current solution as .npz
    np.savez(
        os.path.join(SAVE_DIR, f"solution_iter_{iter:06d}.npz"),
        u=soln.u, v=soln.v, p=soln.p,
    )

    # save results 

    # compute quantities to plot (velocity magnitude + gauge pressure)
    V_mag  = np.sqrt(soln.u**2 + soln.v**2)
    p_diff = soln.p - raw_BCs["pressure-outlet"]["p"]

    # set up the subplot
    quads = [mesh.node_coords[cell] for cell in mesh.cells]

    xlim = (-0.1, 1.1)
    ylim = (-0.3, 0.3)
    Vmin, Vmax = 0.0, 45.0
    pmin, pmax = 500.0, -500.0
    rmin, rmax = -0.0025, 0.0025

    fig_soln, axes = plt.subplots(2, 1, figsize=(9, 8))

    # plot velocity magnitude contour
    c1 = PolyCollection(quads, array=V_mag, cmap="coolwarm", edgecolors="none")
    c1.set_clim(Vmin, Vmax)

    axes[0].add_collection(c1)
    axes[0].set_xlim(*xlim); axes[0].set_ylim(*ylim)
    axes[0].set_aspect("equal")
    axes[0].set_title(f"Velocity Magnitude - Iteration {iter}")
    axes[0].set_xlabel("x"); axes[0].set_ylabel("y")

    fig_soln.colorbar(c1, ax=axes[0], label="Velocity Magnitude [m/s]")

    # plot pressure difference contour
    c2 = PolyCollection(quads, array=p_diff, cmap="viridis", edgecolors="none")
    c2.set_clim(pmin, pmax)

    axes[1].add_collection(c2)
    axes[1].set_xlim(*xlim); axes[1].set_ylim(*ylim)
    axes[1].set_aspect("equal")
    axes[1].set_title(f"Pressure Relative to Freestream - Iteration {iter}")
    axes[1].set_xlabel("x"); axes[1].set_ylabel("y")

    fig_soln.colorbar(c2, ax=axes[1], label=r"$p-p_\infty$ [Pa]")

    # save velocity magnitude + pressure difference contour plot
    plt.tight_layout()
    fig_soln.savefig(
        os.path.join(SAVE_DIR, f"soln_contours_iter_{iter:06d}.png"),
        dpi=300, bbox_inches="tight",
    )
    plt.close(fig_soln)

    # plot + save continuity residual field
    fig_residual, ax_residual = plt.subplots(figsize=(7, 4))

    c3 = PolyCollection(quads, array=residual, cmap="RdBu_r", edgecolors="black", linewidths=0.05)
    c3.set_clim(rmin, rmax)

    ax_residual.add_collection(c3)
    ax_residual.set_aspect("equal")
    ax_residual.set_title(f"Continuity Residual $\\mathbf{{WITH}}$ Rhie-Chow Interpolation \nIteration: {iter}")
    ax_residual.set_xlabel("x")
    ax_residual.set_ylabel("y")

    fig_residual.colorbar(c3, ax=ax_residual, label=r"$R_P = \sum_f \rho \mathbf{u}_f \cdot \mathbf{S}_f$")

    plt.tight_layout()
    fig_residual.savefig(
        os.path.join(SAVE_DIR, f"continuity_residual_iter_{iter:06d}.png"),
        dpi=300, bbox_inches="tight",
    )
    plt.close(fig_residual)

    # plot + save continuity residual history
    fig_hist, ax_hist = plt.subplots(figsize=(9, 4))

    ax_hist.plot(
        residual_hist[:, 0],
        residual_hist[:, 2],
    )

    ax_hist.set_yscale("log")
    ax_hist.set_xlabel("Iteration")
    ax_hist.set_ylabel("L1 Norm")
    ax_hist.set_title("L1 Norm of Continuity Residual")
    ax_hist.grid(True)

    plt.tight_layout()
    fig_hist.savefig(
        os.path.join(SAVE_DIR, f"continuity_residual_history_iter_{iter:06d}.png"),
        dpi=300,
        bbox_inches="tight",
    )
    plt.close(fig_hist)

    # save continuity residual history to CSV
    np.savetxt(
        os.path.join(SAVE_DIR, f"continuity_residual_history.csv"),
        residual_hist,
        delimiter=",",
        header="Iteration, Max, L1_Norm, Net",
        comments="",
    )

    # plot + save pressure coefficient plot
    fig_Cp = plot_pressure_distribution(mesh, soln.p, raw_BCs, xfoil_data, c=1.0)
    plt.tight_layout()
    fig_Cp.savefig(
        os.path.join(SAVE_DIR, f"Cp_plot_iter_{iter:06d}.png"),
        dpi=300, bbox_inches="tight",
    )
    plt.close(fig_Cp)


    print(f"Saved checkpoint at iteration {iter}\n")


# ============================================================ #
#                    CONTINUITY RESIDUAL                       #
# ============================================================ #

def continuity_residual(mesh, soln, A_u_inv, A_v_inv, BCs, rho=1.0):
    """
    Cell-centered continuity residual using the Rhie–Chow interpolated face velocity.
    
    Note: Rhie-Chow needs to be used since that it is what is used in the SIMPLE algorithm. 
    You can have the same cell-centered values but different face values depending on your 
    choice of interpolation.
    """

    residual = np.zeros(mesh.n_cells)

    u = soln.u
    v = soln.v
    p = soln.p

    # cell-centered pressure gradient (node-based)
    grad_p_x, grad_p_y = compute_grad(mesh, None, None, p, BCs, field="p")

    # ================== INTERNAL FACES ================== #
    for node_1, node_2, cell_P, cell_N in mesh.internal_faces:

        key = (node_1, node_2, cell_P, cell_N)
        n_x, n_y      = mesh.face_normals[key]
        S             = mesh.face_lengths[key]
        face_centroid = mesh.face_centroids[key]

        d_Pf = np.linalg.norm(face_centroid - mesh.cell_centroids[cell_P])
        d_Nf = np.linalg.norm(face_centroid - mesh.cell_centroids[cell_N])

        Au_f = linear_interp(A_u_inv[cell_P], A_u_inv[cell_N], d_Pf, d_Nf)
        Av_f = linear_interp(A_v_inv[cell_P], A_v_inv[cell_N], d_Pf, d_Nf)

        u_f, v_f = rhie_and_chow_interp(
            mesh,
            cell_P, cell_N,
            n_x, n_y,
            d_Pf, d_Nf,
            u[cell_P], u[cell_N],
            v[cell_P], v[cell_N],
            p[cell_P], p[cell_N],
            grad_p_x[cell_P], grad_p_x[cell_N],
            grad_p_y[cell_P], grad_p_y[cell_N],
            Au_f, Av_f,
        )

        flux = rho * (u_f * n_x + v_f * n_y) * S
        residual[cell_P] += flux
        residual[cell_N] -= flux


    # ================== BOUNDARY FACES ================== #
    for node_1, node_2, cell, name in mesh.boundary_faces:

        key = (node_1, node_2, cell, name)
        n_x, n_y = mesh.face_normals[key]
        S        = mesh.face_lengths[key]

        u_f, v_f = BCs[name].get_face_velocity(u[cell], v[cell], n_x, n_y)

        flux = rho * (u_f * n_x + v_f * n_y) * S

        residual[cell] += flux

    # store the residual history
    max_residual = np.max(np.abs(residual))
    L1_residual = np.sum(np.abs(residual))
    net_residual = np.sum(residual)

    results = [max_residual, L1_residual, net_residual]

    # print the continuity residual statistics
    print(
        f"Continuity Residual:\n",
        f"max = {max_residual}\n",
        f"L1  = {L1_residual}\n",
        f"net = {net_residual}\n",
    )

    return residual, results