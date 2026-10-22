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


def plot_blasius(mesh, soln, x_eval, y_eval, U_infty, nu, blasius_file, y_wall=0.0, save_path=None):
    '''
    Plot Blasius vs numerical solution from solver for BL profile and Cf.

    Note: 
    * y-axis limits is explicitly set
    '''

    # load blasius table file
    blasius_table  = np.loadtxt(blasius_file, delimiter=",", skiprows=1)
    eta_blasius    = blasius_table[:, 0]
    fprime_blasius = blasius_table[:, 2]

    # extract cell centroids
    x_c = mesh.cell_centroids[:, 0]
    y_c = mesh.cell_centroids[:, 1]

    # initialize plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))


    # ---------- subplot 1: f' vs eta ---------- #    

    # x cell centroid to evaluate
    x_c_eval = x_c[np.argmin(np.abs(x_c - x_eval))]

    # find y values at x_c_eval
    mask = (np.abs(x_c - x_c_eval) < 1e-12) & (y_c <= y_eval) & (y_c >= y_wall)
    col_idx = np.where(mask)[0]
    col_idx = col_idx[np.argsort(y_c[col_idx])]
    y_profile = y_c[col_idx] - y_wall

    # get η and f' from solution
    eta_soln    = y_profile * np.sqrt(U_infty / (nu * x_c_eval))
    fprime_soln = soln.u[col_idx] / U_infty

    # plot η and f' from Blasius and from solution
    ax1.plot(fprime_blasius, eta_blasius, "k-", lw=2, label="Blasius")
    ax1.plot(fprime_soln, eta_soln, "o", color="tab:red", ms=4, label=f"Solution @ x = {x_c_eval:.4g} m")

    ax1.set_xlabel(r"$f' = u/U_\infty$", fontsize=12)
    ax1.set_ylabel(r"$\eta = y\,\sqrt{U_\infty/(\nu x)}$", fontsize=12)
    ax1.set_title("Boundary-Layer Profile", fontsize=16)
    ax1.grid(True, alpha=0.3)
    ax1.legend(loc="best", fontsize=14)


    # ---------- subplot 2: Cf vs x ---------- #

    # y cell centroid to evaluate
    y_c_eval = y_c[np.argmin(np.abs(y_c - y_wall))]

    # find x values at x_c_eval
    mask = (np.abs(y_c - y_c_eval) < 1e-12) & (x_c > 0.0) & (x_c < 0.025)
    col_idx  = np.where(mask)[0]
    col_idx  = col_idx[np.argsort(x_c[col_idx])]
    x_profile = x_c[col_idx]

    # compute ∂u/∂y at the wall
    y1 = y_c[col_idx] - y_wall
    u1 = soln.u[col_idx]
    dudy_wall = (u1 - 0) / y1

    # compute skin friciton coefficient (Cf) from solution and from Blasius
    Cf_soln = 2.0 * nu * dudy_wall / U_infty**2
    Cf_blasius = 0.664 * np.sqrt(nu / (U_infty * x_profile))

    # plot Cf from Blasius and from solution
    ax2.plot(x_profile, Cf_blasius, "k-", lw=2, label=r"Blasius  $0.664/\sqrt{Re_x}$")
    ax2.plot(x_profile, Cf_soln, "o", color="tab:red", ms=4, label="Solution")

    ax2.set_xlabel("x [m]", fontsize=12)
    ax2.set_ylabel(r"$C_f$", fontsize=12)
    ax2.set_title("Skin-Friction Coefficient", fontsize=16)
    ax2.grid(True, alpha=0.3)
    ax2.legend(loc="best", fontsize=14)

    ax2.set_ylim([0, 0.05]) # explicitly set for visualization purposes

    plt.tight_layout()

    # save figure if specified
    if save_path is not None:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close(fig)
    else:
        plt.show()


# ============================================================ #
#                         CHECKPOINTS                          #
# ============================================================ #

def save_checkpoint(SAVE_DIR, iter, mesh, soln, residual_hist, raw_BCs,
                    blasius_file, x_eval, y_eval, nu, y_wall=0.0):
    """
    Save:
    * solution file (.npz)
    * velocity-magnitude contour (.png)
    * continuity-residual history (.png + .csv)
    * Blasius comparison plot (.png)

    Note: 
    * x- and y-axis limits are explicitly set
    """

    os.makedirs(SAVE_DIR, exist_ok=True)

    # save solution as .npz
    np.savez(
        os.path.join(SAVE_DIR, f"solution_iter_{iter:06d}.npz"),
        u=soln.u, v=soln.v, p=soln.p,
    )

    # plot velocity magnitude contour
    V_mag = np.sqrt(soln.u**2 + soln.v**2)
    quads = [mesh.node_coords[cell] for cell in mesh.cells]

    xlim = (0, 0.200) # explicitly set for visualization
    ylim = (0, 0.002) # explicitly set for visualization 

    Vmin, Vmax = 0.0, raw_BCs["velocity-inlet"]["u"]

    fig_v, ax_v = plt.subplots(figsize=(9, 4))

    c1 = PolyCollection(quads, array=V_mag, cmap="coolwarm", edgecolors="none")
    c1.set_clim(Vmin, Vmax)

    ax_v.add_collection(c1)
    ax_v.set_xlim(*xlim); ax_v.set_ylim(*ylim)
    ax_v.set_aspect("auto")
    ax_v.set_title(f"Velocity Magnitude - Iteration {iter}")
    ax_v.set_xlabel("x"); ax_v.set_ylabel("y")

    fig_v.colorbar(c1, ax=ax_v, label="Velocity Magnitude [m/s]")

    # save velocity magnitude contour
    plt.tight_layout()
    fig_v.savefig(
        os.path.join(SAVE_DIR, f"velocity_magnitude_iter_{iter:06d}.png"), 
        dpi=300, bbox_inches="tight"
    )
    plt.close(fig_v)

    # plot continuity residual history
    fig_hist, ax_hist = plt.subplots(figsize=(9, 4))

    ax_hist.plot(residual_hist[:, 0], residual_hist[:, 2])
    ax_hist.set_yscale("log")
    ax_hist.set_xlabel("Iteration")
    ax_hist.set_ylabel("L1 Norm")
    ax_hist.set_title("L1 Norm of Continuity Residual")
    ax_hist.grid(True)

    # save continuity residual history
    plt.tight_layout()
    fig_hist.savefig(
        os.path.join(SAVE_DIR, f"continuity_residual_history_iter_{iter:06d}.png"),
        dpi=300, bbox_inches="tight",
    )
    plt.close(fig_hist)

    # save continuity history to CSV
    np.savetxt(
        os.path.join(SAVE_DIR, "continuity_residual_history.csv"),
        residual_hist,
        delimiter=",",
        header="Iteration, Max, L1_Norm, Net",
        comments="",
    )

    # Blasius comparison plot
    U_infty = np.sqrt(
        raw_BCs["velocity-inlet"]["u"]**2 + raw_BCs["velocity-inlet"]["v"]**2
    )
    plot_blasius(
        mesh, soln, x_eval, y_eval, U_infty, nu, blasius_file,
        y_wall=y_wall,
        save_path=os.path.join(SAVE_DIR, f"blasius_iter_{iter:06d}.png"),
    )

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