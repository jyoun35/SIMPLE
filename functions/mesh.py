import gmsh
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict

class Mesh:
    """
    - Reads a 2D mesh of structured quad cells from a Gmsh .msh file
    - Builds face connectivity and computes geometric quantities
    """

    def __init__(self, filename, plot=False):
        self.filename = filename
        self._read_mesh()                
        self._convert_indices()
        self._build_connectivity() 
        self._compute_geom_params()
        if plot: self.plot()

    def _read_mesh(self):
        """
        Reads nodes and quads from mesh using the Gmsh API
        """

        # initialize Gmsh API to process mesh
        gmsh.initialize()
        gmsh.open(self.filename)

        # extract nodes of the mesh
        node_tags, node_coords, _ = gmsh.model.mesh.getNodes()

        self.node_tags = np.array(node_tags, dtype=int)
        self.node_coords = np.array(node_coords).reshape(-1, 3)[:, :2]
        self.tag_to_idx = {tag: i for i, tag in enumerate(self.node_tags)}

        # extract quad elements of the mesh
        element_types, element_tags, element_nodes = gmsh.model.mesh.getElements(dim=2)

        self.cells = []
        for type, nodes in zip(element_types, element_nodes):
            quads = np.array(nodes, dtype=int).reshape(-1, 4)
            self.cells.extend(quads.tolist())

        self.n_cells = len(self.cells)

        # extract boundary edges
        boundaries = gmsh.model.getPhysicalGroups(dim=1)

        self.boundary_edges = {} 
        for dimension, tag in boundaries:
            boundary_name = gmsh.model.getPhysicalName(dimension, tag)
            boundary_curves = gmsh.model.getEntitiesForPhysicalGroup(dimension, tag)

            boundary_edges = []
            for curve in boundary_curves:
                element_types, element_tags, element_nodes = gmsh.model.mesh.getElements(1, curve)

                for type, nodes in zip(element_types, element_nodes):
                    line_nodes = np.array(nodes, dtype=int).reshape(-1, 2)
                    boundary_edges.extend(line_nodes.tolist())

            self.boundary_edges[boundary_name] = boundary_edges

        # close out Gmsh API
        gmsh.finalize()

    def _convert_indices(self):
        """
        Convert cells and boundaries from Gmsh tag to Python array indices
        """

        # convert cell node tags to indices
        self.cells = [[self.tag_to_idx[node] for node in cell] for cell in self.cells]

        # convert boundary edge node tags to indices
        for name, edges in self.boundary_edges.items():
            self.boundary_edges[name] = [[self.tag_to_idx[node_1], self.tag_to_idx[node_2]] for node_1, node_2 in edges]

    def _build_connectivity(self):
        '''
        Determine mesh connectivity:
        - self.internal_faces = (node_1, node_2, cell_1, cell_2)
        - self.boundary_faces = (node_1, node_2, cell, boundary_name)         
        '''

        # build a mapping from edges to the cells that share them
        edge_to_cells = defaultdict(list)
        for cell_idx, cell in enumerate(self.cells):
            for i in range(4):
                node_1 = cell[i]
                node_2 = cell[(i+1) % 4]
                key = tuple(sorted((node_1, node_2)))
                edge_to_cells[key].append(cell_idx)

        # build a mapping from edges to boundary names
        edge_to_name = defaultdict(list)
        for name, edges in self.boundary_edges.items():
            for node_1, node_2 in edges:
                key = tuple(sorted((node_1, node_2)))
                edge_to_name[key] = name

        # determine internal and boundary faces
        self.internal_faces = []
        self.boundary_faces = [] 

        for (node_1, node_2), cells in edge_to_cells.items():
            if len(cells) == 2:
                self.internal_faces.append((node_1, node_2, cells[0], cells[1]))

            elif len(cells) == 1:
                boundary_name = edge_to_name.get((node_1, node_2))
                self.boundary_faces.append((node_1, node_2, cells[0], boundary_name))

        # count the number of internal and boundary faces
        self.n_internal = len(self.internal_faces)
        self.n_boundary = len(self.boundary_faces)

    def _compute_geom_params(self):
        '''
        Compute the geometric parameters of the mesh:
        - self.cell_centroids = (x, y)
        - self.cell_areas = area
        - self.face_normals = {(node_1, node_2, cell_1, cell_2/boundary_name): np.array([nx, ny])}
        - self.face_lengths = {(node_1, node_2, cell_1, cell_2/boundary_name): edge_length}
        '''

        # initialize arrays for cell centroids and areas
        self.cell_centroids = np.zeros((self.n_cells, 2))
        self.cell_areas = np.zeros(self.n_cells)

        # compute cell centroids and areas
        for cell_idx, cell in enumerate(self.cells):
            cell_pts = self.node_coords[cell]
            self.cell_centroids[cell_idx] = np.mean(cell_pts, axis=0)

            # shoelace formula for area of a quadrilateral
            self.cell_areas[cell_idx] = (1/2) * abs(
                cell_pts[0,0]*cell_pts[1,1] - cell_pts[1,0]*cell_pts[0,1] +
                cell_pts[1,0]*cell_pts[2,1] - cell_pts[2,0]*cell_pts[1,1] +
                cell_pts[2,0]*cell_pts[3,1] - cell_pts[3,0]*cell_pts[2,1] +
                cell_pts[3,0]*cell_pts[0,1] - cell_pts[0,0]*cell_pts[3,1]
            )

        # initialize dictionaries for face centroids, normals, and lengths
        self.face_centroids = {}
        self.face_normals = {}
        self.face_lengths = {}

        # compute face centroids, normals, and lengths for internal faces
        for internal_face in self.internal_faces:

            # collect nodes and cell index and associated points
            node_1, node_2, cell_1, cell_2 = internal_face
            point_1, point_2 = self.node_coords[node_1], self.node_coords[node_2]

            # compute edge centroid, vector and length
            edge_centroid = (1/2) * (point_1 + point_2)
            edge_vector = point_2 - point_1
            edge_length = np.linalg.norm(edge_vector)

            # determine perpendicular normal vector
            nx = -edge_vector[1] / edge_length
            ny =  edge_vector[0] / edge_length

            # check that normal faces away from cell centroid
            if np.dot(edge_centroid - self.cell_centroids[cell_1], (nx, ny)) < 0: 
                nx, ny = -nx, -ny

            # use a key for faces that contains face nodes and connecting cells
            key = (node_1, node_2, cell_1, cell_2)

            # store face centroid, normal, and length
            self.face_centroids[key] = edge_centroid
            self.face_normals[key] = np.array([nx, ny])
            self.face_lengths[key] = edge_length

        # compute face centroids, normals, and lengths for boundary faces
        for boundary_face in self.boundary_faces:

            # collect nodes and cell index and associated points
            node_1, node_2, cell, boundary_name = boundary_face
            point_1, point_2 = self.node_coords[node_1], self.node_coords[node_2]

            # compute edge centroid, vector and length
            edge_centroid = 0.5 * (point_1 + point_2)
            edge_vector = point_2 - point_1
            edge_length = np.linalg.norm(edge_vector)

            # determine perpendicular normal vector
            nx = -edge_vector[1] / edge_length
            ny =  edge_vector[0] / edge_length

            # check that normal faces away from cell centroid
            if np.dot(edge_centroid - self.cell_centroids[cell], (nx, ny)) < 0:
                nx, ny = -nx, -ny

            # use a key for faces that contains face nodes and connecting cells
            key = (node_1, node_2, cell, boundary_name)

            # store face centroid, normal, and length
            self.face_centroids[key] = edge_centroid
            self.face_normals[key] = np.array([nx, ny])
            self.face_lengths[key] = edge_length

    def plot(self):
        """
        Plot the mesh
        """

        # initialize plot
        fig, ax = plt.subplots(figsize=(8,6))

        # plot the nodes of each of the quad cells
        for cell in self.cells:
            pts = self.node_coords[cell]
            pts = np.vstack([pts, pts[0]])
            ax.plot(pts[:,0], pts[:,1], 'k-', linewidth=0.5)

        # plot and label the boundary faces
        if self.boundary_faces:

            # extract boundary name
            boundary_name = list(set(boundary_face[3] for boundary_face in self.boundary_faces))

            # assign colors to boundaries
            color_cycle = ['red','blue','green','orange','purple','cyan','magenta']
            colour_map = {name: color_cycle[i % len(color_cycle)] for i, name in enumerate(boundary_name)}

            # plot each boundary face
            for face in self.boundary_faces:
                node_1, node_2, _, name = face
                pts = self.node_coords[[node_1, node_2]]
                ax.plot(pts[:,0], pts[:,1], color=colour_map.get(name, 'black'), linewidth=2.5, label=name)

            # remove duplicate legend labels
            handles, labels = ax.get_legend_handles_labels()
            unique_labels = dict(zip(labels, handles))

            # show unique legend labels
            if unique_labels: ax.legend(unique_labels.values(), unique_labels.keys())

        # set up the plot
        ax.set_aspect('equal')
        ax.set_xlabel('x'); ax.set_ylabel('y')
        ax.set_title(f'Mesh: {self.filename}')
        plt.tight_layout()
        plt.show()

if __name__ == "__main__":
    mesh = Mesh("fine_CMesh_NACA_2412.msh", plot=True)
    print(f"# of Quad Cells: {mesh.n_cells}")
    print(f"# of Internal Faces: {mesh.n_internal}")
    print(f"# of Boundary Faces: {mesh.n_boundary}")