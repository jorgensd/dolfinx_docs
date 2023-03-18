# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.13.8
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# # Mesh creation in serial and parallel
# Author: Jørgen S. Dokken
#
# In this tutorial we will consider the first important class in DOLFINx, the `dolfinx.mesh.Mesh` class.
#
# A mesh consists of a set of cells. These cells can be intervals, triangles, quadrilaterals, hexahedrons or tetrahedrons.
# Each cell is described by a set of coordinates, and its connectivity.
#
# ## Mesh creation from numpy arrays
# For instance, let us consider a unit square. If we want to discretize it with triangular elements, we could create the set of vertices as a $(4\times 2)$ numpy array

import numpy as np
tri_points = np.array([[0,0],[0,1],[1,0],[1,1]], dtype=np.float64)

# Next, we have to decide on how we want to create the two triangles in the mesh. Let's choose the first cell to consist of vertices $0,1,3$ and the second cell consist of vertices $0,2,3$

triangles = np.array([[0,1,3], [0,2,3]], dtype=np.int64)

# We note that for triangular cells, we could order the vertices in any order say `[[1,3,0],[2,0,3]]`, and the mesh would equivalent.
# Some finite element software reorder cells to ensure consistent integrals over interior facets.
# In DOLFINx, another strategy has been chosen, see {cite}`10.1145/3524456` for more details.
#
# Let's consider the unit square again, but this time we want to discretize it with two quadrilateral cells.

quad_points = np.array([[0,0],[0.3, 0], [1, 0], [0,1], [0.4, 1], [1, 1]], dtype=np.float64)
quadrilaterals = np.array([[0, 1, 3, 4], [1, 2, 4, 5]], dtype=np.int64)

# Note that we do not parse the quadrilateral cell in a clockwise or counter-clockwise fashion.
# Instead, we are using a tensor product ordering.
# The ordering of the sub entities all cell types used in DOLFINx can be found at [Basix supported elements](https://github.com/FEniCS/basix/#supported-elements).
# We also note that this unit square mesh has non-affine elements.
#
# Next, we would like to generate the mesh used in DOLFINx.
# To do so, we need to generate the coordinate element.
# This is the paramterization of each an every element, and the only way of going between the physical element and the reference element.
# We will denote any coordinate on the reference element as $\mathbf{X}$,
# and any coordinate in the physical element as $\mathbf{x}$,
# with the mapping $M$ such that $\mathbf{x} = M(\mathbf{X})$.
#
# We can write
#
# $$
# \begin{align}
# M(\mathbf{X})= \sum_{i=0}^{\text{num vertices}} \mathbf{v}_i\phi_i(\mathbf{X}
# \end{align})
# $$
#
# where $\mathbf{v}_i$ is the $i$th vertex of a cell and $\phi_i$ are the basis functions specifed at [DefElement P1 triangles](https://defelement.com/elements/examples/triangle-Lagrange-1.html) and
# [DefElement Q1 quadrilaterals](https://defelement.com/elements/examples/quadrilateral-Q-1.html).
#
# In DOLFINx we use the [Unified Form Language](https://github.com/FEniCS/ufl/) to define finite elements.
# Therefore we create the `ufl.Mesh`

import ufl
ufl_tri = ufl.Mesh(ufl.VectorElement("Lagrange", ufl.triangle, 1))
ufl_quad = ufl.Mesh(ufl.VectorElement("Lagrange", ufl.quadrilateral, 1))

# This is all the input we need to a DOLFINx mesh

# +
import dolfinx
from mpi4py import MPI

quad_mesh = dolfinx.mesh.create_mesh(
    MPI.COMM_WORLD, quadrilaterals, quad_points, ufl_quad)
tri_mesh = dolfinx.mesh.create_mesh(
    MPI.COMM_WORLD, triangles, tri_points, ufl_tri)
# -

# The only input to this function we have not covered so far is the `MPI.COMM_WORLD`, which is an MPI communicator.
#
# ### MPI Communication
# When we run a python code with `python3 name_of_file.py`. We execute python on a single process on the computer. However, if we launch the code with `mpirun -n N python3 name_of_file.py`, we execute the code on `N` processes at the same time. The `MPI.COMM_WORLD` is the communicator among the `N` processes, which can be used to send and receive data. If we use `MPI.COMM_SELF`, the communicator will not communicate with other processes.
# When we run in serial, `MPI.COMM_WORLD` is equivalent to `MPI.COMM_SELF`.
#
# Two important values in the MPI-communicator is its `rank` and `size`.
# If we run this in serial on either of the communicators above, we get
#

print(f"{MPI.COMM_WORLD.rank=} {MPI.COMM_WORLD.size=}")
print(f"{MPI.COMM_SELF.rank=} {MPI.COMM_SELF.size=}")

# In jupyter noteboooks, we use [ipyparallel](https://ipyparallel.readthedocs.io/en/latest/) to start a cluster and connect to two processes, which we can execute commands on using the magic `%%px` at the top of each cell. See [%%px Cell magic](https://ipyparallel.readthedocs.io/en/latest/tutorial/magics.html#px-cell-magic) for more details.
#
# ```{note}
# When starting a cluster, we do not carry ower any modules or variables from the previously executed code in the script.
# ```

# + tags=["hide-output"]
import ipyparallel as ipp
cluster = ipp.Cluster(engines="mpi", n=2)
rc = cluster.start_and_connect_sync()
# -

# Next, we import `mpi4py` on the two engines and check the rank and size of the two processes.

# +
# %%px
from mpi4py import MPI as MPIpx
import numpy as np
import ufl
import dolfinx

print(f"{MPIpx.COMM_WORLD.rank=} {MPIpx.COMM_WORLD.size=}")
# -

# Next, we want to create the triangle mesh, distributed over the two processes.
# We do this by sending in the points and cells for the mesh in one of two ways:
#
# **1. Send all points and cells on one process**

# %%px
if MPIpx.COMM_WORLD.rank == 0:
    tri_points = np.array([[0,0],[0,1],[1,0],[1,1]], dtype=np.float64)
    triangles = np.array([[0,1,3], [1,2,3]], dtype=np.int64)
else:
    tri_points = np.empty((0,2), dtype=np.float64)
    triangles = np.empty((0,3), dtype=np.int64)
ufl_tri = ufl.Mesh(ufl.VectorElement("Lagrange", ufl.triangle, 1))
tri_mesh = dolfinx.mesh.create_mesh(
    MPIpx.COMM_WORLD, triangles, tri_points, ufl_tri)
cell_index_map = tri_mesh.topology.index_map(tri_mesh.topology.dim)
print(f"Num cells local: {cell_index_map.size_local}\n Num cells global: {cell_index_map.size_global}")

# From the output above, we see the distribution of cells on each process.
#
# **2. Distribute input of points and cells**
#
# For large meshes, reading in all points and cells on a single process would be a bottle-neck.
# Therefore, we can read in the points and cells in a distributed fashion.
# Note that if we do this it important to note that it is assumed that rank 0 has read in the first chunck of points and cells in a continuous fashion. 

# +
# %%px
if MPIpx.COMM_WORLD.rank == 0:
    quadrilaterals = np.array([], dtype=np.int64)
    quad_points = np.array([[0,0],[0.3, 0]], dtype=np.float64)
elif MPIpx.COMM_WORLD.rank == 1:
    quadrilaterals = np.array([[0, 1, 3, 4], [1, 2, 4, 5]], dtype=np.int64)
    quad_points = np.array([[1, 0], [0,1], [0.4, 1], [1, 1]], dtype=np.float64)
else:
    quad_points = np.empty((0,2), dtype=np.float64)
    quadrilaterals = np.empty((0,4), dtype=np.int64)

ufl_quad = ufl.Mesh(ufl.VectorElement("Lagrange", ufl.quadrilateral, 1))
quad_mesh = dolfinx.mesh.create_mesh(
    MPIpx.COMM_WORLD, quadrilaterals, quad_points, ufl_quad)
cell_index_map = quad_mesh.topology.index_map(quad_mesh.topology.dim)
print(f"Num cells local: {cell_index_map.size_local}\n Num cells global: {cell_index_map.size_global}")
# -

# ### Usage of MPI.COMM_SELF
# You might wonder, if we can use multiple processes, when would we ever use `MPI.COMM_SELF`?
# There are many reasons for this. For instance, many simulations are too small to gain from parallelizing.
# Then one could use `MPI.COMM_SELF` with multiple processes to run parameterized studies in parallel

# %%px
serial_points = np.array([[0,0],[0.3, 0], [1, 0], [0,1], [0.4, 1], [1, 1]], dtype=np.float64)
serial_quads = np.array([[0, 1, 3, 4], [1, 2, 4, 5]], dtype=np.int64)
serial_mesh = dolfinx.mesh.create_mesh(
    MPIpx.COMM_SELF, serial_quads, serial_points, ufl_quad)
cell_index_map = serial_mesh.topology.index_map(serial_mesh.topology.dim)
print(f"Num cells local: {cell_index_map.size_local}\n Num cells global: {cell_index_map.size_global}")

# ## Mesh-partitioning
# As we have seen above, we can send in data to mesh creation and get either a distributed mesh out, but how does it work?
# Under the hood, what happens is that DOLFINx calls a graph-partitioning algorithm.
# This algorithm is supplied from either from PT-Scotch{cite}`10.1016/j.parco.2007.12.001`, ParMETIS{cite}`10.1145/369028.369103` or KaHIP{cite}`10.1007/978-3-642-38527-8_16`, depending on what is available with your installation.
#
# We can list the available partitioners with the following code:

# %%px
try:
    from dolfinx.graph import partitioner_scotch
    has_scotch = True
except ImportError:
    has_scotch = False
try:
    from dolfinx.graph import partitioner_kahip
    has_kahip = True
except ImportError:
    has_kahip = False
try:
    from dolfinx.graph import partitioner_parmetis
    has_parmetis = True
except ImportError:
    has_parmetis = False
print(f"{has_scotch=}  {has_kahip=} {has_parmetis=}")

# Given any of these partitioners (we will from now on use Scotch), you can send them into create mesh by calling

# %%px
assert has_scotch
partitioner = dolfinx.mesh.create_cell_partitioner(partitioner_scotch())
quad_mesh = dolfinx.mesh.create_mesh(
    MPIpx.COMM_WORLD, quadrilaterals, quad_points, ufl_quad, partitioner=partitioner)

# ## References
# ```{bibliography}
# :filter: docname in docnames
# ```
#
