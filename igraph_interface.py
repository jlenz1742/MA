import numpy as np
from scipy.sparse import csr_matrix, eye


def laplacian_from_igraph(graph, weights=None, ind_dirichlet=None):

    """Creates Laplacian of undirected graph using provided edge weights"""

    edges = graph.get_edgelist()
    if weights is None:
        weights = np.ones(len(edges))
    else:
        if len(weights) != len(edges):
            raise ValueError("The number of weights have to be equal to number of edges")

    edges = list(zip(*edges))

    row = edges[0]
    col = edges[1]

    nv = graph.vcount()

    # make matrix symmetric
    row, col = np.hstack((row, col)), np.hstack((col, row))
    data = -np.hstack((weights, weights))

    diagonal_dirichlet = np.zeros(nv)

    if ind_dirichlet is not None:
        # Set rows of matrix corresponding to dirichlet vertices to zero

        mask = np.zeros(nv, dtype=np.bool)
        mask[ind_dirichlet] = True
        mask_rows = mask[row]
        data[mask_rows] = 0.0

        # Set diagonal entries, to be added later, to one
        diagonal_dirichlet[ind_dirichlet] = 1.0

    # Only rows which are not listed in ind_dirichlet are non-empty
    off_diag = csr_matrix((data, (row, col)), shape=(nv, nv))

    diag = eye(nv, format="csr")
    diag.setdiag(-(off_diag*np.ones(nv)) + diagonal_dirichlet)

    return off_diag + diag


def laplacian_from_igraph_get_b(graph, weights=None, ind_dirichlet=None):
    """Creates Laplacian of undirected graph using provided edge weights"""

    edges = graph.get_edgelist()
    if weights is None:
        weights = np.ones(len(edges))
    else:
        if len(weights) != len(edges):
            raise ValueError("The number of weights have to be equal to number of edges")

    edges = zip(*edges)

    row = edges[0]
    col = edges[1]

    nv = graph.vcount()

    # make matrix symmetric
    row, col = np.hstack((row, col)), np.hstack((col, row))
    data = -np.hstack((weights, weights))

    diagonal_dirichlet = np.zeros(nv)


    # Only rows which are not listed in ind_dirichlet are non-empty
    off_diag = csr_matrix((data, (row, col)), shape=(nv, nv))

    diag = eye(nv, format="csr")
    diag.setdiag(-(off_diag*np.ones(nv)) + diagonal_dirichlet)

    return off_diag + diag