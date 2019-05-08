import numpy as np
import math
from scipy.sparse.linalg import spsolve
import igraph_interface

"""Particle Tracking Algorithm"""

#Equations from K.S. Sorbie, "The inclusion of molecular diffusion effects in networking modelling"


def get_pressure_field(graph, conductances, outlet_pores, values_dirichlet, values_neumann):

    """ Creates pressure field  under consideration of boundary conditions"""

    # right_hand_side corresponds to "b" of the Equation A*p=b and contains the boundary conditions

    right_hand_side = values_dirichlet + values_neumann

    # Computation of the laplacian matrix (sparse form)

    a = igraph_interface.laplacian_from_igraph(graph, conductances, outlet_pores)

    # Solves the equation p=A^(-1)*b and generates the pressure field for the pore-network model
    p = spsolve(a, right_hand_side)

    return p


def inflow_outflow_per_node(graph, conductances, outlet_pores, pressure_field):

    """The right-hand side contains the the inflows and outflows per node"""

    # Computation of the laplacian matrix

    a = igraph_interface.laplacian_from_igraph_get_b(graph, conductances, outlet_pores)

    # p stands for the pressure field and can be computed with the function "pressure_field"

    p = pressure_field

    # The right-hand side can be computed by A*p

    right_hand_side = a.dot(p)

    return right_hand_side


def generate_pressure_difference(graph, pressure_field):

    """Generates pressure difference for every single edge. The list can be set as an edge attribute"""

    edge_list = np.asarray(graph.get_edgelist())
    pressure_list = pressure_field[edge_list]
    pressure_difference = [0] * len(pressure_list)

    # WARNING: Absolute value of the pressure difference is going to be computed

    for i in range(len(pressure_list)):

        pressure_difference[i] = np.abs(pressure_list[i][1] - pressure_list[i][0])

    return pressure_difference


def incident_flowlist_single_vertex(graph, conductances, pressure_field, vertexId):

    """Creates array of incoming and outgoing flows for a single vertex, Convenction: Inflow>0, Outflows<0"""

    # Creates array with neighboring nodes for given vertexId

    incident_nodes = np.asarray(graph.neighbors(vertexId))

    # Creates array with Id's of the attached edges.
    # The order of incident edges corresponds to the order of incident nodes

    incident_edges = np.asarray(graph.incident(vertexId))

    incident_pressure_differences = pressure_field[incident_nodes] - pressure_field[vertexId]
    incident_conductances = np.asarray([conductances[_] for _ in incident_edges])
    incident_flows = incident_pressure_differences*incident_conductances

    return incident_flows


"""Create Matrices"""


def generate_g_x(transmissibilities, graph, outlet_pores):

    a = transmissibilities
    b = graph
    g_x = igraph_interface.laplacian_from_igraph(b, a, outlet_pores)

    return g_x


def generate_p_alpha(d_initial, alphas):

    p_alpha = 4*math.pow(d_initial, 4)*alphas

    return p_alpha


def generate_g_p(d_initial, alphas, graph, pressure_field):

    p_alphas = generate_p_alpha(d_initial, alphas)
    g_p = np.zeros((graph.vcount(), graph.ecount()))

    for i in range(graph.vcount()):

        list_neighbors = graph.neighbors(i)
        length_list = len(list_neighbors)
        list_edge_ids = np.zeros(length_list)
        list_edge_ids_new = list_edge_ids.astype(int)

        for a in range(length_list):
            list_edge_ids_new[a]=graph.get_eid(i, list_neighbors[a])
            g_p[i, list_edge_ids_new[a]] = (pressure_field[i]-pressure_field[list_neighbors[a]])*p_alphas[list_edge_ids_new[a]]

    return g_p
