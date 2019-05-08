import igraph as ig
import math
from Plot import plot_graph
import Tools_Forward_Problem
import numpy as np
import igraph_interface
import random


def determine_path_of_a_tracer(graph_main, initial_diameter, outlet_pores, pressure_outlet_pores, inlet_pores, total_inflow, start_node):

    diameters = np.full(graph_main.ecount(), initial_diameter)

    alpha_initial = 1.0
    alphas = np.full(graph_main.ecount(), alpha_initial)

    # Set the transmissibilities -> Choose high transmissibility for the last edge (Outflow)

    transmissibilities = np.power(diameters * alpha_initial, 4)

    # Dirichlet boundary conditions - set the pressures

    values_dirichlet = np.zeros(graph_main.vcount())
    values_dirichlet[outlet_pores] = pressure_outlet_pores

    # Neumann boundary conditions - define the inflows

    values_neumann = np.zeros(graph_main.vcount())
    values_neumann[inlet_pores] = total_inflow / len(inlet_pores)

    pressure_field_initial = Tools_Forward_Problem.get_pressure_field(graph_main, transmissibilities, outlet_pores, values_dirichlet,
                                                                      values_neumann)
    pressure_difference_initial = Tools_Forward_Problem.generate_pressure_difference(graph_main, pressure_field_initial)
    fluxes = pressure_difference_initial * transmissibilities

    "Particle Tracking Algorithm"

    next_node = start_node
    path = []
    path.append(next_node)

    while next_node not in outlet_pores:

        neighbors = graph_main.neighbors(next_node)

        fluxes_specific_node = []
        possible_next_nodes = []
        cumulative_probabilities = [0]
        flux_tot = 0

        for neighbor in neighbors:

            current_id = graph_main.get_eid(next_node, neighbor)
            flux = (pressure_field_initial[next_node] - pressure_field_initial[neighbor]) * transmissibilities[
                current_id]

            if flux >= 0:
                flux_tot += flux
                fluxes_specific_node.append(flux)
                possible_next_nodes.append(neighbor)

        probabilities = [x / flux_tot for x in fluxes_specific_node]

        for i in range(len(probabilities)):

            cumulative_probabilities.append(cumulative_probabilities[-1] + probabilities[i])

        cumulative_probabilities.pop(0)
        number = random.uniform(0, 1)

        for i in range(len(cumulative_probabilities)):

            if number <= cumulative_probabilities[i]:

                path.append(possible_next_nodes[i])
                next_node = possible_next_nodes[i]

                break

            else:

                None

        if next_node in outlet_pores:

            break
        else:

            None

    return path
