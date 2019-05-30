import igraph as ig
import Tools_Forward_Problem
import numpy as np
import igraph_interface
import random
import Network_Generator
import Tracer_Tracking_Algorithm
import Plot
import math
import Import_Penetrating_Trees

########################################################################################################################
#                                                                                                                      #
#                                                   INPUT                                                              #
#                                                                                                                      #
########################################################################################################################

# Arteries and Veins

path_artery = r'D:\00 Privat\01_Bildung\01_ETH Zürich\MSc\00_Masterarbeit\database_penetrating_trees\arteryDB'
path_vein = r'D:\00 Privat\01_Bildung\01_ETH Zürich\MSc\00_Masterarbeit\database_penetrating_trees\veinDB'

# number_of_penetrating_vein_trees = 1
# number_of_penetrating_artery_trees = 1

# Length hexagon equal to 62 microns (Diss Schmid)

length_honeycomb = 100
initial_diameter = 1.0

# Penetrating Trees

length_x = 1000
length_y = 1000
scaling_factor = 0.66

# Honeycomb Penetrating Trees (geometrical help only)

length_hexagon_penetrating_trees = 500

# Specify inlet and outlet pores

inlet_pores = [0, 1, 2, 3]
outlet_pores = [152, 153, 154, 155]
start_node_tracer = 0

# BC

total_inflow = 100
pressure_outlet_pores = 1

########################################################################################################################
#                                                                                                                      #
#                                                      Test                                                            #
#                                                                                                                      #
########################################################################################################################

# Main Graph

graph_main = ig.Graph()

# Create geometrical help (graph)

graph_geometrical_help = ig.Graph()

coordinates_limits = dict()
coordinates_limits['x_min'] = 0
coordinates_limits['x_max'] = length_x
coordinates_limits['y_min'] = 0
coordinates_limits['y_max'] = length_y
coordinates_limits['z_min'] = 0
coordinates_limits['z_max'] = 0

combs = Import_Penetrating_Trees.get_number_of_combs(coordinates_limits, length_hexagon_penetrating_trees)

graph_geometrical_help = Network_Generator.create_plane_geometrical_help(graph_geometrical_help, combs[0], combs[1], 0,
                                                                         length_hexagon_penetrating_trees,
                                                                         coordinates_limits)
# Plot.plot_geometrical_help(graph_geometrical_help)

# Create Penetrating Tree Graph (Connected)

graph_penetrating = ig.Graph()

for root in range(graph_geometrical_help.vcount()):

    if graph_geometrical_help.vs[root]['vein_point']:

        x_coordinate_root = graph_geometrical_help.vs[root]['x_coordinate']
        y_coordinate_root = graph_geometrical_help.vs[root]['y_coordinate']
        z_coordinate_root = graph_geometrical_help.vs[root]['z_coordinate']

        start_point = (x_coordinate_root, y_coordinate_root, z_coordinate_root)

        id_vein = Import_Penetrating_Trees.random_choice_of_venous_tree()

        venous_tree = Import_Penetrating_Trees.get_penetrating_tree_from_pkl_file(path_vein, id_vein, scaling_factor)

        graph_penetrating = Network_Generator.add_penetrating_tree_to_tot(venous_tree, graph_penetrating, start_point)

    elif graph_geometrical_help.vs[root]['artery_point']:

        x_coordinate_root = graph_geometrical_help.vs[root]['x_coordinate']
        y_coordinate_root = graph_geometrical_help.vs[root]['y_coordinate']
        z_coordinate_root = graph_geometrical_help.vs[root]['z_coordinate']

        start_point = (x_coordinate_root, y_coordinate_root, z_coordinate_root)

        id_arterial = Import_Penetrating_Trees.random_choice_of_arterial_tree()

        arterial_tree = Import_Penetrating_Trees.get_penetrating_tree_from_pkl_file(path_artery, id_arterial,
                                                                                    scaling_factor)

        graph_penetrating = Network_Generator.add_penetrating_tree_to_tot(arterial_tree, graph_penetrating, start_point)

# Plot.plot_graph(graph_penetrating)

coord_lim_cap_bed = Import_Penetrating_Trees.coordinates_limits(graph_penetrating)
combs_cap_bed = Import_Penetrating_Trees.get_number_of_combs(coord_lim_cap_bed, length_honeycomb)
print(combs_cap_bed)


graph_main = Network_Generator.create_3d_graph(graph_main, combs_cap_bed[0], combs_cap_bed[1], combs_cap_bed[2],
                                               length_honeycomb, coord_lim_cap_bed)

graph_main = Network_Generator.add_penetrating_tree_to_cap_bed(graph_penetrating, graph_main)

Plot.plot_graph(graph_main)

