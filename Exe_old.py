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
import Export_of_graph

########################################################################################################################
#                                                                                                                      #
#                                                   INPUT                                                              #
#                                                                                                                      #
########################################################################################################################

# Arteries and Veins

path_artery = r'D:\00 Privat\01_Bildung\01_ETH Zürich\MSc\00_Masterarbeit\database_penetrating_trees\arteryDB'
path_vein = r'D:\00 Privat\01_Bildung\01_ETH Zürich\MSc\00_Masterarbeit\database_penetrating_trees\veinDB'

number_of_penetrating_vein_trees = 2
number_of_penetrating_artery_trees = 2

# Length hexagon equal to 62 microns (Diss Schmid)

length_honeycomb = 150
initial_diameter = 1.0

# Specify inlet and outlet pores

inlet_pores = [0, 1, 2, 3]
outlet_pores = [152, 153, 154, 155]

start_node_tracer = 0

# BC

total_inflow = 100
pressure_outlet_pores = 1

########################################################################################################################
#                                                                                                                      #
#                                                  Preparation                                                         #
#                                                                                                                      #
########################################################################################################################

# Main Graph

graph_main = ig.Graph()

# Evaluate penetrating trees (files) which should be part of the model

files = Import_Penetrating_Trees.random_choice_of_trees(number_of_penetrating_vein_trees,
                                                        number_of_penetrating_artery_trees)

files_arteries = files['Files_arteries']
files_veins = files['Files_veins']

# Import penetrating trees

graphs_artery_penetrating_trees = []
graphs_vein_penetrating_trees = []

# Import of artery graphs

for artery_file in files_arteries:

    graph_temp = Import_Penetrating_Trees.get_penetrating_tree_from_pkl_file(path_artery, artery_file, 1)
    graphs_artery_penetrating_trees.append(graph_temp)

# Import of vein graphs

for vein_file in files_veins:

    graph_temp = Import_Penetrating_Trees.get_penetrating_tree_from_pkl_file(path_vein, vein_file, 1)
    graphs_artery_penetrating_trees.append(graph_temp)

# Merge lists with vein and artery graphs

all_penetrating_trees = graphs_artery_penetrating_trees + graphs_vein_penetrating_trees

# Find max/min values for x, y and z coordinate (consider every penetrating tree (veins and arteries))

coordinates_limits = Import_Penetrating_Trees.get_coordinates_limits_from_several_graphs(all_penetrating_trees)

# Calculate number of combs in each direction (x, y, z) to generate the honeycomb network

number_of_combs = Import_Penetrating_Trees.get_number_of_combs(coordinates_limits, length_honeycomb)
print('number of combs: ', number_of_combs)

########################################################################################################################
#                                                                                                                      #
#                                                  EXECUTION                                                           #
#                                                                                                                      #
########################################################################################################################

# Generate whole network including penetrating trees -------------------------------------------------------------------


# Generate capillary bed (honeycomb network) for cuboid which is limited by coordinate min/max of penetrating trees

graph_main = Network_Generator.create_3d_graph(graph_main, number_of_combs[0], number_of_combs[1], number_of_combs[2],
                                               length_honeycomb, coordinates_limits)

# Add penetrating trees (veins/arteries) to capillary bed

for graph in all_penetrating_trees:

    graph_main = Network_Generator.add_penetrating_tree_to_cap_bed(graph, graph_main)


# TRACER PATH ----------------------------------------------------------------------------------------------------------

# Plot.plot_path(graph_main, path)


# PLOT -----------------------------------------------------------------------------------------------------------------

Plot.plot_graph(graph_main)

# TESTING --------------------------------------------------------------------------------------------------------------

print(ig.summary(graph_main))
print(graph_main.es[10])
print(graph_main.vs[10])




