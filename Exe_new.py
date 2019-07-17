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
import Chose_activated_region
import time

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

# Beta-Distribution for diameters of capillary bed (in meters)

diameter_standard_deviation = 1 / math.pow(10, 6)
diameter_mean = 4 / math.pow(10, 6)
diameter_min = 2.5 / math.pow(10, 6)
diameter_max = 9 / math.pow(10, 6)

# Penetrating Trees (in microns)

length_x = 400
length_y = 400
scaling_factor = 1

# Honeycomb Penetrating Trees (geometrical help only, in microns)

length_hexagon_penetrating_trees = 200

# BC (in Pascal)

pressure_veins = 1333.22
pressure_arteries = 10600

# Activated Edges ( in microns)

coordinates_sphere = {'x': 200, 'y': 200, 'z': 200}
radius_sphere = 100

# Modifiable Edges -> coordinates should be the same than for the activated edges
all_edges = 0           # 1 or 0, 0: not all edges, 1: all edges
coordinates_modifiable_sphere = coordinates_sphere
radii_sphere_mod = [100, 120, 140, 160, 200]

key_modifiable_edges = [0, 1, 2, 3]         # 0: Only CB, 1: Only Arteries, 2: CB and Arteries, 3: all

# Summary for txt file

summary_dict = {'L_Honeycomb_CB': length_honeycomb, 'd_std': diameter_standard_deviation, 'd_mean': diameter_mean,
                'd_min': diameter_min, 'd_max': diameter_max, 'L_x_total': length_x, 'L_y_total': length_y,
                'L_Honeycomb_Help': length_hexagon_penetrating_trees, 'Coordinates_Sphere': coordinates_sphere,
                'radius_sphere_activated': radius_sphere, 'Coordinates_modifiable_sphere': coordinates_sphere,
                'radius_sphere_modifiable': radii_sphere_mod, 'All Edges': all_edges,
                'Modifiable_Edges_Key': key_modifiable_edges, 'pressure_veins': pressure_veins,
                'pressure_arteries': pressure_arteries}

########################################################################################################################
#                                                                                                                      #
#                                                      EXE                                                             #
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

        key_number_vein = 1

        x_coordinate_root = graph_geometrical_help.vs[root]['x_coordinate']
        y_coordinate_root = graph_geometrical_help.vs[root]['y_coordinate']
        z_coordinate_root = graph_geometrical_help.vs[root]['z_coordinate']

        start_point = (x_coordinate_root, y_coordinate_root, z_coordinate_root)

        id_vein = Import_Penetrating_Trees.random_choice_of_venous_tree()

        venous_tree = Import_Penetrating_Trees.get_penetrating_tree_from_pkl_file(path_vein, id_vein, scaling_factor,
                                                                                  key_number_vein)

        graph_penetrating = Network_Generator.add_penetrating_tree_to_tot(venous_tree, graph_penetrating, start_point)

    elif graph_geometrical_help.vs[root]['artery_point']:

        key_number_artery = 2

        x_coordinate_root = graph_geometrical_help.vs[root]['x_coordinate']
        y_coordinate_root = graph_geometrical_help.vs[root]['y_coordinate']
        z_coordinate_root = graph_geometrical_help.vs[root]['z_coordinate']

        start_point = (x_coordinate_root, y_coordinate_root, z_coordinate_root)

        id_arterial = Import_Penetrating_Trees.random_choice_of_arterial_tree()

        arterial_tree = Import_Penetrating_Trees.get_penetrating_tree_from_pkl_file(path_artery, id_arterial,
                                                                                    scaling_factor, key_number_artery)

        graph_penetrating = Network_Generator.add_penetrating_tree_to_tot(arterial_tree, graph_penetrating, start_point)

# Plot.plot_graph(graph_penetrating)
print(time.strftime("%H%M%S"))
print('Penetrating Trees are merged ... ')

coord_lim_cap_bed = Import_Penetrating_Trees.coordinates_limits(graph_penetrating)
combs_cap_bed = Import_Penetrating_Trees.get_number_of_combs(coord_lim_cap_bed, length_honeycomb)

diameter_dist_information = {'mean': diameter_mean, 'std': diameter_standard_deviation, 'min': diameter_min,
                             'max': diameter_max}

print(time.strftime("%H%M%S"))
print('Start to create capillary bed ... ')

graph_main = Network_Generator.create_3d_graph(graph_main, combs_cap_bed[0], combs_cap_bed[1], combs_cap_bed[2],
                                               length_honeycomb, coord_lim_cap_bed, diameter_dist_information)

print(time.strftime("%H%M%S"))
print('Capillary bed has been created ... ')

graph_main = Network_Generator.add_penetrating_tree_to_cap_bed(graph_penetrating, graph_main, length_honeycomb)

print(time.strftime("%H%M%S"))
print('Penetrating tree has been added to the capillary bed ...')

print('Select activation region ... ')

graph_main = Chose_activated_region.define_activated_region(graph_main, coordinates_sphere, radius_sphere)

if all_edges == 0:

    for radius in radii_sphere_mod:

        graph_main = Chose_activated_region.modifiable_region(graph_main, coordinates_modifiable_sphere, radius)

else:

    None
print(time.strftime("%H%M%S"))
print('Export of the csv files and selection of the modifiable region ... ')

summary_dict['number_of_edges'] = graph_main.ecount()
summary_dict['number_of_vertices'] = graph_main.vcount()

Export_of_graph.create_csv_files_from_graph(graph_main, pressure_veins, pressure_arteries, all_edges, summary_dict,
                                            key_modifiable_edges, radii_sphere_mod)

print(time.strftime("%H%M%S"))
print('END')

Plot.plot_graph(graph_main)

