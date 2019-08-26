import igraph as ig
import math
import pandas as pd
import numpy as np
import matplotlib as mpl
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt


def volume_fractions(graph):

    x_min = min(graph.vs['x_coordinate'])
    x_max = max(graph.vs['x_coordinate'])
    y_min = min(graph.vs['y_coordinate'])
    y_max = max(graph.vs['y_coordinate'])
    z_min = min(graph.vs['y_coordinate'])
    z_max = max(graph.vs['y_coordinate'])

    delta_x = x_max - x_min
    delta_y = y_max - y_min
    delta_z = z_max - z_min

    volume_cuboid = (delta_x * delta_y * delta_z)*math.pow(10, -18)

    count_arterial_vessels = 0
    count_venous_vessels = 0
    count_capillary_bed_honeycomb = 0
    count_capillary_bed_connections = 0

    tot_volume_arterial_vessels = 0
    tot_volume_venous_vessels = 0
    tot_volume_capillary_bed_honeycomb = 0
    tot_volume_capillary_bed_connections = 0

    for edge in range(graph.ecount()):

        if graph.es[edge]['Type'] == 0:

            count_capillary_bed_honeycomb += 1
            len_edge = graph.es[edge]['edge_length']
            radius = graph.es[edge]['diameter'] / 2

            tot_volume_capillary_bed_honeycomb += math.pi * math.pow(radius, 2) * len_edge

        elif graph.es[edge]['Type'] == 2:

            count_arterial_vessels += 1
            len_edge = graph.es[edge]['edge_length']
            radius = graph.es[edge]['diameter'] / 2

            tot_volume_arterial_vessels += math.pi * math.pow(radius, 2) * len_edge

        elif graph.es[edge]['Type'] == 1:

            count_venous_vessels += 1
            len_edge = graph.es[edge]['edge_length']
            radius = graph.es[edge]['diameter'] / 2

            tot_volume_venous_vessels += math.pi * math.pow(radius, 2) * len_edge

        elif graph.es[edge]['Type'] == 3:

            count_capillary_bed_connections += 1
            len_edge = graph.es[edge]['edge_length']
            radius = graph.es[edge]['diameter'] / 2

            tot_volume_capillary_bed_connections += math.pi * math.pow(radius, 2) * len_edge

    fraction_all_vessels = (tot_volume_arterial_vessels + tot_volume_venous_vessels +
                            tot_volume_capillary_bed_connections + tot_volume_capillary_bed_honeycomb)/volume_cuboid

    fraction_arteries = tot_volume_arterial_vessels / volume_cuboid

    fraction_venous = tot_volume_venous_vessels / volume_cuboid

    fraction_capillaries = (tot_volume_capillary_bed_honeycomb + tot_volume_capillary_bed_connections) / volume_cuboid

    fraction_dict = {'All': fraction_all_vessels, 'Veins': fraction_venous, 'Arteries': fraction_arteries, 'Capillaries': fraction_capillaries}

    return fraction_dict

########################################################################################################################
#                                                    Volume Fractions                                                  #
########################################################################################################################

for i in range(2):

    path = 'D:\\00 Privat\\01_Bildung\\01_ETH Zürich\MSc\\00_Masterarbeit\\03_Network_Study_Large\\Networks\\' \
           + str(i) + '\\graph.pkl'

    graph_ = ig.Graph.Read_Pickle(path)
    print(volume_fractions(graph_))
