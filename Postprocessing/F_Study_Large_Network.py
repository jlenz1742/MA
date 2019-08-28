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


def flow_versus_depth(graph, segments, meshdata_file, path_eval, network):

    df_start = pd.read_csv(meshdata_file)

    z_min = min(graph.vs['z_coordinate'])
    z_max = max(graph.vs['z_coordinate'])

    delta_z = z_max - z_min

    segment_length = delta_z / segments

    regions = []
    labels_x = []
    averaged_flow_rates_per_region = []

    for i in range(segments):

        start = z_min + i * segment_length
        end = start + segment_length

        labels_x.append(np.round((start+end)/2, 2))
        regions.append([start, end])

    for region in range(len(regions)):

        reg = regions[region]
        z_lower = reg[0]
        z_upper = reg[1]

        flows_in_region = []
        eids_in_region = []
        corresponding_length = []

        for i in range(graph.ecount()):

            if graph.es[i]['Type'] == 0 or graph.es[i]['Type'] == 3:

                source_node = graph.es[i].source
                target_node = graph.es[i].target

                z_coord_source_node = graph.vs[source_node]['z_coordinate']
                z_coord_target_node = graph.vs[target_node]['z_coordinate']

                if z_lower < z_coord_source_node < z_upper:

                    flows_in_region.append(np.absolute(df_start['tav_Fplasma'][2 * i]))
                    eids_in_region.append(i)
                    corresponding_length.append(graph.es[i]['edge_length'])
                    graph.es[i]['RegionID'] = region

                elif z_lower < z_coord_target_node < z_upper:

                    flows_in_region.append(np.absolute(df_start['tav_Fplasma'][2 * i]))
                    eids_in_region.append(i)
                    corresponding_length.append(graph.es[i]['edge_length'])
                    graph.es[i]['RegionID'] = region

                elif z_coord_source_node > z_upper and z_coord_target_node < z_lower:

                    flows_in_region.append(np.absolute(df_start['tav_Fplasma'][2 * i]))
                    eids_in_region.append(i)
                    corresponding_length.append(graph.es[i]['edge_length'])
                    graph.es[i]['RegionID'] = region

                elif z_coord_source_node < z_lower and z_coord_target_node > z_upper:

                    flows_in_region.append(np.absolute(df_start['tav_Fplasma'][2 * i]))
                    eids_in_region.append(i)
                    corresponding_length.append(graph.es[i]['edge_length'])
                    graph.es[i]['RegionID'] = region

            else:

                None

        flows_as_array = np.array(flows_in_region)
        length_as_array = np.array(corresponding_length)

        total_length = np.sum(corresponding_length)

        averaged_flow_in_region = np.sum((flows_as_array * length_as_array)) / total_length

        averaged_flow_rates_per_region.append(averaged_flow_in_region)
        # plot_tree_with_regions(graph, region)

    plt.plot(labels_x, averaged_flow_rates_per_region)
    plt.xticks(labels_x, labels_x, rotation=45,  size=7)
    plt.ylabel('(Length) Averaged Flow Rates')
    plt.xlabel('Depth in $\mu$m')
    plt.gcf().subplots_adjust(bottom=0.2)
    plt.savefig(path_eval + '\\Flow_Versus_Depth_Network_' + str(network) + '.png')
    plt.clf()

    return None


########################################################################################################################
#                                                    Volume Fractions                                                  #
########################################################################################################################

# for i in range(2):
#
#     path = 'D:\\00 Privat\\01_Bildung\\01_ETH Zürich\MSc\\00_Masterarbeit\\03_Network_Study_Large\\Networks\\' \
#            + str(i) + '\\graph.pkl'
#
#     graph_ = ig.Graph.Read_Pickle(path)
#     print(volume_fractions(graph_))


########################################################################################################################
#                                               Depth versus Flow Rate                                                 #
########################################################################################################################

# ACHTUNG ES FEHLEN NOCH DIE BASEFLOW ORDNER MIT MESHDATA FILES -> MÜSSEN ZUERST ERSTELLT WERDEN

# for i in range(2):
#
#     path = 'D:\\00 Privat\\01_Bildung\\01_ETH Zürich\MSc\\00_Masterarbeit\\03_Network_Study_Large\\Networks\\' \
#            + str(i) + '\\graph.pkl'
#
#     path1 = 'D:\\00 Privat\\01_Bildung\\01_ETH Zürich\MSc\\00_Masterarbeit\\03_Network_Study_Large\\Flow_Problem\\' \
#            + str(i) + '_Baseflow\\out\\meshdata_249.csv'
#
#     path2 = 'D:\\00 Privat\\01_Bildung\\01_ETH Zürich\MSc\\00_Masterarbeit\\03_Network_Study_Large\\Flow_Problem\\' \
#             'Evaluation'
#
#     graph_ = ig.Graph.Read_Pickle(path)
#     flow_versus_depth(graph_, 10, path1, path2, i)