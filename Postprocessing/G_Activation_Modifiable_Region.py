import math
import igraph as ig
import matplotlib as mpl
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import time
import pandas as pd
import os
import numpy as np
import csv


def define_activated_region(graph, coords_sphere, r_sphere, path_csv_file):

    # Delete current selection

    for e in range(graph.ecount()):

        graph.es[e]['Activated'] = 0

    # Start new selection

    activated_edge_ids = []

    for edge in range(graph.ecount()):

        p1 = graph.es[edge].source
        x_1 = graph.vs[p1]['x_coordinate'] / math.pow(10, 6)
        y_1 = graph.vs[p1]['y_coordinate'] / math.pow(10, 6)
        z_1 = graph.vs[p1]['z_coordinate'] / math.pow(10, 6)

        p2 = graph.es[edge].target
        x_2 = graph.vs[p2]['x_coordinate'] / math.pow(10, 6)
        y_2 = graph.vs[p2]['y_coordinate'] / math.pow(10, 6)
        z_2 = graph.vs[p2]['z_coordinate'] / math.pow(10, 6)

        x_3 = coords_sphere['x'] / math.pow(10, 6)
        y_3 = coords_sphere['y'] / math.pow(10, 6)
        z_3 = coords_sphere['z'] / math.pow(10, 6)

        radius = r_sphere / math.pow(10, 6)

        a = math.pow(x_2 - x_1, 2) + math.pow(y_2 - y_1, 2) + math.pow(z_2 - z_1, 2)
        b = 2 * ((x_2 - x_1)*(x_1 - x_3) + (y_2 - y_1)*(y_1 - y_3) + (z_2 - z_1)*(z_1 - z_3))
        c = math.pow(x_3, 2) + math.pow(y_3, 2) + math.pow(z_3, 2) + math.pow(x_1, 2) + math.pow(y_1, 2) + math.pow(z_1, 2) - 2 * (x_3 * x_1 + y_3 * y_1 + z_3 * z_1) - math.pow(radius, 2)

        value = math.pow(b, 2) - 4 * a * c

        if value >= 0:

            u_1 = (-b + math.sqrt(value)) / (2 * a)
            u_2 = (-b - math.sqrt(value)) / (2 * a)

            # Line segment doesnt intersect but is inside sphere

            if u_1 < 0 and u_2 > 1:

                activated_edge_ids.append(edge)

            elif u_2 < 0 and u_1 > 1:

                activated_edge_ids.append(edge)

            # Line segment intersects at one point

            elif (0 <= u_1 <= 1) and (u_2 > 1 or u_2 < 0):

                activated_edge_ids.append(edge)

            elif (0 <= u_2 <= 1) and (u_1 > 1 or u_1 < 0):

                activated_edge_ids.append(edge)

            # Line segment intersects at two points

            elif (0 <= u_2 <= 1) and (0 <= u_1 <= 1):

                activated_edge_ids.append(edge)

            # Line segment is tangential

            elif (u_1 == u_2) and (0 <= u_1 <= 1):

                activated_edge_ids.append(edge)

            else:

                continue

        else:

            continue

    for activated_edge in activated_edge_ids:

        if graph.es[activated_edge]['Type'] == 0 or graph.es[activated_edge]['Type'] == 3:

            graph.es[activated_edge]['Activated'] = 1

    # Create CSV

    time_str = time.strftime("%Y%m%d_%H%M%S")

    activated_eids = []

    for edge in range(graph.ecount()):

        if graph.es[edge]['Activated'] == 1:
            activated_eids.append(edge)

    data = {'Activated': activated_eids}
    activated_eids_df = pd.DataFrame(data)
    activated_eids_df_new = activated_eids_df.T

    activated_eids_df_new.to_csv(path_csv_file + '\\activated_eids_temp.csv', index=False)

    with open(path_csv_file + '\\activated_eids_temp.csv', 'r') as f:
        with open(path_csv_file + '\\activated_eids.csv', 'w') as f1:
            next(f)  # skip header line
            for line in f:
                f1.write(line)

    os.remove(path_csv_file + '\\activated_eids_temp.csv')

    return graph


def define_activated_region_large_networks(graph, coords_sphere, r_sphere, path_csv_file, name_spec):

    # Delete current selection

    for e in range(graph.ecount()):

        graph.es[e]['Activated'] = 0

    # Start new selection

    activated_edge_ids = []

    for edge in range(graph.ecount()):

        p1 = graph.es[edge].source
        x_1 = graph.vs[p1]['x_coordinate'] / math.pow(10, 6)
        y_1 = graph.vs[p1]['y_coordinate'] / math.pow(10, 6)
        z_1 = graph.vs[p1]['z_coordinate'] / math.pow(10, 6)

        p2 = graph.es[edge].target
        x_2 = graph.vs[p2]['x_coordinate'] / math.pow(10, 6)
        y_2 = graph.vs[p2]['y_coordinate'] / math.pow(10, 6)
        z_2 = graph.vs[p2]['z_coordinate'] / math.pow(10, 6)

        x_3 = coords_sphere['x'] / math.pow(10, 6)
        y_3 = coords_sphere['y'] / math.pow(10, 6)
        z_3 = coords_sphere['z'] / math.pow(10, 6)

        radius = r_sphere / math.pow(10, 6)

        a = math.pow(x_2 - x_1, 2) + math.pow(y_2 - y_1, 2) + math.pow(z_2 - z_1, 2)
        b = 2 * ((x_2 - x_1)*(x_1 - x_3) + (y_2 - y_1)*(y_1 - y_3) + (z_2 - z_1)*(z_1 - z_3))
        c = math.pow(x_3, 2) + math.pow(y_3, 2) + math.pow(z_3, 2) + math.pow(x_1, 2) + math.pow(y_1, 2) + math.pow(z_1, 2) - 2 * (x_3 * x_1 + y_3 * y_1 + z_3 * z_1) - math.pow(radius, 2)

        value = math.pow(b, 2) - 4 * a * c

        if value >= 0:

            u_1 = (-b + math.sqrt(value)) / (2 * a)
            u_2 = (-b - math.sqrt(value)) / (2 * a)

            # Line segment doesnt intersect but is inside sphere

            if u_1 < 0 and u_2 > 1:

                activated_edge_ids.append(edge)

            elif u_2 < 0 and u_1 > 1:

                activated_edge_ids.append(edge)

            # Line segment intersects at one point

            elif (0 <= u_1 <= 1) and (u_2 > 1 or u_2 < 0):

                activated_edge_ids.append(edge)

            elif (0 <= u_2 <= 1) and (u_1 > 1 or u_1 < 0):

                activated_edge_ids.append(edge)

            # Line segment intersects at two points

            elif (0 <= u_2 <= 1) and (0 <= u_1 <= 1):

                activated_edge_ids.append(edge)

            # Line segment is tangential

            elif (u_1 == u_2) and (0 <= u_1 <= 1):

                activated_edge_ids.append(edge)

            else:

                continue

        else:

            continue

    for activated_edge in activated_edge_ids:

        if graph.es[activated_edge]['Type'] == 0 or graph.es[activated_edge]['Type'] == 3:

            graph.es[activated_edge]['Activated'] = 1

    # Create CSV

    time_str = time.strftime("%Y%m%d_%H%M%S")

    activated_eids = []

    for edge in range(graph.ecount()):

        if graph.es[edge]['Activated'] == 1:
            activated_eids.append(edge)

    data = {'Activated': activated_eids}
    activated_eids_df = pd.DataFrame(data)
    activated_eids_df_new = activated_eids_df.T

    activated_eids_df_new.to_csv(path_csv_file + '\\activated_eids_temp.csv', index=False)

    with open(path_csv_file + '\\activated_eids_temp.csv', 'r') as f:
        with open(path_csv_file + '\\activated_eids_' + name_spec + '.csv', 'w') as f1:
            next(f)  # skip header line
            for line in f:
                f1.write(line)

    os.remove(path_csv_file + '\\activated_eids_temp.csv')

    return graph


def modifiable_all_different_types(graph, type_id):

    ''' Selects all edges of specific type '''

    # type id = 0: Capillaries
    # type_id = 1: Veins
    # type_id = 2: Arteries

    # Delete current selection

    for e in range(graph.ecount()):

        graph.es[e]['Modifiable'] = 0

    # Start new selection

    modifiable_eids = []

    for e in range(graph.ecount()):

        if type_id == 0:

            if graph.es[e]['Type'] == 0 or graph.es[e]['Type'] == 3:

                modifiable_eids.append(e)
                graph.es[e]['Modifiable'] = 1

        else:

            if graph.es[e]['Type'] == type_id:

                modifiable_eids.append(e)
                graph.es[e]['Modifiable'] = 1

    # Create CSV

    print(len(modifiable_eids))

    for edge in range(graph.ecount()):

        if graph.es[edge]['Modifiable'] == 1:
            modifiable_eids.append(edge)

    if type_id == 0:

        np.savetxt(path_csv_file + 'reacting_eids_all_cap_only.txt', [modifiable_eids], delimiter=',', fmt='%d')

    elif type_id == 1:

        np.savetxt(path_csv_file + 'reacting_eids_all_veins_only.txt', [modifiable_eids], delimiter=',', fmt='%d')

    elif type_id == 2:

        np.savetxt(path_csv_file + 'reacting_eids_all_arteries_only.txt', [modifiable_eids], delimiter=',', fmt='%d')

    return graph


def modifiable_region(graph, coords_sphere, r_sphere, path_csv_file):

    # Delete current selection

    for e in range(graph.ecount()):

        graph.es[e]['Modifiable'] = 0

    # Start new selection

    modifiable_edges = []

    for edge in range(graph.ecount()):

        p1 = graph.es[edge].source
        x_1 = graph.vs[p1]['x_coordinate'] / math.pow(10, 6)
        y_1 = graph.vs[p1]['y_coordinate'] / math.pow(10, 6)
        z_1 = graph.vs[p1]['z_coordinate'] / math.pow(10, 6)

        p2 = graph.es[edge].target
        x_2 = graph.vs[p2]['x_coordinate'] / math.pow(10, 6)
        y_2 = graph.vs[p2]['y_coordinate'] / math.pow(10, 6)
        z_2 = graph.vs[p2]['z_coordinate'] / math.pow(10, 6)

        x_3 = coords_sphere['x'] / math.pow(10, 6)
        y_3 = coords_sphere['y'] / math.pow(10, 6)
        z_3 = coords_sphere['z'] / math.pow(10, 6)

        radius = r_sphere / math.pow(10, 6)

        a = math.pow(x_2 - x_1, 2) + math.pow(y_2 - y_1, 2) + math.pow(z_2 - z_1, 2)
        b = 2 * ((x_2 - x_1)*(x_1 - x_3) + (y_2 - y_1)*(y_1 - y_3) + (z_2 - z_1)*(z_1 - z_3))
        c = math.pow(x_3, 2) + math.pow(y_3, 2) + math.pow(z_3, 2) + math.pow(x_1, 2) + math.pow(y_1, 2) + math.pow(z_1, 2) - 2 * (x_3 * x_1 + y_3 * y_1 + z_3 * z_1) - math.pow(radius, 2)

        value = math.pow(b, 2) - 4 * a * c

        if value >= 0:

            u_1 = (-b + math.sqrt(value)) / (2 * a)
            u_2 = (-b - math.sqrt(value)) / (2 * a)

            # Line segment doesnt intersect but is inside sphere

            if u_1 < 0 and u_2 > 1:

                modifiable_edges.append(edge)

            elif u_2 < 0 and u_1 > 1:

                modifiable_edges.append(edge)

            # Line segment intersects at one point

            elif (0 <= u_1 <= 1) and (u_2 > 1 or u_2 < 0):

                modifiable_edges.append(edge)

            elif (0 <= u_2 <= 1) and (u_1 > 1 or u_1 < 0):

                modifiable_edges.append(edge)

            # Line segment intersects at two points

            elif (0 <= u_2 <= 1) and (0 <= u_1 <= 1):

                modifiable_edges.append(edge)

            # Line segment is tangential

            elif (u_1 == u_2) and (0 <= u_1 <= 1):

                modifiable_edges.append(edge)

            else:

                continue

        else:

            continue

    for k in modifiable_edges:

        if graph.es[k]['Type'] == 0 or graph.es[k]['Type'] == 3:

            graph.es[k]['Modifiable'] = 1

    # Create CSV

    np.savetxt(path_csv_file + 'reacting_eids_all_radius_' + str(r_sphere) + '.txt', [modifiable_edges], delimiter=',',
               fmt='%d')

    return graph


def plot_chosen_region(graph, path):

    # Graph need to have coordinates of each node as vertex attribute

    mpl.rcParams['legend.fontsize'] = 10

    fig = plt.figure()
    ax = fig.gca(projection='3d')

    for edge in range(graph.ecount()):

        x = []
        y = []
        z = []

        x.append(graph.vs[graph.es[edge].source]['x_coordinate'])
        x.append(graph.vs[graph.es[edge].target]['x_coordinate'])

        y.append(graph.vs[graph.es[edge].source]['y_coordinate'])
        y.append(graph.vs[graph.es[edge].target]['y_coordinate'])

        z.append(graph.vs[graph.es[edge].source]['z_coordinate'])
        z.append(graph.vs[graph.es[edge].target]['z_coordinate'])

        if graph.es[edge]['Activated'] == 1:

            ax.plot(x, y, z, color='green')

        elif graph.es[edge]['Modifiable'] ==1:

            ax.plot(x, y, z, color='red')

        else:

            ax.plot(x, y, z, color='lightgray', alpha=0.5)

    plt.xlabel('X')
    plt.ylabel('Y')

    # plt.savefig(path)
    plt.show()

    return graph


def plot_chosen_region_activated(graph, path):

    # Graph need to have coordinates of each node as vertex attribute

    mpl.rcParams['legend.fontsize'] = 10

    fig = plt.figure()
    ax = fig.gca(projection='3d')

    for edge in range(graph.ecount()):

        x = []
        y = []
        z = []

        x.append(graph.vs[graph.es[edge].source]['x_coordinate'])
        x.append(graph.vs[graph.es[edge].target]['x_coordinate'])

        y.append(graph.vs[graph.es[edge].source]['y_coordinate'])
        y.append(graph.vs[graph.es[edge].target]['y_coordinate'])

        z.append(graph.vs[graph.es[edge].source]['z_coordinate'])
        z.append(graph.vs[graph.es[edge].target]['z_coordinate'])

        if graph.es[edge]['Activated'] == 1:

            ax.plot(x, y, z, color='green')

        else:

            # ax.plot(x, y, z, color='lightgray', alpha=0.5)
            continue

    plt.xlabel('X')
    plt.ylabel('Y')

    # plt.savefig(path)
    plt.show()

    return graph


# Functions ALL and ALL CAPS only

def distance_plot(coord_mittelpunkt, segments):

    return


########################################################################################################################

# SMALL NETWORKS
#
# path = 'D:\\00 Privat\\01_Bildung\\01_ETH Zürich\MSc\\00_Masterarbeit\\02_Network_Study_Small\\Networks\\' \
#        + str(8) + '\\graph.pkl'
#
# path_csv_file = 'D:\\00 Privat\\01_Bildung\\01_ETH Zürich\MSc\\00_Masterarbeit\\02_Network_Study_Small\\Networks\\' \
#        + str(8) + '\\adjointMethod\\'
#
# graph_ = ig.Graph.Read_Pickle(path)
# coor = {'x': 200, 'y': 400, 'z': 400}
# types = [0, 1, 2]
# radii = [100, 125, 150, 175, 200, 300]
#
# graph_ = define_activated_region(graph_, coor, 80, path_csv_file)
#
# for t in types:
#
#     graph_ = modifiable_all_different_types(graph_, t)
#
# for radius in radii:
#
#     graph_ = modifiable_region(graph_, coor, radius, path_csv_file)
#
# # plot_chosen_region(graph_, 10)


########################################################################################################################


# LARGE NETWORKS

path = 'D:\\00 Privat\\01_Bildung\\01_ETH Zürich\MSc\\00_Masterarbeit\\03_Network_Study_Large\\Networks\\' \
       + str(1) + '\\graph.pkl'

path_csv_file = 'D:\\00 Privat\\01_Bildung\\01_ETH Zürich\MSc\\00_Masterarbeit\\03_Network_Study_Large\\Networks\\' \
       + str(1) + '\\adjointMethod\\'

# Höheneinfluss

coordinates_hohe = [[750, 1000, 100], [750, 1000, 250], [750, 1000, 400], [750, 1000, 650]]
names = ['h_100', 'h_250', 'h_400', 'h_650']

# Lageneinfluss

coordinates_lage = [[250, 600, 400], [250, 1500, 400], [1100, 600, 400], [1100, 1500, 400]]
names_lage = ['l_1', 'l_2', 'l_3', 'l_4']
r = 120

for i in range(len(coordinates_hohe)):

    print(coordinates_hohe[i])
    print(names[i])

    graph_ = ig.Graph.Read_Pickle(path)
    coor = {'x': coordinates_hohe[i][0], 'y': coordinates_hohe[i][1], 'z': coordinates_hohe[i][2]}

    graph_ = define_activated_region_large_networks(graph_, coor, r, path_csv_file, names[i])
    # plot_chosen_region_activated(graph_, 10)

for i in range(len(coordinates_lage)):

    print(coordinates_lage[i])
    print(names_lage[i])

    graph_ = ig.Graph.Read_Pickle(path)
    coor = {'x': coordinates_lage[i][0], 'y': coordinates_lage[i][1], 'z': coordinates_lage[i][2]}

    graph_ = define_activated_region_large_networks(graph_, coor, r, path_csv_file, names_lage[i])
    # plot_chosen_region_activated(graph_, 10)

types = [0, 2]
graph_ = ig.Graph.Read_Pickle(path)

for t in types:

    graph_ = modifiable_all_different_types(graph_, t)
