import math
import igraph as ig
import matplotlib as mpl
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import time
import pandas as pd
import os


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

    modifiable_eids = []

    for edge in range(graph.ecount()):

        if graph.es[edge]['Modifiable'] == 1:
            modifiable_eids.append(edge)

    data = {'Modifiable': modifiable_eids}
    modifiable_eids_df = pd.DataFrame(data)
    modifiable_eids_df_new = modifiable_eids_df.T

    modifiable_eids_df_new.to_csv(path_csv_file + '\\reacting_eids_temp.csv', index=False)

    with open(path_csv_file + '\\reacting_eids_temp.csv', 'r') as f:
        with open(path_csv_file + '\\reacting_eids.csv', 'w') as f1:
            next(f)  # skip header line
            for line in f:
                f1.write(line)

    os.remove(path_csv_file + '\\reacting_eids_temp.csv')

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


path = 'D:\\00 Privat\\01_Bildung\\01_ETH Zürich\MSc\\00_Masterarbeit\\02_Network_Study_Small\\Networks\\' \
       + str(0) + '\\graph.pkl'

path_csv_file = 'D:\\00 Privat\\01_Bildung\\01_ETH Zürich\MSc\\00_Masterarbeit\\02_Network_Study_Small\\Networks\\' \
       + str(0) + '\\adjointMethod\\'

graph_ = ig.Graph.Read_Pickle(path)
coor = {'x': 100, 'y': 200, 'z': 100}

graph_new = define_activated_region(graph_, coor, 100, path_csv_file)
graph_new_1 = modifiable_region(graph_, coor, 200, path_csv_file)
plot_chosen_region(graph_new, 10)
