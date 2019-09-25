import pandas as pd
import numpy as np
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 unused import
import matplotlib.pyplot as plt
import csv
import math
import os
import igraph as ig

def define_activated_region(graph, coords_sphere, r_sphere):

    edges_in_current_region = []

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

                edges_in_current_region.append(edge)

            elif u_2 < 0 and u_1 > 1:

                edges_in_current_region.append(edge)

            # Line segment intersects at one point

            elif (0 <= u_1 <= 1) and (u_2 > 1 or u_2 < 0):

                edges_in_current_region.append(edge)

            elif (0 <= u_2 <= 1) and (u_1 > 1 or u_1 < 0):

                edges_in_current_region.append(edge)

            # Line segment intersects at two points

            elif (0 <= u_2 <= 1) and (0 <= u_1 <= 1):

                edges_in_current_region.append(edge)

            # Line segment is tangential

            elif (u_1 == u_2) and (0 <= u_1 <= 1):

                edges_in_current_region.append(edge)

            else:

                continue

        else:

            continue

    return edges_in_current_region



def plot_distance():

    path = r'D:\00 Privat\01_Bildung\01_ETH Zürich\MSc\00_Masterarbeit\02_Network_Study_Small\Adjoint'

    path_0_graph = r'D:\00 Privat\01_Bildung\01_ETH Zürich\MSc\00_Masterarbeit\02_Network_Study_Small\Networks\0\graph.pkl'
    graph_ = ig.Graph.Read_Pickle(path_0_graph)
    coord = {'x': 200, 'y': 400, 'z': 400}

    radius_min = 10.0
    radius_max = 300.0
    r_steps = 100
    delta_radius = radius_max - radius_min
    step_radius = delta_radius / r_steps

    radii = np.arange(radius_min, radius_max, step_radius)

    diameter_change = []

    print(radii)
    print(len(radii))
    for radius in radii:
        print(radius)
        current_selection = define_activated_region(graph_, coord, radius)
        print(len(current_selection))
        print(current_selection)

    # networks = ['0_Out_', '3_Out_', '4_Out_', '6_Out_']
    networks = ['0_Out_']
    histogram_data_id = ['All_Cap']
    # histogram_data_id = ['All']

    data = []

    for network in networks:

        graph_ = ig.Graph()

        if network == '0_Out_':

            graph_ = ig.Graph.Read_Pickle(path_0_graph)

        print(network)

        for radius in histogram_data_id:
            print(radius)

            current_input_path = path + '\\' + network + '\\' + radius + '\\' + 'in\\adjointMethod' + '\\'
            current_output_path = path + '\\' + network + '\\' + radius + '\\' + 'out' + '\\'
            meshdata_start = current_output_path + 'meshdata_9999.csv'

            files = []
            # r=root, d=directories, f = files
            for r, d, f in os.walk(current_output_path):
                for file in f:

                    if file[0:8] == 'meshdata':
                        files.append(file)

            files.remove('meshdata_9999.csv')
            meshdata_final = current_output_path + files[0]

            df_start = pd.read_csv(meshdata_start)
            df_final = pd.read_csv(meshdata_final)

            t_1 = np.array(df_start['D'])
            t_2 = np.array(df_final['D'])

            t = t_2 / t_1
            t_new = []

            for i in range(len(t)):

                if (i % 2) == 0:
                    t_new.append(t[i])

            t_new_array = np.asarray(t_new)
            reacting_eids = []

            if radius == 'All':

                capillaries_eids = []

                for i in range(graph_.ecount()):

                    if graph_.es[i]['Type'] == 1 or graph_.es[i]['Type'] == 2:
                        capillaries_eids.append(i)

                diameter_changes_reacting_edges = t_new_array[capillaries_eids]
                diameter_changes_reacting_edges_percent = np.abs((diameter_changes_reacting_edges - 1) * 100)

            else:

                with open(current_input_path + 'reacting_eids.csv', newline='') as f:
                    reader = csv.reader(f)
                    row1 = next(reader)  # gets the first line
                    row2 = next(f)
                    reacting_eids = list(map(int, row2.split(',')))

                diameter_changes_reacting_edges = t_new_array[reacting_eids]
                diameter_changes_reacting_edges_percent = np.abs((diameter_changes_reacting_edges - 1) * 100)

            # HIER KANN BEGRENZUNG EINGEBAUEN WERDEN

            grenze_prozentuela_anderung = percent
            diameter_changes_selection_index = np.where(
                diameter_changes_reacting_edges_percent > grenze_prozentuela_anderung)
            diameter_changes_selection_FINAL = (
                diameter_changes_reacting_edges_percent[diameter_changes_selection_index[0]])

            data.append(diameter_changes_selection_FINAL)

    return data



    return


plot_distance()