import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Polygon
import os
import pandas as pd
import csv
import igraph as ig
from collections import OrderedDict
import math
import seaborn as sns

########################################################################################################################
#
# DIAMETER BOXPLOTS (R100 - ALL)
#
########################################################################################################################

# Box Plot Diameter All Vessel Types


def get_diameter_data_for_different_networks(percent):

    path = r'D:\00 Privat\01_Bildung\01_ETH Zürich\MSc\00_Masterarbeit\02_Network_Study_Small\Adjoint'

    networks = ['0_Out_', '3_Out_', '4_Out_', '6_Out_']
    # networks = ['0_Out_']
    histogram_data_id = ['R_100', 'R_125', 'R_150', 'R_200', 'All']
    # histogram_data_id = ['All']

    data_network_0 = {}
    data_network_3 = {}
    data_network_4 = {}
    data_network_6 = {}

    data = {'0_Out_': {}, '3_Out_': {}, '4_Out_': {}, '6_Out_': {}, '0_Out_Tot': {}, '3_Out_Tot': {}, '4_Out_Tot': {},
            '6_Out_Tot': {}}

    for network in networks:

        print(network)

        for radius in histogram_data_id:

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
            data[network + 'Tot']['Number_Vessels'] = len(t_new_array)
            reacting_eids = []

            if radius == 'All':

                diameter_changes_reacting_edges = t_new_array
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

            # print(diameter_changes_selection_FINAL)
            data[network][radius] = diameter_changes_selection_FINAL


    # print(len(data_r_100), data_r_100)
    # print(len(data_r_150), data_r_150)
    # print(len(data_r_200), data_r_200)
    # print(len(data_r_300), data_r_300)
    # print(len(data_r_all), data_r_all)

    return data


def plot_diameter_all_vessel_types_NEW(prozent):

    data = get_diameter_data_for_different_networks(prozent)

    df = pd.DataFrame(columns=['Group', 'Network 1', 'Network 2', 'Network 3', 'Network 4', ''])

    number_affected_vessels_100 = [len(data['0_Out_']['R_100']), len(data['3_Out_']['R_100']),
                                   len(data['4_Out_']['R_100']), len(data['6_Out_']['R_100'])]

    number_affected_vessels_125 = [len(data['0_Out_']['R_125']), len(data['3_Out_']['R_125']),
                                   len(data['4_Out_']['R_125']), len(data['6_Out_']['R_125'])]

    number_affected_vessels_150 = [len(data['0_Out_']['R_150']), len(data['3_Out_']['R_150']),
                                   len(data['4_Out_']['R_150']), len(data['6_Out_']['R_150'])]

    number_affected_vessels_200 = [len(data['0_Out_']['R_200']), len(data['3_Out_']['R_200']),
                                   len(data['4_Out_']['R_200']), len(data['6_Out_']['R_200'])]

    number_affected_vessels_all = [len(data['0_Out_']['All']), len(data['3_Out_']['All']),
                                   len(data['4_Out_']['All']), len(data['6_Out_']['All'])]

    number_affecte_vessels = [number_affected_vessels_100, number_affected_vessels_125, number_affected_vessels_150,
                              number_affected_vessels_200, number_affected_vessels_all]

    # R_100

    for i in range(len(data['0_Out_']['R_100'])):

        x1 = data['0_Out_']['R_100'][i]
        df = df.append({'Group': 'R100', 'Network 1': x1},
                       ignore_index=True)

    for i in range(len(data['3_Out_']['R_100'])):

        x1 = data['3_Out_']['R_100'][i]
        df = df.append({'Group': 'R100', 'Network 2': x1},
                       ignore_index=True)

    for i in range(len(data['4_Out_']['R_100'])):

        x1 = data['4_Out_']['R_100'][i]
        df = df.append({'Group': 'R100', 'Network 3': x1},
                       ignore_index=True)

    for i in range(len(data['6_Out_']['R_100'])):

        x1 = data['6_Out_']['R_100'][i]
        df = df.append({'Group': 'R100', 'Network 4': x1},
                       ignore_index=True)

     # R 125

    for i in range(len(data['0_Out_']['R_125'])):
        x1 = data['0_Out_']['R_125'][i]
        df = df.append({'Group': 'R125', 'Network 1': x1},
                       ignore_index=True)

    for i in range(len(data['3_Out_']['R_125'])):
        x1 = data['3_Out_']['R_125'][i]
        df = df.append({'Group': 'R125', 'Network 2': x1},
                       ignore_index=True)

    for i in range(len(data['4_Out_']['R_125'])):
        x1 = data['4_Out_']['R_125'][i]
        df = df.append({'Group': 'R125', 'Network 3': x1},
                       ignore_index=True)

    for i in range(len(data['6_Out_']['R_125'])):
        x1 = data['6_Out_']['R_125'][i]
        df = df.append({'Group': 'R125', 'Network 4': x1},
                       ignore_index=True)

     # R 150

    for i in range(len(data['0_Out_']['R_150'])):
        x1 = data['0_Out_']['R_150'][i]
        df = df.append({'Group': 'R150', 'Network 1': x1},
                       ignore_index=True)

    for i in range(len(data['3_Out_']['R_150'])):
        x1 = data['3_Out_']['R_150'][i]
        df = df.append({'Group': 'R150', 'Network 2': x1},
                       ignore_index=True)

    for i in range(len(data['4_Out_']['R_150'])):
        x1 = data['4_Out_']['R_150'][i]
        df = df.append({'Group': 'R150', 'Network 3': x1},
                       ignore_index=True)

    for i in range(len(data['6_Out_']['R_150'])):
        x1 = data['6_Out_']['R_150'][i]
        df = df.append({'Group': 'R150', 'Network 4': x1},
                       ignore_index=True)

    # R 200

    for i in range(len(data['0_Out_']['R_200'])):

        x1 = data['0_Out_']['R_200'][i]
        df = df.append({'Group': 'R200', 'Network 1': x1},
                       ignore_index=True)

    for i in range(len(data['3_Out_']['R_200'])):

        x1 = data['3_Out_']['R_200'][i]
        df = df.append({'Group': 'R200', 'Network 2': x1},
                       ignore_index=True)

    for i in range(len(data['4_Out_']['R_200'])):

        x1 = data['4_Out_']['R_200'][i]
        df = df.append({'Group': 'R200', 'Network 3': x1},
                       ignore_index=True)

    for i in range(len(data['6_Out_']['R_200'])):

        x1 = data['6_Out_']['R_200'][i]
        df = df.append({'Group': 'R200', 'Network 4': x1},
                       ignore_index=True)

    # All

    for i in range(len(data['0_Out_']['All'])):
        x1 = data['0_Out_']['All'][i]
        df = df.append({'Group': 'All', 'Network 1': x1},
                       ignore_index=True)

    for i in range(len(data['3_Out_']['All'])):
        x1 = data['3_Out_']['All'][i]
        df = df.append({'Group': 'All', 'Network 2': x1},
                       ignore_index=True)

    for i in range(len(data['4_Out_']['All'])):
        x1 = data['4_Out_']['All'][i]
        df = df.append({'Group': 'All', 'Network 3': x1},
                       ignore_index=True)

    for i in range(len(data['6_Out_']['All'])):
        x1 = data['6_Out_']['All'][i]
        df = df.append({'Group': 'All', 'Network 4': x1},
                       ignore_index=True)

    print(df)
    dd = pd.melt(df, id_vars=['Group'], value_vars=['Network 1', 'Network 2', 'Network 3', 'Network 4'], var_name='Networks')
    plt.figure(figsize=(14, 10))
    testPlot = sns.boxplot(x='Group', y='value', data=dd, hue='Networks', showmeans=True, fliersize=1,
                    meanprops={"marker": "+", "markersize": 6, "markerfacecolor": "black", "markeredgecolor": "black"})

    # Shrink current axis by 20%
    box = testPlot.get_position()
    testPlot.set_position([box.x0, box.y0, box.width * 0.8, box.height])

    # Put a legend to the right of the current axis
    testPlot.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    testPlot.set_ylim(0, 18)

    ind = 0
    for tick in range(len(testPlot.get_xticklabels())):

        print(tick)
        testPlot.text(tick - .1, 17.5, str(np.round(number_affecte_vessels[tick][1] / data['3_Out_Tot']['Number_Vessels'] * 100, 2)) + '\%', horizontalalignment='center', color='orange', rotation=70, size=10)
        testPlot.text(tick + .1, 17.5, str(np.round(number_affecte_vessels[tick][2] / data['4_Out_Tot']['Number_Vessels'] * 100, 2)) + '\%', horizontalalignment='center', color='green', rotation=70, size=10)
        testPlot.text(tick + .3, 17.5, str(np.round(number_affecte_vessels[tick][3] / data['6_Out_Tot']['Number_Vessels'] * 100, 2)) + '\%', horizontalalignment='center', color='red', rotation=70, size=10)
        testPlot.text(tick - .3, 17.5, str(np.round(number_affecte_vessels[tick][0] / data['0_Out_Tot']['Number_Vessels'] * 100, 2)) + '\%', horizontalalignment='center', color='blue', rotation=70, size=10)
        ind += 2

    # plt.legend(loc='upper right')
    plt.savefig(
        r'D:\00 Privat\01_Bildung\01_ETH Zürich\MSc\00_Masterarbeit\02_Network_Study_Small\Adjoint\Diameter_Change_R100_to_ALL\boxplot_diameter_all_types.png')


# Box Plot Diameter Capillaries


def get_diameter_data_for_different_networks_capillaries(percent):

    path = r'D:\00 Privat\01_Bildung\01_ETH Zürich\MSc\00_Masterarbeit\02_Network_Study_Small\Adjoint'

    path_0_graph = r'D:\00 Privat\01_Bildung\01_ETH Zürich\MSc\00_Masterarbeit\02_Network_Study_Small\Networks\0\graph.pkl'
    path_3_graph = r'D:\00 Privat\01_Bildung\01_ETH Zürich\MSc\00_Masterarbeit\02_Network_Study_Small\Networks\3\graph.pkl'
    path_4_graph = r'D:\00 Privat\01_Bildung\01_ETH Zürich\MSc\00_Masterarbeit\02_Network_Study_Small\Networks\4\graph.pkl'
    path_6_graph = r'D:\00 Privat\01_Bildung\01_ETH Zürich\MSc\00_Masterarbeit\02_Network_Study_Small\Networks\6\graph.pkl'

    networks = ['0_Out_', '3_Out_', '4_Out_', '6_Out_']
    # networks = ['0_Out_']
    histogram_data_id = ['R_100', 'R_125', 'R_150', 'R_200', 'All']
    # histogram_data_id = ['All']

    data_network_0 = {}
    data_network_3 = {}
    data_network_4 = {}
    data_network_6 = {}

    data = {'0_Out_': {}, '3_Out_': {}, '4_Out_': {}, '6_Out_': {}, '0_Out_Tot': {}, '3_Out_Tot': {}, '4_Out_Tot': {},
            '6_Out_Tot': {}}

    for network in networks:

        graph_ = ig.Graph()

        if network == '0_Out_':

            graph_ = ig.Graph.Read_Pickle(path_0_graph)

        elif network == '3_Out_':

            graph_ = ig.Graph.Read_Pickle(path_3_graph)

        elif network == '4_Out_':

            graph_ = ig.Graph.Read_Pickle(path_4_graph)

        elif network == '6_Out_':

            graph_ = ig.Graph.Read_Pickle(path_6_graph)

        print(network)

        for radius in histogram_data_id:

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

            counter = 0

            for edge in range(graph_.ecount()):

                if graph_.es[edge]['Type'] == 0 or graph_.es[edge]['Type'] == 3:
                    counter += 1

            data[network + 'Tot']['Number_Vessels'] = counter
            reacting_eids = []

            if radius == 'All':

                reacting_eids_new = []

                for edge in range(graph_.ecount()):

                    if graph_.es[edge]['Type'] == 0 or graph_.es[edge]['Type'] == 3:
                        reacting_eids_new.append(edge)

                diameter_changes_reacting_edges = t_new_array[reacting_eids_new]
                diameter_changes_reacting_edges_percent = np.abs((diameter_changes_reacting_edges - 1) * 100)

            else:

                with open(current_input_path + 'reacting_eids.csv', newline='') as f:
                    reader = csv.reader(f)
                    row1 = next(reader)  # gets the first line
                    row2 = next(f)
                    reacting_eids = list(map(int, row2.split(',')))
                print('reacting_eids ', len(reacting_eids), reacting_eids)
                reacting_eids_new = []

                for reacting_eid in reacting_eids:

                    if graph_.es[reacting_eid]['Type'] == 0 or graph_.es[reacting_eid]['Type'] == 3:
                        reacting_eids_new.append(reacting_eid)

                print('reacting NEW ', len(reacting_eids_new), reacting_eids_new)
                diameter_changes_reacting_edges = t_new_array[reacting_eids_new]
                diameter_changes_reacting_edges_percent = np.abs((diameter_changes_reacting_edges - 1) * 100)

            # HIER KANN BEGRENZUNG EINGEBAUEN WERDEN

            grenze_prozentuela_anderung = percent
            diameter_changes_selection_index = np.where(
                diameter_changes_reacting_edges_percent > grenze_prozentuela_anderung)
            diameter_changes_selection_FINAL = (
            diameter_changes_reacting_edges_percent[diameter_changes_selection_index[0]])

            # print(diameter_changes_selection_FINAL)
            data[network][radius] = diameter_changes_selection_FINAL


    # print(len(data_r_100), data_r_100)
    # print(len(data_r_150), data_r_150)
    # print(len(data_r_200), data_r_200)
    # print(len(data_r_300), data_r_300)
    # print(len(data_r_all), data_r_all)

    return data


def plot_diameter_capillaries_types_NEW(prozent):

    data = get_diameter_data_for_different_networks_capillaries(prozent)

    df = pd.DataFrame(columns=['Group', 'Network 1', 'Network 2', 'Network 3', 'Network 4', ''])

    number_affected_vessels_100 = [len(data['0_Out_']['R_100']), len(data['3_Out_']['R_100']),
                                   len(data['4_Out_']['R_100']), len(data['6_Out_']['R_100'])]

    number_affected_vessels_125 = [len(data['0_Out_']['R_125']), len(data['3_Out_']['R_125']),
                                   len(data['4_Out_']['R_125']), len(data['6_Out_']['R_125'])]

    number_affected_vessels_150 = [len(data['0_Out_']['R_150']), len(data['3_Out_']['R_150']),
                                   len(data['4_Out_']['R_150']), len(data['6_Out_']['R_150'])]

    number_affected_vessels_200 = [len(data['0_Out_']['R_200']), len(data['3_Out_']['R_200']),
                                   len(data['4_Out_']['R_200']), len(data['6_Out_']['R_200'])]

    number_affected_vessels_all = [len(data['0_Out_']['All']), len(data['3_Out_']['All']),
                                   len(data['4_Out_']['All']), len(data['6_Out_']['All'])]

    number_affecte_vessels = [number_affected_vessels_100, number_affected_vessels_125, number_affected_vessels_150,
                              number_affected_vessels_200, number_affected_vessels_all]

    # R_100

    for i in range(len(data['0_Out_']['R_100'])):

        x1 = data['0_Out_']['R_100'][i]
        df = df.append({'Group': 'R100', 'Network 1': x1},
                       ignore_index=True)

    for i in range(len(data['3_Out_']['R_100'])):

        x1 = data['3_Out_']['R_100'][i]
        df = df.append({'Group': 'R100', 'Network 2': x1},
                       ignore_index=True)

    for i in range(len(data['4_Out_']['R_100'])):

        x1 = data['4_Out_']['R_100'][i]
        df = df.append({'Group': 'R100', 'Network 3': x1},
                       ignore_index=True)

    for i in range(len(data['6_Out_']['R_100'])):

        x1 = data['6_Out_']['R_100'][i]
        df = df.append({'Group': 'R100', 'Network 4': x1},
                       ignore_index=True)

     # R 125

    for i in range(len(data['0_Out_']['R_125'])):
        x1 = data['0_Out_']['R_125'][i]
        df = df.append({'Group': 'R125', 'Network 1': x1},
                       ignore_index=True)

    for i in range(len(data['3_Out_']['R_125'])):
        x1 = data['3_Out_']['R_125'][i]
        df = df.append({'Group': 'R125', 'Network 2': x1},
                       ignore_index=True)

    for i in range(len(data['4_Out_']['R_125'])):
        x1 = data['4_Out_']['R_125'][i]
        df = df.append({'Group': 'R125', 'Network 3': x1},
                       ignore_index=True)

    for i in range(len(data['6_Out_']['R_125'])):
        x1 = data['6_Out_']['R_125'][i]
        df = df.append({'Group': 'R125', 'Network 4': x1},
                       ignore_index=True)

     # R 150

    for i in range(len(data['0_Out_']['R_150'])):
        x1 = data['0_Out_']['R_150'][i]
        df = df.append({'Group': 'R150', 'Network 1': x1},
                       ignore_index=True)

    for i in range(len(data['3_Out_']['R_150'])):
        x1 = data['3_Out_']['R_150'][i]
        df = df.append({'Group': 'R150', 'Network 2': x1},
                       ignore_index=True)

    for i in range(len(data['4_Out_']['R_150'])):
        x1 = data['4_Out_']['R_150'][i]
        df = df.append({'Group': 'R150', 'Network 3': x1},
                       ignore_index=True)

    for i in range(len(data['6_Out_']['R_150'])):
        x1 = data['6_Out_']['R_150'][i]
        df = df.append({'Group': 'R150', 'Network 4': x1},
                       ignore_index=True)

    # R 200

    for i in range(len(data['0_Out_']['R_200'])):

        x1 = data['0_Out_']['R_200'][i]
        df = df.append({'Group': 'R200', 'Network 1': x1},
                       ignore_index=True)

    for i in range(len(data['3_Out_']['R_200'])):

        x1 = data['3_Out_']['R_200'][i]
        df = df.append({'Group': 'R200', 'Network 2': x1},
                       ignore_index=True)

    for i in range(len(data['4_Out_']['R_200'])):

        x1 = data['4_Out_']['R_200'][i]
        df = df.append({'Group': 'R200', 'Network 3': x1},
                       ignore_index=True)

    for i in range(len(data['6_Out_']['R_200'])):

        x1 = data['6_Out_']['R_200'][i]
        df = df.append({'Group': 'R200', 'Network 4': x1},
                       ignore_index=True)

    # All

    for i in range(len(data['0_Out_']['All'])):
        x1 = data['0_Out_']['All'][i]
        df = df.append({'Group': 'All', 'Network 1': x1},
                       ignore_index=True)

    for i in range(len(data['3_Out_']['All'])):
        x1 = data['3_Out_']['All'][i]
        df = df.append({'Group': 'All', 'Network 2': x1},
                       ignore_index=True)

    for i in range(len(data['4_Out_']['All'])):
        x1 = data['4_Out_']['All'][i]
        df = df.append({'Group': 'All', 'Network 3': x1},
                       ignore_index=True)

    for i in range(len(data['6_Out_']['All'])):
        x1 = data['6_Out_']['All'][i]
        df = df.append({'Group': 'All', 'Network 4': x1},
                       ignore_index=True)

    print(df)
    dd = pd.melt(df, id_vars=['Group'], value_vars=['Network 1', 'Network 2', 'Network 3', 'Network 4'], var_name='Networks')
    plt.figure(figsize=(14, 10))
    testPlot = sns.boxplot(x='Group', y='value', data=dd, hue='Networks', showmeans=True, fliersize=1,
                    meanprops={"marker": "+", "markersize": 6, "markerfacecolor": "black", "markeredgecolor": "black"})

    # Shrink current axis by 20%
    box = testPlot.get_position()
    testPlot.set_position([box.x0, box.y0, box.width * 0.8, box.height])

    # Put a legend to the right of the current axis
    testPlot.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    testPlot.set_ylim(0, 18)

    ind = 0
    for tick in range(len(testPlot.get_xticklabels())):

        print(tick)
        testPlot.text(tick - .1, 17.5, str(np.round(number_affecte_vessels[tick][1] / data['3_Out_Tot']['Number_Vessels'] * 100, 2)) + '\%', horizontalalignment='center', color='orange', rotation=70, size=10)
        testPlot.text(tick + .1, 17.5, str(np.round(number_affecte_vessels[tick][2] / data['4_Out_Tot']['Number_Vessels'] * 100, 2)) + '\%', horizontalalignment='center', color='green', rotation=70, size=10)
        testPlot.text(tick + .3, 17.5, str(np.round(number_affecte_vessels[tick][3] / data['6_Out_Tot']['Number_Vessels'] * 100, 2)) + '\%', horizontalalignment='center', color='red', rotation=70, size=10)
        testPlot.text(tick - .3, 17.5, str(np.round(number_affecte_vessels[tick][0] / data['0_Out_Tot']['Number_Vessels'] * 100, 2)) + '\%', horizontalalignment='center', color='blue', rotation=70, size=10)
        ind += 2

    # plt.legend(loc='upper right')
    plt.savefig(
        r'D:\00 Privat\01_Bildung\01_ETH Zürich\MSc\00_Masterarbeit\02_Network_Study_Small\Adjoint\Diameter_Change_R100_to_ALL\boxplot_diameter_capillaries.png')


# Box Plot Diameter Venules


def get_diameter_data_for_different_networks_venules(percent):

    path = r'D:\00 Privat\01_Bildung\01_ETH Zürich\MSc\00_Masterarbeit\02_Network_Study_Small\Adjoint'

    path_0_graph = r'D:\00 Privat\01_Bildung\01_ETH Zürich\MSc\00_Masterarbeit\02_Network_Study_Small\Networks\0\graph.pkl'
    path_3_graph = r'D:\00 Privat\01_Bildung\01_ETH Zürich\MSc\00_Masterarbeit\02_Network_Study_Small\Networks\3\graph.pkl'
    path_4_graph = r'D:\00 Privat\01_Bildung\01_ETH Zürich\MSc\00_Masterarbeit\02_Network_Study_Small\Networks\4\graph.pkl'
    path_6_graph = r'D:\00 Privat\01_Bildung\01_ETH Zürich\MSc\00_Masterarbeit\02_Network_Study_Small\Networks\6\graph.pkl'

    networks = ['0_Out_', '3_Out_', '4_Out_', '6_Out_']
    # networks = ['0_Out_']
    histogram_data_id = ['R_100', 'R_125', 'R_150', 'R_200', 'All']
    # histogram_data_id = ['All']

    data_network_0 = {}
    data_network_3 = {}
    data_network_4 = {}
    data_network_6 = {}

    data = {'0_Out_': {}, '3_Out_': {}, '4_Out_': {}, '6_Out_': {}, '0_Out_Tot': {}, '3_Out_Tot': {}, '4_Out_Tot': {},
            '6_Out_Tot': {}}

    for network in networks:

        graph_ = ig.Graph()

        if network == '0_Out_':

            graph_ = ig.Graph.Read_Pickle(path_0_graph)

        elif network == '3_Out_':

            graph_ = ig.Graph.Read_Pickle(path_3_graph)

        elif network == '4_Out_':

            graph_ = ig.Graph.Read_Pickle(path_4_graph)

        elif network == '6_Out_':

            graph_ = ig.Graph.Read_Pickle(path_6_graph)

        print(network)

        for radius in histogram_data_id:

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

            counter = 0

            for edge in range(graph_.ecount()):

                if graph_.es[edge]['Type'] == 1:
                    counter += 1

            data[network + 'Tot']['Number_Vessels'] = counter
            reacting_eids = []

            if radius == 'All':

                reacting_eids_new = []

                for edge in range(graph_.ecount()):

                    if graph_.es[edge]['Type'] == 1:
                        reacting_eids_new.append(edge)

                diameter_changes_reacting_edges = t_new_array[reacting_eids_new]
                diameter_changes_reacting_edges_percent = np.abs((diameter_changes_reacting_edges - 1) * 100)

            else:

                with open(current_input_path + 'reacting_eids.csv', newline='') as f:
                    reader = csv.reader(f)
                    row1 = next(reader)  # gets the first line
                    row2 = next(f)
                    reacting_eids = list(map(int, row2.split(',')))
                print('reacting_eids ', len(reacting_eids), reacting_eids)
                reacting_eids_new = []

                for reacting_eid in reacting_eids:

                    if graph_.es[reacting_eid]['Type'] == 1:
                        reacting_eids_new.append(reacting_eid)

                print('reacting NEW ', len(reacting_eids_new), reacting_eids_new)
                diameter_changes_reacting_edges = t_new_array[reacting_eids_new]
                diameter_changes_reacting_edges_percent = np.abs((diameter_changes_reacting_edges - 1) * 100)

            # HIER KANN BEGRENZUNG EINGEBAUEN WERDEN

            grenze_prozentuela_anderung = percent
            diameter_changes_selection_index = np.where(
                diameter_changes_reacting_edges_percent > grenze_prozentuela_anderung)
            diameter_changes_selection_FINAL = (
            diameter_changes_reacting_edges_percent[diameter_changes_selection_index[0]])

            # print(diameter_changes_selection_FINAL)
            data[network][radius] = diameter_changes_selection_FINAL


    # print(len(data_r_100), data_r_100)
    # print(len(data_r_150), data_r_150)
    # print(len(data_r_200), data_r_200)
    # print(len(data_r_300), data_r_300)
    # print(len(data_r_all), data_r_all)

    return data


def plot_diameter_venules_types_NEW(prozent):

    data = get_diameter_data_for_different_networks_venules(prozent)

    df = pd.DataFrame(columns=['Group', 'Network 1', 'Network 2', 'Network 3', 'Network 4', ''])

    number_affected_vessels_100 = [len(data['0_Out_']['R_100']), len(data['3_Out_']['R_100']),
                                   len(data['4_Out_']['R_100']), len(data['6_Out_']['R_100'])]

    number_affected_vessels_125 = [len(data['0_Out_']['R_125']), len(data['3_Out_']['R_125']),
                                   len(data['4_Out_']['R_125']), len(data['6_Out_']['R_125'])]

    number_affected_vessels_150 = [len(data['0_Out_']['R_150']), len(data['3_Out_']['R_150']),
                                   len(data['4_Out_']['R_150']), len(data['6_Out_']['R_150'])]

    number_affected_vessels_200 = [len(data['0_Out_']['R_200']), len(data['3_Out_']['R_200']),
                                   len(data['4_Out_']['R_200']), len(data['6_Out_']['R_200'])]

    number_affected_vessels_all = [len(data['0_Out_']['All']), len(data['3_Out_']['All']),
                                   len(data['4_Out_']['All']), len(data['6_Out_']['All'])]

    number_affecte_vessels = [number_affected_vessels_100, number_affected_vessels_125, number_affected_vessels_150,
                              number_affected_vessels_200, number_affected_vessels_all]

    # R_100

    for i in range(len(data['0_Out_']['R_100'])):

        x1 = data['0_Out_']['R_100'][i]
        df = df.append({'Group': 'R100', 'Network 1': x1},
                       ignore_index=True)

    for i in range(len(data['3_Out_']['R_100'])):

        x1 = data['3_Out_']['R_100'][i]
        df = df.append({'Group': 'R100', 'Network 2': x1},
                       ignore_index=True)

    for i in range(len(data['4_Out_']['R_100'])):

        x1 = data['4_Out_']['R_100'][i]
        df = df.append({'Group': 'R100', 'Network 3': x1},
                       ignore_index=True)

    for i in range(len(data['6_Out_']['R_100'])):

        x1 = data['6_Out_']['R_100'][i]
        df = df.append({'Group': 'R100', 'Network 4': x1},
                       ignore_index=True)

     # R 125

    for i in range(len(data['0_Out_']['R_125'])):
        x1 = data['0_Out_']['R_125'][i]
        df = df.append({'Group': 'R125', 'Network 1': x1},
                       ignore_index=True)

    for i in range(len(data['3_Out_']['R_125'])):
        x1 = data['3_Out_']['R_125'][i]
        df = df.append({'Group': 'R125', 'Network 2': x1},
                       ignore_index=True)

    for i in range(len(data['4_Out_']['R_125'])):
        x1 = data['4_Out_']['R_125'][i]
        df = df.append({'Group': 'R125', 'Network 3': x1},
                       ignore_index=True)

    for i in range(len(data['6_Out_']['R_125'])):
        x1 = data['6_Out_']['R_125'][i]
        df = df.append({'Group': 'R125', 'Network 4': x1},
                       ignore_index=True)

     # R 150

    for i in range(len(data['0_Out_']['R_150'])):
        x1 = data['0_Out_']['R_150'][i]
        df = df.append({'Group': 'R150', 'Network 1': x1},
                       ignore_index=True)

    for i in range(len(data['3_Out_']['R_150'])):
        x1 = data['3_Out_']['R_150'][i]
        df = df.append({'Group': 'R150', 'Network 2': x1},
                       ignore_index=True)

    for i in range(len(data['4_Out_']['R_150'])):
        x1 = data['4_Out_']['R_150'][i]
        df = df.append({'Group': 'R150', 'Network 3': x1},
                       ignore_index=True)

    for i in range(len(data['6_Out_']['R_150'])):
        x1 = data['6_Out_']['R_150'][i]
        df = df.append({'Group': 'R150', 'Network 4': x1},
                       ignore_index=True)

    # R 200

    for i in range(len(data['0_Out_']['R_200'])):

        x1 = data['0_Out_']['R_200'][i]
        df = df.append({'Group': 'R200', 'Network 1': x1},
                       ignore_index=True)

    for i in range(len(data['3_Out_']['R_200'])):

        x1 = data['3_Out_']['R_200'][i]
        df = df.append({'Group': 'R200', 'Network 2': x1},
                       ignore_index=True)

    for i in range(len(data['4_Out_']['R_200'])):

        x1 = data['4_Out_']['R_200'][i]
        df = df.append({'Group': 'R200', 'Network 3': x1},
                       ignore_index=True)

    for i in range(len(data['6_Out_']['R_200'])):

        x1 = data['6_Out_']['R_200'][i]
        df = df.append({'Group': 'R200', 'Network 4': x1},
                       ignore_index=True)

    # All

    for i in range(len(data['0_Out_']['All'])):
        x1 = data['0_Out_']['All'][i]
        df = df.append({'Group': 'All', 'Network 1': x1},
                       ignore_index=True)

    for i in range(len(data['3_Out_']['All'])):
        x1 = data['3_Out_']['All'][i]
        df = df.append({'Group': 'All', 'Network 2': x1},
                       ignore_index=True)

    for i in range(len(data['4_Out_']['All'])):
        x1 = data['4_Out_']['All'][i]
        df = df.append({'Group': 'All', 'Network 3': x1},
                       ignore_index=True)

    for i in range(len(data['6_Out_']['All'])):
        x1 = data['6_Out_']['All'][i]
        df = df.append({'Group': 'All', 'Network 4': x1},
                       ignore_index=True)

    print(df)
    dd = pd.melt(df, id_vars=['Group'], value_vars=['Network 1', 'Network 2', 'Network 3', 'Network 4'], var_name='Networks')
    plt.figure(figsize=(14, 10))
    testPlot = sns.boxplot(x='Group', y='value', data=dd, hue='Networks', showmeans=True, fliersize=1,
                    meanprops={"marker": "+", "markersize": 6, "markerfacecolor": "black", "markeredgecolor": "black"})

    # Shrink current axis by 20%
    box = testPlot.get_position()
    testPlot.set_position([box.x0, box.y0, box.width * 0.8, box.height])

    # Put a legend to the right of the current axis
    testPlot.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    testPlot.set_ylim(0, 18)

    ind = 0
    for tick in range(len(testPlot.get_xticklabels())):

        print(tick)
        testPlot.text(tick - .1, 17.5, str(np.round(number_affecte_vessels[tick][1] / data['3_Out_Tot']['Number_Vessels'] * 100, 2)) + '\%', horizontalalignment='center', color='orange', rotation=70, size=10)
        testPlot.text(tick + .1, 17.5, str(np.round(number_affecte_vessels[tick][2] / data['4_Out_Tot']['Number_Vessels'] * 100, 2)) + '\%', horizontalalignment='center', color='green', rotation=70, size=10)
        testPlot.text(tick + .3, 17.5, str(np.round(number_affecte_vessels[tick][3] / data['6_Out_Tot']['Number_Vessels'] * 100, 2)) + '\%', horizontalalignment='center', color='red', rotation=70, size=10)
        testPlot.text(tick - .3, 17.5, str(np.round(number_affecte_vessels[tick][0] / data['0_Out_Tot']['Number_Vessels'] * 100, 2)) + '\%', horizontalalignment='center', color='blue', rotation=70, size=10)
        ind += 2

    # plt.legend(loc='upper right')
    plt.savefig(
        r'D:\00 Privat\01_Bildung\01_ETH Zürich\MSc\00_Masterarbeit\02_Network_Study_Small\Adjoint\Diameter_Change_R100_to_ALL\boxplot_diameter_venules.png')


# Box Plot Diameter arterioles

def get_diameter_data_for_different_networks_arterioles(percent):

    path = r'D:\00 Privat\01_Bildung\01_ETH Zürich\MSc\00_Masterarbeit\02_Network_Study_Small\Adjoint'

    path_0_graph = r'D:\00 Privat\01_Bildung\01_ETH Zürich\MSc\00_Masterarbeit\02_Network_Study_Small\Networks\0\graph.pkl'
    path_3_graph = r'D:\00 Privat\01_Bildung\01_ETH Zürich\MSc\00_Masterarbeit\02_Network_Study_Small\Networks\3\graph.pkl'
    path_4_graph = r'D:\00 Privat\01_Bildung\01_ETH Zürich\MSc\00_Masterarbeit\02_Network_Study_Small\Networks\4\graph.pkl'
    path_6_graph = r'D:\00 Privat\01_Bildung\01_ETH Zürich\MSc\00_Masterarbeit\02_Network_Study_Small\Networks\6\graph.pkl'

    networks = ['0_Out_', '3_Out_', '4_Out_', '6_Out_']
    # networks = ['0_Out_']
    histogram_data_id = ['R_100', 'R_125', 'R_150', 'R_200', 'All']
    # histogram_data_id = ['All']

    data_network_0 = {}
    data_network_3 = {}
    data_network_4 = {}
    data_network_6 = {}

    data = {'0_Out_': {}, '3_Out_': {}, '4_Out_': {}, '6_Out_': {}, '0_Out_Tot': {}, '3_Out_Tot': {}, '4_Out_Tot': {},
            '6_Out_Tot': {}}

    for network in networks:

        graph_ = ig.Graph()

        if network == '0_Out_':

            graph_ = ig.Graph.Read_Pickle(path_0_graph)

        elif network == '3_Out_':

            graph_ = ig.Graph.Read_Pickle(path_3_graph)

        elif network == '4_Out_':

            graph_ = ig.Graph.Read_Pickle(path_4_graph)

        elif network == '6_Out_':

            graph_ = ig.Graph.Read_Pickle(path_6_graph)

        print(network)

        for radius in histogram_data_id:

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

            counter = 0

            for edge in range(graph_.ecount()):

                if graph_.es[edge]['Type'] == 2:
                    counter += 1

            data[network + 'Tot']['Number_Vessels'] = counter
            reacting_eids = []

            if radius == 'All':

                reacting_eids_new = []

                for edge in range(graph_.ecount()):

                    if graph_.es[edge]['Type'] == 2:
                        reacting_eids_new.append(edge)

                diameter_changes_reacting_edges = t_new_array[reacting_eids_new]
                diameter_changes_reacting_edges_percent = np.abs((diameter_changes_reacting_edges - 1) * 100)

            else:

                with open(current_input_path + 'reacting_eids.csv', newline='') as f:
                    reader = csv.reader(f)
                    row1 = next(reader)  # gets the first line
                    row2 = next(f)
                    reacting_eids = list(map(int, row2.split(',')))

                reacting_eids_new = []

                for reacting_eid in reacting_eids:

                    if graph_.es[reacting_eid]['Type'] == 2:
                        reacting_eids_new.append(reacting_eid)

                diameter_changes_reacting_edges = t_new_array[reacting_eids_new]
                diameter_changes_reacting_edges_percent = np.abs((diameter_changes_reacting_edges - 1) * 100)

            # HIER KANN BEGRENZUNG EINGEBAUEN WERDEN

            grenze_prozentuela_anderung = percent
            diameter_changes_selection_index = np.where(
                diameter_changes_reacting_edges_percent > grenze_prozentuela_anderung)
            diameter_changes_selection_FINAL = (
            diameter_changes_reacting_edges_percent[diameter_changes_selection_index[0]])

            # print(diameter_changes_selection_FINAL)
            data[network][radius] = diameter_changes_selection_FINAL


    # print(len(data_r_100), data_r_100)
    # print(len(data_r_150), data_r_150)
    # print(len(data_r_200), data_r_200)
    # print(len(data_r_300), data_r_300)
    # print(len(data_r_all), data_r_all)

    return data


def plot_diameter_arterioles_types_NEW(prozent):

    data = get_diameter_data_for_different_networks_arterioles(prozent)

    df = pd.DataFrame(columns=['Group', 'Network 1', 'Network 2', 'Network 3', 'Network 4', ''])

    number_affected_vessels_100 = [len(data['0_Out_']['R_100']), len(data['3_Out_']['R_100']),
                                   len(data['4_Out_']['R_100']), len(data['6_Out_']['R_100'])]

    number_affected_vessels_125 = [len(data['0_Out_']['R_125']), len(data['3_Out_']['R_125']),
                                   len(data['4_Out_']['R_125']), len(data['6_Out_']['R_125'])]

    number_affected_vessels_150 = [len(data['0_Out_']['R_150']), len(data['3_Out_']['R_150']),
                                   len(data['4_Out_']['R_150']), len(data['6_Out_']['R_150'])]

    number_affected_vessels_200 = [len(data['0_Out_']['R_200']), len(data['3_Out_']['R_200']),
                                   len(data['4_Out_']['R_200']), len(data['6_Out_']['R_200'])]

    number_affected_vessels_all = [len(data['0_Out_']['All']), len(data['3_Out_']['All']),
                                   len(data['4_Out_']['All']), len(data['6_Out_']['All'])]

    number_affecte_vessels = [number_affected_vessels_100, number_affected_vessels_125, number_affected_vessels_150,
                              number_affected_vessels_200, number_affected_vessels_all]

    # R_100

    for i in range(len(data['0_Out_']['R_100'])):

        x1 = data['0_Out_']['R_100'][i]
        df = df.append({'Group': 'R100', 'Network 1': x1},
                       ignore_index=True)

    for i in range(len(data['3_Out_']['R_100'])):

        x1 = data['3_Out_']['R_100'][i]
        df = df.append({'Group': 'R100', 'Network 2': x1},
                       ignore_index=True)

    for i in range(len(data['4_Out_']['R_100'])):

        x1 = data['4_Out_']['R_100'][i]
        df = df.append({'Group': 'R100', 'Network 3': x1},
                       ignore_index=True)

    for i in range(len(data['6_Out_']['R_100'])):

        x1 = data['6_Out_']['R_100'][i]
        df = df.append({'Group': 'R100', 'Network 4': x1},
                       ignore_index=True)

     # R 125

    for i in range(len(data['0_Out_']['R_125'])):
        x1 = data['0_Out_']['R_125'][i]
        df = df.append({'Group': 'R125', 'Network 1': x1},
                       ignore_index=True)

    for i in range(len(data['3_Out_']['R_125'])):
        x1 = data['3_Out_']['R_125'][i]
        df = df.append({'Group': 'R125', 'Network 2': x1},
                       ignore_index=True)

    for i in range(len(data['4_Out_']['R_125'])):
        x1 = data['4_Out_']['R_125'][i]
        df = df.append({'Group': 'R125', 'Network 3': x1},
                       ignore_index=True)

    for i in range(len(data['6_Out_']['R_125'])):
        x1 = data['6_Out_']['R_125'][i]
        df = df.append({'Group': 'R125', 'Network 4': x1},
                       ignore_index=True)

     # R 150

    for i in range(len(data['0_Out_']['R_150'])):
        x1 = data['0_Out_']['R_150'][i]
        df = df.append({'Group': 'R150', 'Network 1': x1},
                       ignore_index=True)

    for i in range(len(data['3_Out_']['R_150'])):
        x1 = data['3_Out_']['R_150'][i]
        df = df.append({'Group': 'R150', 'Network 2': x1},
                       ignore_index=True)

    for i in range(len(data['4_Out_']['R_150'])):
        x1 = data['4_Out_']['R_150'][i]
        df = df.append({'Group': 'R150', 'Network 3': x1},
                       ignore_index=True)

    for i in range(len(data['6_Out_']['R_150'])):
        x1 = data['6_Out_']['R_150'][i]
        df = df.append({'Group': 'R150', 'Network 4': x1},
                       ignore_index=True)

    # R 200

    for i in range(len(data['0_Out_']['R_200'])):

        x1 = data['0_Out_']['R_200'][i]
        df = df.append({'Group': 'R200', 'Network 1': x1},
                       ignore_index=True)

    for i in range(len(data['3_Out_']['R_200'])):

        x1 = data['3_Out_']['R_200'][i]
        df = df.append({'Group': 'R200', 'Network 2': x1},
                       ignore_index=True)

    for i in range(len(data['4_Out_']['R_200'])):

        x1 = data['4_Out_']['R_200'][i]
        df = df.append({'Group': 'R200', 'Network 3': x1},
                       ignore_index=True)

    for i in range(len(data['6_Out_']['R_200'])):

        x1 = data['6_Out_']['R_200'][i]
        df = df.append({'Group': 'R200', 'Network 4': x1},
                       ignore_index=True)

    # All

    for i in range(len(data['0_Out_']['All'])):
        x1 = data['0_Out_']['All'][i]
        df = df.append({'Group': 'All', 'Network 1': x1},
                       ignore_index=True)

    for i in range(len(data['3_Out_']['All'])):
        x1 = data['3_Out_']['All'][i]
        df = df.append({'Group': 'All', 'Network 2': x1},
                       ignore_index=True)

    for i in range(len(data['4_Out_']['All'])):
        x1 = data['4_Out_']['All'][i]
        df = df.append({'Group': 'All', 'Network 3': x1},
                       ignore_index=True)

    for i in range(len(data['6_Out_']['All'])):
        x1 = data['6_Out_']['All'][i]
        df = df.append({'Group': 'All', 'Network 4': x1},
                       ignore_index=True)

    print(df)
    dd = pd.melt(df, id_vars=['Group'], value_vars=['Network 1', 'Network 2', 'Network 3', 'Network 4'], var_name='Networks')
    plt.figure(figsize=(14, 10))
    testPlot = sns.boxplot(x='Group', y='value', data=dd, hue='Networks', showmeans=True, fliersize=1,
                    meanprops={"marker": "+", "markersize": 6, "markerfacecolor": "black", "markeredgecolor": "black"})

    # Shrink current axis by 20%
    box = testPlot.get_position()
    testPlot.set_position([box.x0, box.y0, box.width * 0.8, box.height])

    # Put a legend to the right of the current axis
    testPlot.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    testPlot.set_ylim(0, 18)

    ind = 0
    for tick in range(len(testPlot.get_xticklabels())):

        print(tick)
        testPlot.text(tick - .1, 17.5, str(np.round(number_affecte_vessels[tick][1] / data['3_Out_Tot']['Number_Vessels'] * 100, 2)) + '\%', horizontalalignment='center', color='orange', rotation=70, size=10)
        testPlot.text(tick + .1, 17.5, str(np.round(number_affecte_vessels[tick][2] / data['4_Out_Tot']['Number_Vessels'] * 100, 2)) + '\%', horizontalalignment='center', color='green', rotation=70, size=10)
        testPlot.text(tick + .3, 17.5, str(np.round(number_affecte_vessels[tick][3] / data['6_Out_Tot']['Number_Vessels'] * 100, 2)) + '\%', horizontalalignment='center', color='red', rotation=70, size=10)
        testPlot.text(tick - .3, 17.5, str(np.round(number_affecte_vessels[tick][0] / data['0_Out_Tot']['Number_Vessels'] * 100, 2)) + '\%', horizontalalignment='center', color='blue', rotation=70, size=10)
        ind += 2

    # plt.legend(loc='upper right')
    plt.savefig(
        r'D:\00 Privat\01_Bildung\01_ETH Zürich\MSc\00_Masterarbeit\02_Network_Study_Small\Adjoint\Diameter_Change_R100_to_ALL\boxplot_diameter_arterioles.png')


# Box Plot Diameter Venules AND Arterioles


def get_diameter_data_for_different_networks_arterioles_and_venules(percent):

    path = r'D:\00 Privat\01_Bildung\01_ETH Zürich\MSc\00_Masterarbeit\02_Network_Study_Small\Adjoint'

    path_0_graph = r'D:\00 Privat\01_Bildung\01_ETH Zürich\MSc\00_Masterarbeit\02_Network_Study_Small\Networks\0\graph.pkl'
    path_3_graph = r'D:\00 Privat\01_Bildung\01_ETH Zürich\MSc\00_Masterarbeit\02_Network_Study_Small\Networks\3\graph.pkl'
    path_4_graph = r'D:\00 Privat\01_Bildung\01_ETH Zürich\MSc\00_Masterarbeit\02_Network_Study_Small\Networks\4\graph.pkl'
    path_6_graph = r'D:\00 Privat\01_Bildung\01_ETH Zürich\MSc\00_Masterarbeit\02_Network_Study_Small\Networks\6\graph.pkl'

    networks = ['0_Out_', '3_Out_', '4_Out_', '6_Out_']
    # networks = ['0_Out_']
    histogram_data_id = ['R_100', 'R_125', 'R_150', 'R_200', 'All']
    # histogram_data_id = ['All']

    data_network_0 = {}
    data_network_3 = {}
    data_network_4 = {}
    data_network_6 = {}

    data = {'0_Out_': {}, '3_Out_': {}, '4_Out_': {}, '6_Out_': {}, '0_Out_Tot': {}, '3_Out_Tot': {}, '4_Out_Tot': {},
            '6_Out_Tot': {}}

    for network in networks:

        graph_ = ig.Graph()

        if network == '0_Out_':

            graph_ = ig.Graph.Read_Pickle(path_0_graph)

        elif network == '3_Out_':

            graph_ = ig.Graph.Read_Pickle(path_3_graph)

        elif network == '4_Out_':

            graph_ = ig.Graph.Read_Pickle(path_4_graph)

        elif network == '6_Out_':

            graph_ = ig.Graph.Read_Pickle(path_6_graph)

        print(network)

        for radius in histogram_data_id:

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

            counter = 0

            for edge in range(graph_.ecount()):

                if graph_.es[edge]['Type'] == 2 or graph_.es[edge]['Type'] == 1:
                    counter += 1

            data[network + 'Tot']['Number_Vessels'] = counter
            reacting_eids = []

            if radius == 'All':

                reacting_eids_new = []

                for edge in range(graph_.ecount()):

                    if graph_.es[edge]['Type'] == 2 or graph_.es[edge]['Type'] == 1:
                        reacting_eids_new.append(edge)

                diameter_changes_reacting_edges = t_new_array[reacting_eids_new]
                diameter_changes_reacting_edges_percent = np.abs((diameter_changes_reacting_edges - 1) * 100)

            else:

                with open(current_input_path + 'reacting_eids.csv', newline='') as f:
                    reader = csv.reader(f)
                    row1 = next(reader)  # gets the first line
                    row2 = next(f)
                    reacting_eids = list(map(int, row2.split(',')))

                reacting_eids_new = []

                for reacting_eid in reacting_eids:

                    if graph_.es[reacting_eid]['Type'] == 2 or graph_.es[reacting_eid]['Type'] == 1:
                        reacting_eids_new.append(reacting_eid)

                diameter_changes_reacting_edges = t_new_array[reacting_eids_new]
                diameter_changes_reacting_edges_percent = np.abs((diameter_changes_reacting_edges - 1) * 100)

            # HIER KANN BEGRENZUNG EINGEBAUEN WERDEN

            grenze_prozentuela_anderung = percent
            diameter_changes_selection_index = np.where(
                diameter_changes_reacting_edges_percent > grenze_prozentuela_anderung)
            diameter_changes_selection_FINAL = (
            diameter_changes_reacting_edges_percent[diameter_changes_selection_index[0]])

            # print(diameter_changes_selection_FINAL)
            data[network][radius] = diameter_changes_selection_FINAL


    # print(len(data_r_100), data_r_100)
    # print(len(data_r_150), data_r_150)
    # print(len(data_r_200), data_r_200)
    # print(len(data_r_300), data_r_300)
    # print(len(data_r_all), data_r_all)

    return data


def plot_diameter_arterioles_and_venules_types_NEW(prozent):

    data = get_diameter_data_for_different_networks_arterioles_and_venules(prozent)

    df = pd.DataFrame(columns=['Group', 'Network 1', 'Network 2', 'Network 3', 'Network 4', ''])

    number_affected_vessels_100 = [len(data['0_Out_']['R_100']), len(data['3_Out_']['R_100']),
                                   len(data['4_Out_']['R_100']), len(data['6_Out_']['R_100'])]

    number_affected_vessels_125 = [len(data['0_Out_']['R_125']), len(data['3_Out_']['R_125']),
                                   len(data['4_Out_']['R_125']), len(data['6_Out_']['R_125'])]

    number_affected_vessels_150 = [len(data['0_Out_']['R_150']), len(data['3_Out_']['R_150']),
                                   len(data['4_Out_']['R_150']), len(data['6_Out_']['R_150'])]

    number_affected_vessels_200 = [len(data['0_Out_']['R_200']), len(data['3_Out_']['R_200']),
                                   len(data['4_Out_']['R_200']), len(data['6_Out_']['R_200'])]

    number_affected_vessels_all = [len(data['0_Out_']['All']), len(data['3_Out_']['All']),
                                   len(data['4_Out_']['All']), len(data['6_Out_']['All'])]

    number_affecte_vessels = [number_affected_vessels_100, number_affected_vessels_125, number_affected_vessels_150,
                              number_affected_vessels_200, number_affected_vessels_all]

    # R_100

    for i in range(len(data['0_Out_']['R_100'])):

        x1 = data['0_Out_']['R_100'][i]
        df = df.append({'Group': 'R100', 'Network 1': x1},
                       ignore_index=True)

    for i in range(len(data['3_Out_']['R_100'])):

        x1 = data['3_Out_']['R_100'][i]
        df = df.append({'Group': 'R100', 'Network 2': x1},
                       ignore_index=True)

    for i in range(len(data['4_Out_']['R_100'])):

        x1 = data['4_Out_']['R_100'][i]
        df = df.append({'Group': 'R100', 'Network 3': x1},
                       ignore_index=True)

    for i in range(len(data['6_Out_']['R_100'])):

        x1 = data['6_Out_']['R_100'][i]
        df = df.append({'Group': 'R100', 'Network 4': x1},
                       ignore_index=True)

     # R 125

    for i in range(len(data['0_Out_']['R_125'])):
        x1 = data['0_Out_']['R_125'][i]
        df = df.append({'Group': 'R125', 'Network 1': x1},
                       ignore_index=True)

    for i in range(len(data['3_Out_']['R_125'])):
        x1 = data['3_Out_']['R_125'][i]
        df = df.append({'Group': 'R125', 'Network 2': x1},
                       ignore_index=True)

    for i in range(len(data['4_Out_']['R_125'])):
        x1 = data['4_Out_']['R_125'][i]
        df = df.append({'Group': 'R125', 'Network 3': x1},
                       ignore_index=True)

    for i in range(len(data['6_Out_']['R_125'])):
        x1 = data['6_Out_']['R_125'][i]
        df = df.append({'Group': 'R125', 'Network 4': x1},
                       ignore_index=True)

     # R 150

    for i in range(len(data['0_Out_']['R_150'])):
        x1 = data['0_Out_']['R_150'][i]
        df = df.append({'Group': 'R150', 'Network 1': x1},
                       ignore_index=True)

    for i in range(len(data['3_Out_']['R_150'])):
        x1 = data['3_Out_']['R_150'][i]
        df = df.append({'Group': 'R150', 'Network 2': x1},
                       ignore_index=True)

    for i in range(len(data['4_Out_']['R_150'])):
        x1 = data['4_Out_']['R_150'][i]
        df = df.append({'Group': 'R150', 'Network 3': x1},
                       ignore_index=True)

    for i in range(len(data['6_Out_']['R_150'])):
        x1 = data['6_Out_']['R_150'][i]
        df = df.append({'Group': 'R150', 'Network 4': x1},
                       ignore_index=True)

    # R 200

    for i in range(len(data['0_Out_']['R_200'])):

        x1 = data['0_Out_']['R_200'][i]
        df = df.append({'Group': 'R200', 'Network 1': x1},
                       ignore_index=True)

    for i in range(len(data['3_Out_']['R_200'])):

        x1 = data['3_Out_']['R_200'][i]
        df = df.append({'Group': 'R200', 'Network 2': x1},
                       ignore_index=True)

    for i in range(len(data['4_Out_']['R_200'])):

        x1 = data['4_Out_']['R_200'][i]
        df = df.append({'Group': 'R200', 'Network 3': x1},
                       ignore_index=True)

    for i in range(len(data['6_Out_']['R_200'])):

        x1 = data['6_Out_']['R_200'][i]
        df = df.append({'Group': 'R200', 'Network 4': x1},
                       ignore_index=True)

    # All

    for i in range(len(data['0_Out_']['All'])):
        x1 = data['0_Out_']['All'][i]
        df = df.append({'Group': 'All', 'Network 1': x1},
                       ignore_index=True)

    for i in range(len(data['3_Out_']['All'])):
        x1 = data['3_Out_']['All'][i]
        df = df.append({'Group': 'All', 'Network 2': x1},
                       ignore_index=True)

    for i in range(len(data['4_Out_']['All'])):
        x1 = data['4_Out_']['All'][i]
        df = df.append({'Group': 'All', 'Network 3': x1},
                       ignore_index=True)

    for i in range(len(data['6_Out_']['All'])):
        x1 = data['6_Out_']['All'][i]
        df = df.append({'Group': 'All', 'Network 4': x1},
                       ignore_index=True)

    print(df)
    dd = pd.melt(df, id_vars=['Group'], value_vars=['Network 1', 'Network 2', 'Network 3', 'Network 4'], var_name='Networks')
    plt.figure(figsize=(14, 10))
    testPlot = sns.boxplot(x='Group', y='value', data=dd, hue='Networks', showmeans=True, fliersize=1,
                    meanprops={"marker": "+", "markersize": 6, "markerfacecolor": "black", "markeredgecolor": "black"})

    # Shrink current axis by 20%
    box = testPlot.get_position()
    testPlot.set_position([box.x0, box.y0, box.width * 0.8, box.height])

    # Put a legend to the right of the current axis
    testPlot.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    testPlot.set_ylim(0, 18)

    ind = 0
    for tick in range(len(testPlot.get_xticklabels())):

        print(tick)
        testPlot.text(tick - .1, 17.5, str(np.round(number_affecte_vessels[tick][1] / data['3_Out_Tot']['Number_Vessels'] * 100, 2)) + '\%', horizontalalignment='center', color='orange', rotation=70, size=10)
        testPlot.text(tick + .1, 17.5, str(np.round(number_affecte_vessels[tick][2] / data['4_Out_Tot']['Number_Vessels'] * 100, 2)) + '\%', horizontalalignment='center', color='green', rotation=70, size=10)
        testPlot.text(tick + .3, 17.5, str(np.round(number_affecte_vessels[tick][3] / data['6_Out_Tot']['Number_Vessels'] * 100, 2)) + '\%', horizontalalignment='center', color='red', rotation=70, size=10)
        testPlot.text(tick - .3, 17.5, str(np.round(number_affecte_vessels[tick][0] / data['0_Out_Tot']['Number_Vessels'] * 100, 2)) + '\%', horizontalalignment='center', color='blue', rotation=70, size=10)
        ind += 2

    # plt.legend(loc='upper right')
    plt.savefig(
        r'D:\00 Privat\01_Bildung\01_ETH Zürich\MSc\00_Masterarbeit\02_Network_Study_Small\Adjoint\Diameter_Change_R100_to_ALL\boxplot_diameter_arterioles_and_venules.png')


########################################################################################################################
#
# DIAMETER BOXPLOTS - Direct Comparison (All Cap, All Arteries)
#
########################################################################################################################


def get_diameter_data_all_all_art_all_caps(percent):

    path = r'D:\00 Privat\01_Bildung\01_ETH Zürich\MSc\00_Masterarbeit\02_Network_Study_Small\Adjoint'

    networks = ['0_Out_', '3_Out_', '4_Out_', '6_Out_']
    # networks = ['0_Out_']
    histogram_data_id = ['All_Arteries', 'All_Cap', 'All']
    # histogram_data_id = ['All']

    data_network_0 = {}
    data_network_3 = {}
    data_network_4 = {}
    data_network_6 = {}

    data = {'0_Out_': {}, '3_Out_': {}, '4_Out_': {}, '6_Out_': {}, '0_Out_Tot': {}, '3_Out_Tot': {}, '4_Out_Tot': {},
            '6_Out_Tot': {}}

    for network in networks:

        print(network)

        for radius in histogram_data_id:

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
            data[network + 'Tot']['Number_Vessels'] = len(t_new_array)
            reacting_eids = []

            if radius == 'All':

                diameter_changes_reacting_edges = t_new_array
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

            # print(diameter_changes_selection_FINAL)
            data[network][radius] = diameter_changes_selection_FINAL


    # print(len(data_r_100), data_r_100)
    # print(len(data_r_150), data_r_150)
    # print(len(data_r_200), data_r_200)
    # print(len(data_r_300), data_r_300)
    # print(len(data_r_all), data_r_all)

    return data


def plot_diameter_all_caps_all_art_all_Comparison(prozent):

    data = get_diameter_data_all_all_art_all_caps(prozent)

    df = pd.DataFrame(columns=['Group', 'Network 1', 'Network 2', 'Network 3', 'Network 4', ''])

    number_affected_vessels_all_arteries = [len(data['0_Out_']['All_Arteries']), len(data['3_Out_']['All_Arteries']),
                                   len(data['4_Out_']['All_Arteries']), len(data['6_Out_']['All_Arteries'])]

    number_affected_vessels_all_cap = [len(data['0_Out_']['All_Cap']), len(data['3_Out_']['All_Cap']),
                                   len(data['4_Out_']['All_Cap']), len(data['6_Out_']['All_Cap'])]

    number_affected_vessels_all = [len(data['0_Out_']['All']), len(data['3_Out_']['All']),
                                   len(data['4_Out_']['All']), len(data['6_Out_']['All'])]

    number_affecte_vessels = [number_affected_vessels_all_arteries, number_affected_vessels_all_cap, number_affected_vessels_all]

    # All_Arteries

    for i in range(len(data['0_Out_']['All_Arteries'])):

        x1 = data['0_Out_']['All_Arteries'][i]
        df = df.append({'Group': 'All_Arteries', 'Network 1': x1},
                       ignore_index=True)

    for i in range(len(data['3_Out_']['All_Arteries'])):

        x1 = data['3_Out_']['All_Arteries'][i]
        df = df.append({'Group': 'All_Arteries', 'Network 2': x1},
                       ignore_index=True)

    for i in range(len(data['4_Out_']['All_Arteries'])):

        x1 = data['4_Out_']['All_Arteries'][i]
        df = df.append({'Group': 'All_Arteries', 'Network 3': x1},
                       ignore_index=True)

    for i in range(len(data['6_Out_']['All_Arteries'])):

        x1 = data['6_Out_']['All_Arteries'][i]
        df = df.append({'Group': 'All_Arteries', 'Network 4': x1},
                       ignore_index=True)

     # All Cap

    for i in range(len(data['0_Out_']['All_Cap'])):
        x1 = data['0_Out_']['All_Cap'][i]
        df = df.append({'Group': 'All_Cap', 'Network 1': x1},
                       ignore_index=True)

    for i in range(len(data['3_Out_']['All_Cap'])):
        x1 = data['3_Out_']['All_Cap'][i]
        df = df.append({'Group': 'All_Cap', 'Network 2': x1},
                       ignore_index=True)

    for i in range(len(data['4_Out_']['All_Cap'])):
        x1 = data['4_Out_']['All_Cap'][i]
        df = df.append({'Group': 'All_Cap', 'Network 3': x1},
                       ignore_index=True)

    for i in range(len(data['6_Out_']['All_Cap'])):
        x1 = data['6_Out_']['All_Cap'][i]
        df = df.append({'Group': 'All_Cap', 'Network 4': x1},
                       ignore_index=True)

    # All

    for i in range(len(data['0_Out_']['All'])):
        x1 = data['0_Out_']['All'][i]
        df = df.append({'Group': 'All', 'Network 1': x1},
                       ignore_index=True)

    for i in range(len(data['3_Out_']['All'])):
        x1 = data['3_Out_']['All'][i]
        df = df.append({'Group': 'All', 'Network 2': x1},
                       ignore_index=True)

    for i in range(len(data['4_Out_']['All'])):
        x1 = data['4_Out_']['All'][i]
        df = df.append({'Group': 'All', 'Network 3': x1},
                       ignore_index=True)

    for i in range(len(data['6_Out_']['All'])):
        x1 = data['6_Out_']['All'][i]
        df = df.append({'Group': 'All', 'Network 4': x1},
                       ignore_index=True)

    print(df)
    dd = pd.melt(df, id_vars=['Group'], value_vars=['Network 1', 'Network 2', 'Network 3', 'Network 4'], var_name='Networks')
    plt.figure(figsize=(14, 10))
    testPlot = sns.boxplot(x='Group', y='value', data=dd, hue='Networks', showmeans=True, fliersize=1,
                    meanprops={"marker": "+", "markersize": 6, "markerfacecolor": "black", "markeredgecolor": "black"})

    # Shrink current axis by 20%
    box = testPlot.get_position()
    testPlot.set_position([box.x0, box.y0, box.width * 0.8, box.height])

    # Put a legend to the right of the current axis
    testPlot.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    testPlot.set_ylim(-6, 40)

    ind = 0
    for tick in range(len(testPlot.get_xticklabels())):

        print(tick)
        testPlot.text(tick - .1, -2, str(np.round(number_affecte_vessels[tick][1] / data['3_Out_Tot']['Number_Vessels'] * 100, 2)) + '\%', horizontalalignment='center', color='orange', rotation=60, size=10)
        testPlot.text(tick + .1, -2, str(np.round(number_affecte_vessels[tick][2] / data['4_Out_Tot']['Number_Vessels'] * 100, 2)) + '\%', horizontalalignment='center', color='green', rotation=60, size=10)
        testPlot.text(tick + .3, -2, str(np.round(number_affecte_vessels[tick][3] / data['6_Out_Tot']['Number_Vessels'] * 100, 2)) + '\%', horizontalalignment='center', color='red', rotation=60, size=10)
        testPlot.text(tick - .3, -2, str(np.round(number_affecte_vessels[tick][0] / data['0_Out_Tot']['Number_Vessels'] * 100, 2)) + '\%', horizontalalignment='center', color='blue', rotation=60, size=10)
        ind += 2

    # plt.legend(loc='upper right')
    plt.savefig(
        r'D:\00 Privat\01_Bildung\01_ETH Zürich\MSc\00_Masterarbeit\02_Network_Study_Small\Adjoint\Direct_Comparison\boxplot_diameter_all_all_arteries_all_caps.png')

plot_diameter_all_caps_all_art_all_Comparison(1)


########################################################################################################################
#
# FLOW BOXPLOTS - Direct Comparison (All Cap, All Arteries)
#
########################################################################################################################


def get_flow_data_for_different_networks_arteries_direct_comp(percent):

    path = r'D:\00 Privat\01_Bildung\01_ETH Zürich\MSc\00_Masterarbeit\02_Network_Study_Small\Adjoint'

    networks = ['0_Out_', '3_Out_', '4_Out_', '6_Out_']
    # networks = ['0_Out_']
    histogram_data_id = ['All_Arteries']
    # histogram_data_id = ['All']

    data = []
    for network in networks:

        print(network)

        for radius in histogram_data_id:

            current_input_path = path + '\\' + network + '\\' + radius + '\\' + 'in\\adjointMethod' + '\\'
            current_output_path = path + '\\' + network + '\\' + radius + '\\' + 'out' + '\\'
            meshdata_start = current_output_path + 'meshdata_9999.csv'

            r = csv.reader(open(current_input_path + 'activated_eids.csv'))
            lines = list(r)
            activated_eids = lines[0]
            activated_eids = list(map(int, activated_eids))

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

            f_start = np.array(df_start['tav_Fplasma'])
            f_final = np.array(df_final['tav_Fplasma'])

            f_start_new = []
            f_end_new = []

            for i in range(len(f_start)):

                if (i % 2) == 0:
                    f_start_new.append(f_start[i])
                    f_end_new.append((f_final[i]))

            f_start_new_array = np.asarray(f_start_new)
            f_end_new_array = np.asarray(f_end_new)

            f_ratio = f_start_new_array / f_end_new_array
            flow_changes_reacting_edges_percent = np.abs((f_ratio - 1) * 100)

            # Length

            l_start = np.array(df_start['L'])
            l_start_new = []

            for i in range(len(l_start)):

                if (i % 2) == 0:
                    l_start_new.append(l_start[i])

            l_start_new_array = np.asarray(l_start_new)

            # print(len(flow_changes_reacting_edges_percent))
            # print(len(l_start_new_array))
            # print(len(activated_eids))

            for i in reversed(activated_eids):

                flow_changes_reacting_edges_percent = np.delete(flow_changes_reacting_edges_percent, i)
                l_start_new_array = np.delete(l_start_new_array, i)

            # print(len(flow_changes_reacting_edges_percent))
            # print(len(l_start_new_array))

            # HIER KANN BEGRENZUNG EINGEBAUEN WERDEN

            grenze_prozentuela_anderung = percent
            flow_changes_selection_index = np.where(flow_changes_reacting_edges_percent > grenze_prozentuela_anderung)
            flow_changes_selction = (flow_changes_reacting_edges_percent[flow_changes_selection_index[0]])
            flow_changes_selection_FINAL = []

            counter = 0

            for x in flow_changes_selction:

                if x < 100:

                    counter += 1
                    flow_changes_selection_FINAL.append(x)

                else:

                    # print(x)
                    None

            # print(len(flow_changes_selection_FINAL))
            # print(counter)
            # print(diameter_changes_selection_FINAL)

            data.append(flow_changes_selection_FINAL)

    return data


def plot_flow_arteries_direct_comp(prozent):

    random_dists = ['Network 1', 'Network 2', ' Network 3', 'Network 4']

    # N = 10
    #
    # norm = np.random.normal(1, 1, N)
    # print(norm)
    # logn = np.random.lognormal(1, 1, N)
    # expo = np.random.exponential(1, N)
    # gumb = np.random.gumbel(6, 4, N)
    # tria = np.random.triangular(2, 9, 11, N)

    # Generate some random indices that we'll use to resample the original data
    # arrays. For code brevity, just use the same random indices for each array
    # bootstrap_indices = np.random.randint(0, N, N)

    data = get_flow_data_for_different_networks_arteries_direct_comp(prozent)
    # data = [
    #     norm,
    #     logn,
    #     expo,
    #     gumb,
    #     tria,
    # ]

    fig, ax1 = plt.subplots(figsize=(10, 6))
    fig.canvas.set_window_title('A Boxplot Example')
    fig.subplots_adjust(left=0.075, right=0.95, top=0.9, bottom=0.25)

    bp = ax1.boxplot(data, notch=0, sym='+', vert=1, whis=1.5)
    plt.setp(bp['boxes'], color='black')
    plt.setp(bp['whiskers'], color='black')
    plt.setp(bp['fliers'], color='black', marker='+', markersize=0.2)
    plt.setp(bp['medians'], color='green')

    # Add a horizontal grid to the plot, but make it very light in color
    # so we can use it for reading data values but not be distracting
    ax1.yaxis.grid(True, linestyle='-', which='major', color='lightgrey',
                   alpha=0.5)

    # Hide these grid behind plot objects
    ax1.set_axisbelow(True)
    ax1.set_title('Vessels outside activated region included only')
    ax1.set_xlabel('')
    ax1.set_ylabel('Relative Flow Change in %')

    # Now fill the boxes with desired colors
    box_colors = ['lightgray', 'lightgray']
    num_boxes = len(data)
    medians = np.empty(num_boxes)
    for i in range(num_boxes):
        box = bp['boxes'][i]
        boxX = []
        boxY = []
        for j in range(5):
            boxX.append(box.get_xdata()[j])
            boxY.append(box.get_ydata()[j])
        box_coords = np.column_stack([boxX, boxY])
        # Alternate between Dark Khaki and Royal Blue

        ax1.add_patch(Polygon(box_coords, facecolor=box_colors[i % 2]))

        # Now draw the median lines back over what we just filled in
        med = bp['medians'][i]
        medianX = []
        medianY = []
        for j in range(2):
            medianX.append(med.get_xdata()[j])
            medianY.append(med.get_ydata()[j])
            # ax1.plot(medianX, medianY, 'k')
        medians[i] = medianY[0]
        # Finally, overplot the sample averages, with horizontal alignment
        # in the center of each box
        ax1.plot(np.average(med.get_xdata()), np.average(data[i]),
                 color='Royalblue', marker='*', markeredgecolor='Royalblue')

    # Set the axes ranges and axes labels
    ax1.set_xlim(0.5, num_boxes + 0.5)
    top = 60
    bottom = -14
    ax1.set_ylim(bottom, top)
    ax1.set_xticklabels(random_dists,
                        rotation=45, fontsize=8)

    # Due to the Y-axis scale being different across samples, it can be
    # hard to compare differences in medians across the samples. Add upper
    # X-axis tick labels with the sample medians to aid in comparison
    # (just use two decimal places of precision)
    pos = np.arange(num_boxes) + 1
    upper_labels = [str(np.round(s, 2)) for s in medians]
    lower_labels = []
    lower_labels.append(str(np.round(np.mean(data[0]), 2)))
    lower_labels.append(str(np.round(np.mean(data[1]), 2)))
    lower_labels.append(str(np.round(np.mean(data[2]), 2)))
    lower_labels.append(str(np.round(np.mean(data[3]), 2)))


    weights = ['bold', 'semibold']
    for tick, label in zip(range(num_boxes), ax1.get_xticklabels()):
        k = tick % 2
        ax1.text(pos[tick], .1, upper_labels[tick],
                 transform=ax1.get_xaxis_transform(),
                 horizontalalignment='center', size='x-small',
                 weight=weights[k], color='green')

        ax1.text(pos[tick], .05, lower_labels[tick],
                 transform=ax1.get_xaxis_transform(),
                 horizontalalignment='center', size='x-small',
                 weight=weights[k], color='Royalblue')

    # Finally, add a basic legend
    # fig.text(0.80, 0.08, f'{N} Random Numbers',
    #          backgroundcolor=box_colors[0], color='black', weight='roman',
    #          size='x-small')
    # fig.text(0.80, 0.045, 'IID Bootstrap Resample',
    #          backgroundcolor=box_colors[1],
    #          color='white', weight='roman', size='x-small')
    # fig.text(0.80, 0.015, '*', color='white', backgroundcolor='silver',
    #          weight='roman', size='medium')
    # fig.text(0.815, 0.013, ' Average Value', color='black', weight='roman',
    #          size='x-small')

    plt.savefig(r'D:\00 Privat\01_Bildung\01_ETH Zürich\MSc\00_Masterarbeit\02_Network_Study_Small\Adjoint\Direct_Comparison\boxplot_flow_arterioles.png')


def get_flow_data_for_different_networks_caps_direct_comp(percent):

    path = r'D:\00 Privat\01_Bildung\01_ETH Zürich\MSc\00_Masterarbeit\02_Network_Study_Small\Adjoint'

    networks = ['0_Out_', '3_Out_', '4_Out_', '6_Out_']
    # networks = ['0_Out_']
    histogram_data_id = ['All_Cap']
    # histogram_data_id = ['All']

    data = []
    for network in networks:

        print(network)

        for radius in histogram_data_id:

            current_input_path = path + '\\' + network + '\\' + radius + '\\' + 'in\\adjointMethod' + '\\'
            current_output_path = path + '\\' + network + '\\' + radius + '\\' + 'out' + '\\'
            meshdata_start = current_output_path + 'meshdata_9999.csv'

            r = csv.reader(open(current_input_path + 'activated_eids.csv'))
            lines = list(r)
            activated_eids = lines[0]
            activated_eids = list(map(int, activated_eids))

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

            f_start = np.array(df_start['tav_Fplasma'])
            f_final = np.array(df_final['tav_Fplasma'])

            f_start_new = []
            f_end_new = []

            for i in range(len(f_start)):

                if (i % 2) == 0:
                    f_start_new.append(f_start[i])
                    f_end_new.append((f_final[i]))

            f_start_new_array = np.asarray(f_start_new)
            f_end_new_array = np.asarray(f_end_new)

            f_ratio = f_start_new_array / f_end_new_array
            flow_changes_reacting_edges_percent = np.abs((f_ratio - 1) * 100)

            # Length

            l_start = np.array(df_start['L'])
            l_start_new = []

            for i in range(len(l_start)):

                if (i % 2) == 0:
                    l_start_new.append(l_start[i])

            l_start_new_array = np.asarray(l_start_new)

            # print(len(flow_changes_reacting_edges_percent))
            # print(len(l_start_new_array))
            # print(len(activated_eids))

            for i in reversed(activated_eids):

                flow_changes_reacting_edges_percent = np.delete(flow_changes_reacting_edges_percent, i)
                l_start_new_array = np.delete(l_start_new_array, i)

            # print(len(flow_changes_reacting_edges_percent))
            # print(len(l_start_new_array))

            # HIER KANN BEGRENZUNG EINGEBAUEN WERDEN

            grenze_prozentuela_anderung = percent
            flow_changes_selection_index = np.where(flow_changes_reacting_edges_percent > grenze_prozentuela_anderung)
            flow_changes_selction = (flow_changes_reacting_edges_percent[flow_changes_selection_index[0]])
            flow_changes_selection_FINAL = []

            counter = 0

            for x in flow_changes_selction:

                if x < 100:

                    counter += 1
                    flow_changes_selection_FINAL.append(x)

                else:

                    # print(x)
                    None

            # print(len(flow_changes_selection_FINAL))
            # print(counter)
            # print(diameter_changes_selection_FINAL)

            data.append(flow_changes_selection_FINAL)

    return data


def plot_flow_caps_direct_comp(prozent):

    random_dists = ['Network 1', 'Network 2', ' Network 3', 'Network 4']

    # N = 10
    #
    # norm = np.random.normal(1, 1, N)
    # print(norm)
    # logn = np.random.lognormal(1, 1, N)
    # expo = np.random.exponential(1, N)
    # gumb = np.random.gumbel(6, 4, N)
    # tria = np.random.triangular(2, 9, 11, N)

    # Generate some random indices that we'll use to resample the original data
    # arrays. For code brevity, just use the same random indices for each array
    # bootstrap_indices = np.random.randint(0, N, N)

    data = get_flow_data_for_different_networks_caps_direct_comp(prozent)
    # data = [
    #     norm,
    #     logn,
    #     expo,
    #     gumb,
    #     tria,
    # ]

    fig, ax1 = plt.subplots(figsize=(10, 6))
    fig.canvas.set_window_title('A Boxplot Example')
    fig.subplots_adjust(left=0.075, right=0.95, top=0.9, bottom=0.25)

    bp = ax1.boxplot(data, notch=0, sym='+', vert=1, whis=1.5)
    plt.setp(bp['boxes'], color='black')
    plt.setp(bp['whiskers'], color='black')
    plt.setp(bp['fliers'], color='black', marker='+', markersize=0.2)
    plt.setp(bp['medians'], color='green')

    # Add a horizontal grid to the plot, but make it very light in color
    # so we can use it for reading data values but not be distracting
    ax1.yaxis.grid(True, linestyle='-', which='major', color='lightgrey',
                   alpha=0.5)

    # Hide these grid behind plot objects
    ax1.set_axisbelow(True)
    ax1.set_title('Vessels outside activated region included only')
    ax1.set_xlabel('')
    ax1.set_ylabel('Relative Flow Change in %')

    # Now fill the boxes with desired colors
    box_colors = ['lightgray', 'lightgray']
    num_boxes = len(data)
    medians = np.empty(num_boxes)
    for i in range(num_boxes):
        box = bp['boxes'][i]
        boxX = []
        boxY = []
        for j in range(5):
            boxX.append(box.get_xdata()[j])
            boxY.append(box.get_ydata()[j])
        box_coords = np.column_stack([boxX, boxY])
        # Alternate between Dark Khaki and Royal Blue

        ax1.add_patch(Polygon(box_coords, facecolor=box_colors[i % 2]))

        # Now draw the median lines back over what we just filled in
        med = bp['medians'][i]
        medianX = []
        medianY = []
        for j in range(2):
            medianX.append(med.get_xdata()[j])
            medianY.append(med.get_ydata()[j])
            # ax1.plot(medianX, medianY, 'k')
        medians[i] = medianY[0]
        # Finally, overplot the sample averages, with horizontal alignment
        # in the center of each box
        ax1.plot(np.average(med.get_xdata()), np.average(data[i]),
                 color='Royalblue', marker='*', markeredgecolor='Royalblue')

    # Set the axes ranges and axes labels
    ax1.set_xlim(0.5, num_boxes + 0.5)
    top = 60
    bottom = -14
    ax1.set_ylim(bottom, top)
    ax1.set_xticklabels(random_dists,
                        rotation=45, fontsize=8)

    # Due to the Y-axis scale being different across samples, it can be
    # hard to compare differences in medians across the samples. Add upper
    # X-axis tick labels with the sample medians to aid in comparison
    # (just use two decimal places of precision)
    pos = np.arange(num_boxes) + 1
    upper_labels = [str(np.round(s, 2)) for s in medians]
    lower_labels = []
    lower_labels.append(str(np.round(np.mean(data[0]), 2)))
    lower_labels.append(str(np.round(np.mean(data[1]), 2)))
    lower_labels.append(str(np.round(np.mean(data[2]), 2)))
    lower_labels.append(str(np.round(np.mean(data[3]), 2)))


    weights = ['bold', 'semibold']
    for tick, label in zip(range(num_boxes), ax1.get_xticklabels()):
        k = tick % 2
        ax1.text(pos[tick], .1, upper_labels[tick],
                 transform=ax1.get_xaxis_transform(),
                 horizontalalignment='center', size='x-small',
                 weight=weights[k], color='green')

        ax1.text(pos[tick], .05, lower_labels[tick],
                 transform=ax1.get_xaxis_transform(),
                 horizontalalignment='center', size='x-small',
                 weight=weights[k], color='Royalblue')

    # Finally, add a basic legend
    # fig.text(0.80, 0.08, f'{N} Random Numbers',
    #          backgroundcolor=box_colors[0], color='black', weight='roman',
    #          size='x-small')
    # fig.text(0.80, 0.045, 'IID Bootstrap Resample',
    #          backgroundcolor=box_colors[1],
    #          color='white', weight='roman', size='x-small')
    # fig.text(0.80, 0.015, '*', color='white', backgroundcolor='silver',
    #          weight='roman', size='medium')
    # fig.text(0.815, 0.013, ' Average Value', color='black', weight='roman',
    #          size='x-small')

    plt.savefig(r'D:\00 Privat\01_Bildung\01_ETH Zürich\MSc\00_Masterarbeit\02_Network_Study_Small\Adjoint\Direct_Comparison\boxplot_flow_caps.png')


########################################################################################################################
#
# FLOW BOXPLOTS (R100 - ALL)
#
########################################################################################################################

# Box Plot Flow All Vessel Types REST


def exclude_edges_with_too_small_flows(network_id):

    path = r'D:\00 Privat\01_Bildung\01_ETH Zürich\MSc\00_Masterarbeit\02_Network_Study_Small\Adjoint' + '\\' \
           + network_id + '\All\out\meshdata_9999.csv'

    df_start = pd.read_csv(path)

    # FLOW

    f_start = np.array(df_start['tav_Fplasma'])
    f_start_new = []

    for i in range(len(f_start)):

        if (i % 2) == 0:
            f_start_new.append(f_start[i])
            # f_end_new.append((f_final[i]))

    f_start_new_array = np.asarray(f_start_new)

    # LENGTH

    l_start = np.array(df_start['L'])
    l_start_new = []

    for i in range(len(l_start)):

        if (i % 2) == 0:
            l_start_new.append(l_start[i])

    l_start_new_array = np.asarray(l_start_new)

    length_total_sum = np.sum(l_start_new_array)

    q_sim_zeitpunkt_9999 = 1/length_total_sum * np.sum((l_start_new_array * abs(f_start_new_array)))

    a = np.where(np.abs(f_start_new_array) > q_sim_zeitpunkt_9999 / 30)
    relevant_edges = list(a[0])

    return relevant_edges


def get_flow_data_for_different_networks_NOT_activated_region(percent):

    path = r'D:\00 Privat\01_Bildung\01_ETH Zürich\MSc\00_Masterarbeit\02_Network_Study_Small\Adjoint'

    networks = ['0_Out_', '3_Out_', '4_Out_', '6_Out_']
    # networks = ['0_Out_']
    histogram_data_id = ['R_100', 'R_125', 'R_150', 'R_200', 'All']
    # histogram_data_id = ['All']

    data_network_0 = {}
    data_network_3 = {}
    data_network_4 = {}
    data_network_6 = {}

    data = {'0_Out_': {}, '3_Out_': {}, '4_Out_': {}, '6_Out_': {}, '0_Out_Tot': {}, '3_Out_Tot': {}, '4_Out_Tot': {},
            '6_Out_Tot': {}}

    for network in networks:

        print(network)

        for radius in histogram_data_id:

            current_input_path = path + '\\' + network + '\\' + radius + '\\' + 'in\\adjointMethod' + '\\'
            current_output_path = path + '\\' + network + '\\' + radius + '\\' + 'out' + '\\'
            meshdata_start = current_output_path + 'meshdata_9999.csv'

            r = csv.reader(open(current_input_path + 'activated_eids.csv'))
            lines = list(r)
            activated_eids = lines[0]
            activated_eids = list(map(int, activated_eids))

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
            data[network + 'Tot']['Number_Vessels'] = len(t_new_array)

            f_start = np.array(df_start['tav_Fplasma'])
            f_final = np.array(df_final['tav_Fplasma'])

            f_start_new = []
            f_end_new = []

            for i in range(len(f_start)):

                if (i % 2) == 0:
                    f_start_new.append(f_start[i])
                    f_end_new.append((f_final[i]))

            f_start_new_array = np.asarray(f_start_new)
            f_end_new_array = np.asarray(f_end_new)

            f_ratio = np.abs(f_start_new_array / f_end_new_array)
            relevant_edges_flow_higher_1_100 = exclude_edges_with_too_small_flows(network)

            for activated_edge in activated_eids:

                try:
                    relevant_edges_flow_higher_1_100.remove(activated_edge)

                except:

                    print("Too Small Baselineflow in this Edge. Already deleted ")

            flow_changes_reacting_edges = f_ratio[relevant_edges_flow_higher_1_100]
            flow_changes_reacting_edges_percent = np.abs((flow_changes_reacting_edges - 1) * 100)

            # HIER KANN BEGRENZUNG EINGEBAUEN WERDEN

            grenze_prozentuela_anderung = percent
            flow_changes_selection_index = np.where(flow_changes_reacting_edges_percent > grenze_prozentuela_anderung)
            flow_changes_selection_FINAL = (flow_changes_reacting_edges_percent[flow_changes_selection_index[0]])

            a = np.where(flow_changes_selection_FINAL < 1000)

            flow_changes_selection_FINAL = flow_changes_selection_FINAL[a]

            # print(diameter_changes_selection_FINAL)
            data[network][radius] = flow_changes_selection_FINAL

    # print(len(data_r_100), data_r_100)
    # print(len(data_r_150), data_r_150)
    # print(len(data_r_200), data_r_200)
    # print(len(data_r_300), data_r_300)
    # print(len(data_r_all), data_r_all)

    return data


def plot_flow_all_vessel_types_NEW(prozent):

    print('Start Plotting')

    data = get_flow_data_for_different_networks_NOT_activated_region(prozent)

    print('recieved data')
    df = pd.DataFrame(columns=['Group', 'Network 1', 'Network 2', 'Network 3', 'Network 4', ''])

    number_affected_vessels_100 = [len(data['0_Out_']['R_100']), len(data['3_Out_']['R_100']),
                                   len(data['4_Out_']['R_100']), len(data['6_Out_']['R_100'])]

    number_affected_vessels_125 = [len(data['0_Out_']['R_125']), len(data['3_Out_']['R_125']),
                                   len(data['4_Out_']['R_125']), len(data['6_Out_']['R_125'])]

    number_affected_vessels_150 = [len(data['0_Out_']['R_150']), len(data['3_Out_']['R_150']),
                                   len(data['4_Out_']['R_150']), len(data['6_Out_']['R_150'])]

    number_affected_vessels_200 = [len(data['0_Out_']['R_200']), len(data['3_Out_']['R_200']),
                                   len(data['4_Out_']['R_200']), len(data['6_Out_']['R_200'])]

    number_affected_vessels_all = [len(data['0_Out_']['All']), len(data['3_Out_']['All']),
                                   len(data['4_Out_']['All']), len(data['6_Out_']['All'])]

    number_affecte_vessels = [number_affected_vessels_100, number_affected_vessels_125, number_affected_vessels_150,
                              number_affected_vessels_200, number_affected_vessels_all]

    # R_100

    print('collect r100 data')
    for i in range(len(data['0_Out_']['R_100'])):

        x1 = data['0_Out_']['R_100'][i]
        df = df.append({'Group': 'R100', 'Network 1': x1},
                       ignore_index=True)

    for i in range(len(data['3_Out_']['R_100'])):

        x1 = data['3_Out_']['R_100'][i]
        df = df.append({'Group': 'R100', 'Network 2': x1},
                       ignore_index=True)

    for i in range(len(data['4_Out_']['R_100'])):

        x1 = data['4_Out_']['R_100'][i]
        df = df.append({'Group': 'R100', 'Network 3': x1},
                       ignore_index=True)

    for i in range(len(data['6_Out_']['R_100'])):

        x1 = data['6_Out_']['R_100'][i]
        df = df.append({'Group': 'R100', 'Network 4': x1},
                       ignore_index=True)

     # R 125

    print('collect r125 data')

    for i in range(len(data['0_Out_']['R_125'])):
        x1 = data['0_Out_']['R_125'][i]
        df = df.append({'Group': 'R125', 'Network 1': x1},
                       ignore_index=True)

    for i in range(len(data['3_Out_']['R_125'])):
        x1 = data['3_Out_']['R_125'][i]
        df = df.append({'Group': 'R125', 'Network 2': x1},
                       ignore_index=True)

    for i in range(len(data['4_Out_']['R_125'])):
        x1 = data['4_Out_']['R_125'][i]
        df = df.append({'Group': 'R125', 'Network 3': x1},
                       ignore_index=True)

    for i in range(len(data['6_Out_']['R_125'])):
        x1 = data['6_Out_']['R_125'][i]
        df = df.append({'Group': 'R125', 'Network 4': x1},
                       ignore_index=True)

     # R 150

    print('collect r150 data')
    for i in range(len(data['0_Out_']['R_150'])):
        x1 = data['0_Out_']['R_150'][i]
        df = df.append({'Group': 'R150', 'Network 1': x1},
                       ignore_index=True)

    for i in range(len(data['3_Out_']['R_150'])):
        x1 = data['3_Out_']['R_150'][i]
        df = df.append({'Group': 'R150', 'Network 2': x1},
                       ignore_index=True)

    for i in range(len(data['4_Out_']['R_150'])):
        x1 = data['4_Out_']['R_150'][i]
        df = df.append({'Group': 'R150', 'Network 3': x1},
                       ignore_index=True)

    for i in range(len(data['6_Out_']['R_150'])):
        x1 = data['6_Out_']['R_150'][i]
        df = df.append({'Group': 'R150', 'Network 4': x1},
                       ignore_index=True)

    # R 200

    print('collect r200 data')
    for i in range(len(data['0_Out_']['R_200'])):

        x1 = data['0_Out_']['R_200'][i]
        df = df.append({'Group': 'R200', 'Network 1': x1},
                       ignore_index=True)

    for i in range(len(data['3_Out_']['R_200'])):

        x1 = data['3_Out_']['R_200'][i]
        df = df.append({'Group': 'R200', 'Network 2': x1},
                       ignore_index=True)

    for i in range(len(data['4_Out_']['R_200'])):

        x1 = data['4_Out_']['R_200'][i]
        df = df.append({'Group': 'R200', 'Network 3': x1},
                       ignore_index=True)

    for i in range(len(data['6_Out_']['R_200'])):

        x1 = data['6_Out_']['R_200'][i]
        df = df.append({'Group': 'R200', 'Network 4': x1},
                       ignore_index=True)

    # All

    print('collect all data')
    for i in range(len(data['0_Out_']['All'])):
        x1 = data['0_Out_']['All'][i]
        df = df.append({'Group': 'All', 'Network 1': x1},
                       ignore_index=True)

    for i in range(len(data['3_Out_']['All'])):
        x1 = data['3_Out_']['All'][i]
        df = df.append({'Group': 'All', 'Network 2': x1},
                       ignore_index=True)

    for i in range(len(data['4_Out_']['All'])):
        x1 = data['4_Out_']['All'][i]
        df = df.append({'Group': 'All', 'Network 3': x1},
                       ignore_index=True)

    for i in range(len(data['6_Out_']['All'])):
        x1 = data['6_Out_']['All'][i]
        df = df.append({'Group': 'All', 'Network 4': x1},
                       ignore_index=True)

    print(df)
    dd = pd.melt(df, id_vars=['Group'], value_vars=['Network 1', 'Network 2', 'Network 3', 'Network 4'], var_name='Networks')
    plt.figure(figsize=(14, 10))
    testPlot = sns.boxplot(x='Group', y='value', data=dd, hue='Networks', showmeans=True, fliersize=1,
                    meanprops={"marker": "s", "markersize": 10, "markerfacecolor": "black", "markeredgecolor": "black"})

    # Shrink current axis by 20%
    box = testPlot.get_position()
    testPlot.set_position([box.x0, box.y0, box.width * 0.8, box.height])

    # Put a legend to the right of the current axis
    testPlot.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    testPlot.set_ylim(-7, 30)

    ind = 0
    for tick in range(len(testPlot.get_xticklabels())):

        print(tick)
        testPlot.text(tick - .1, -2, str(np.round(number_affecte_vessels[tick][1] / data['3_Out_Tot']['Number_Vessels'] * 100, 2)) + '\%', horizontalalignment='center', color='orange', rotation=70, size=10)
        testPlot.text(tick + .1, -2, str(np.round(number_affecte_vessels[tick][2] / data['4_Out_Tot']['Number_Vessels'] * 100, 2)) + '\%', horizontalalignment='center', color='green', rotation=70, size=10)
        testPlot.text(tick + .3, -2, str(np.round(number_affecte_vessels[tick][3] / data['6_Out_Tot']['Number_Vessels'] * 100, 2)) + '\%', horizontalalignment='center', color='red', rotation=70, size=10)
        testPlot.text(tick - .3, -2, str(np.round(number_affecte_vessels[tick][0] / data['0_Out_Tot']['Number_Vessels'] * 100, 2)) + '\%', horizontalalignment='center', color='blue', rotation=70, size=10)
        ind += 2

    # plt.legend(loc='upper right')
    plt.savefig(
         r'D:\00 Privat\01_Bildung\01_ETH Zürich\MSc\00_Masterarbeit\02_Network_Study_Small\Adjoint\Flow_Change_R100_to_ALL\Rest\flow_diameter_all_types.png')


# Box Plot Flow All Vessel Types ACTIVATED REGION


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


def get_flow_data_for_different_networks_ACTIVATED_REGION(percent):

    path = r'D:\00 Privat\01_Bildung\01_ETH Zürich\MSc\00_Masterarbeit\02_Network_Study_Small\Adjoint'

    path_0_graph = r'D:\00 Privat\01_Bildung\01_ETH Zürich\MSc\00_Masterarbeit\02_Network_Study_Small\Networks\0\graph.pkl'
    path_3_graph = r'D:\00 Privat\01_Bildung\01_ETH Zürich\MSc\00_Masterarbeit\02_Network_Study_Small\Networks\3\graph.pkl'
    path_4_graph = r'D:\00 Privat\01_Bildung\01_ETH Zürich\MSc\00_Masterarbeit\02_Network_Study_Small\Networks\4\graph.pkl'
    path_6_graph = r'D:\00 Privat\01_Bildung\01_ETH Zürich\MSc\00_Masterarbeit\02_Network_Study_Small\Networks\6\graph.pkl'

    networks = ['0_Out_', '3_Out_', '4_Out_', '6_Out_']
    # networks = ['0_Out_']
    histogram_data_id = ['R_100', 'R_125', 'R_150', 'R_200', 'All']
    # histogram_data_id = ['All']

    coor = {'x': 200, 'y': 400, 'z': 400}
    rr = 80

    data_network_0 = {}
    data_network_3 = {}
    data_network_4 = {}
    data_network_6 = {}

    data = {'0_Out_': {}, '3_Out_': {}, '4_Out_': {}, '6_Out_': {}, '0_Out_Tot': {}, '3_Out_Tot': {}, '4_Out_Tot': {},
            '6_Out_Tot': {}}

    for network in networks:

        graph_ = ig.Graph()

        if network == '0_Out_':

            graph_ = ig.Graph.Read_Pickle(path_0_graph)

        elif network == '3_Out_':

            graph_ = ig.Graph.Read_Pickle(path_3_graph)

        elif network == '4_Out_':

            graph_ = ig.Graph.Read_Pickle(path_4_graph)

        elif network == '6_Out_':

            graph_ = ig.Graph.Read_Pickle(path_6_graph)

        print(network)

        for radius in histogram_data_id:

            current_input_path = path + '\\' + network + '\\' + radius + '\\' + 'in\\adjointMethod' + '\\'
            current_output_path = path + '\\' + network + '\\' + radius + '\\' + 'out' + '\\'
            meshdata_start = current_output_path + 'meshdata_9999.csv'

            r = csv.reader(open(current_input_path + 'activated_eids.csv'))
            lines = list(r)
            activated_eids = lines[0]
            activated_eids = list(map(int, activated_eids))

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
            data[network + 'Tot']['Number_Vessels'] = len(t_new_array)

            f_start = np.array(df_start['tav_Fplasma'])
            f_final = np.array(df_final['tav_Fplasma'])

            f_start_new = []
            f_end_new = []

            for i in range(len(f_start)):

                if (i % 2) == 0:
                    f_start_new.append(f_start[i])
                    f_end_new.append((f_final[i]))

            f_start_new_array = np.asarray(f_start_new)
            f_end_new_array = np.asarray(f_end_new)

            f_ratio = np.abs(f_start_new_array / f_end_new_array)

            edges_in_activated_region = define_activated_region(graph_, coor, rr)
            relevant_edges_flow_higher_1_100 = exclude_edges_with_too_small_flows(network)

            activated_eids_relevant = []

            for edge in edges_in_activated_region:

                if edge in relevant_edges_flow_higher_1_100:

                    activated_eids_relevant.append(edge)

            # for activated_edge in activated_eids:
            #
            #     try:
            #         relevant_edges_flow_higher_1_100.remove(activated_edge)
            #
            #     except:
            #
            #         print("Too Small Baselineflow in this Edge. Already deleted ")

            flow_changes_reacting_edges = f_ratio[activated_eids_relevant]
            flow_changes_reacting_edges_percent = np.abs((flow_changes_reacting_edges - 1) * 100)

            # HIER KANN BEGRENZUNG EINGEBAUEN WERDEN

            grenze_prozentuela_anderung = percent
            flow_changes_selection_index = np.where(flow_changes_reacting_edges_percent > grenze_prozentuela_anderung)
            flow_changes_selection_FINAL = (flow_changes_reacting_edges_percent[flow_changes_selection_index[0]])

            # print(diameter_changes_selection_FINAL)
            data[network][radius] = flow_changes_selection_FINAL

    # print(len(data_r_100), data_r_100)
    # print(len(data_r_150), data_r_150)
    # print(len(data_r_200), data_r_200)
    # print(len(data_r_300), data_r_300)
    # print(len(data_r_all), data_r_all)

    return data


def plot_flow_all_vessel_types_NEW_ACTIVATED_REGION(prozent):

    print('Start Plotting')

    data = get_flow_data_for_different_networks_ACTIVATED_REGION(prozent)

    print('recieved data')
    df = pd.DataFrame(columns=['Group', 'Network 1', 'Network 2', 'Network 3', 'Network 4', ''])

    number_affected_vessels_100 = [len(data['0_Out_']['R_100']), len(data['3_Out_']['R_100']),
                                   len(data['4_Out_']['R_100']), len(data['6_Out_']['R_100'])]

    number_affected_vessels_125 = [len(data['0_Out_']['R_125']), len(data['3_Out_']['R_125']),
                                   len(data['4_Out_']['R_125']), len(data['6_Out_']['R_125'])]

    number_affected_vessels_150 = [len(data['0_Out_']['R_150']), len(data['3_Out_']['R_150']),
                                   len(data['4_Out_']['R_150']), len(data['6_Out_']['R_150'])]

    number_affected_vessels_200 = [len(data['0_Out_']['R_200']), len(data['3_Out_']['R_200']),
                                   len(data['4_Out_']['R_200']), len(data['6_Out_']['R_200'])]

    number_affected_vessels_all = [len(data['0_Out_']['All']), len(data['3_Out_']['All']),
                                   len(data['4_Out_']['All']), len(data['6_Out_']['All'])]

    number_affecte_vessels = [number_affected_vessels_100, number_affected_vessels_125, number_affected_vessels_150,
                              number_affected_vessels_200, number_affected_vessels_all]

    # R_100

    print('collect r100 data')
    for i in range(len(data['0_Out_']['R_100'])):

        x1 = data['0_Out_']['R_100'][i]
        df = df.append({'Group': 'R100', 'Network 1': x1},
                       ignore_index=True)

    for i in range(len(data['3_Out_']['R_100'])):

        x1 = data['3_Out_']['R_100'][i]
        df = df.append({'Group': 'R100', 'Network 2': x1},
                       ignore_index=True)

    for i in range(len(data['4_Out_']['R_100'])):

        x1 = data['4_Out_']['R_100'][i]
        df = df.append({'Group': 'R100', 'Network 3': x1},
                       ignore_index=True)

    for i in range(len(data['6_Out_']['R_100'])):

        x1 = data['6_Out_']['R_100'][i]
        df = df.append({'Group': 'R100', 'Network 4': x1},
                       ignore_index=True)

     # R 125

    print('collect r125 data')

    for i in range(len(data['0_Out_']['R_125'])):
        x1 = data['0_Out_']['R_125'][i]
        df = df.append({'Group': 'R125', 'Network 1': x1},
                       ignore_index=True)

    for i in range(len(data['3_Out_']['R_125'])):
        x1 = data['3_Out_']['R_125'][i]
        df = df.append({'Group': 'R125', 'Network 2': x1},
                       ignore_index=True)

    for i in range(len(data['4_Out_']['R_125'])):
        x1 = data['4_Out_']['R_125'][i]
        df = df.append({'Group': 'R125', 'Network 3': x1},
                       ignore_index=True)

    for i in range(len(data['6_Out_']['R_125'])):
        x1 = data['6_Out_']['R_125'][i]
        df = df.append({'Group': 'R125', 'Network 4': x1},
                       ignore_index=True)

     # R 150

    print('collect r150 data')
    for i in range(len(data['0_Out_']['R_150'])):
        x1 = data['0_Out_']['R_150'][i]
        df = df.append({'Group': 'R150', 'Network 1': x1},
                       ignore_index=True)

    for i in range(len(data['3_Out_']['R_150'])):
        x1 = data['3_Out_']['R_150'][i]
        df = df.append({'Group': 'R150', 'Network 2': x1},
                       ignore_index=True)

    for i in range(len(data['4_Out_']['R_150'])):
        x1 = data['4_Out_']['R_150'][i]
        df = df.append({'Group': 'R150', 'Network 3': x1},
                       ignore_index=True)

    for i in range(len(data['6_Out_']['R_150'])):
        x1 = data['6_Out_']['R_150'][i]
        df = df.append({'Group': 'R150', 'Network 4': x1},
                       ignore_index=True)

    # R 200

    print('collect r200 data')
    for i in range(len(data['0_Out_']['R_200'])):

        x1 = data['0_Out_']['R_200'][i]
        df = df.append({'Group': 'R200', 'Network 1': x1},
                       ignore_index=True)

    for i in range(len(data['3_Out_']['R_200'])):

        x1 = data['3_Out_']['R_200'][i]
        df = df.append({'Group': 'R200', 'Network 2': x1},
                       ignore_index=True)

    for i in range(len(data['4_Out_']['R_200'])):

        x1 = data['4_Out_']['R_200'][i]
        df = df.append({'Group': 'R200', 'Network 3': x1},
                       ignore_index=True)

    for i in range(len(data['6_Out_']['R_200'])):

        x1 = data['6_Out_']['R_200'][i]
        df = df.append({'Group': 'R200', 'Network 4': x1},
                       ignore_index=True)

    # All

    print('collect all data')
    for i in range(len(data['0_Out_']['All'])):
        x1 = data['0_Out_']['All'][i]
        df = df.append({'Group': 'All', 'Network 1': x1},
                       ignore_index=True)

    for i in range(len(data['3_Out_']['All'])):
        x1 = data['3_Out_']['All'][i]
        df = df.append({'Group': 'All', 'Network 2': x1},
                       ignore_index=True)

    for i in range(len(data['4_Out_']['All'])):
        x1 = data['4_Out_']['All'][i]
        df = df.append({'Group': 'All', 'Network 3': x1},
                       ignore_index=True)

    for i in range(len(data['6_Out_']['All'])):
        x1 = data['6_Out_']['All'][i]
        df = df.append({'Group': 'All', 'Network 4': x1},
                       ignore_index=True)

    print(df)
    dd = pd.melt(df, id_vars=['Group'], value_vars=['Network 1', 'Network 2', 'Network 3', 'Network 4'], var_name='Networks')
    plt.figure(figsize=(14, 10))
    testPlot = sns.boxplot(x='Group', y='value', data=dd, hue='Networks', showmeans=True, fliersize=1,
                    meanprops={"marker": "s", "markersize": 10, "markerfacecolor": "black", "markeredgecolor": "black"})

    # Shrink current axis by 20%
    box = testPlot.get_position()
    testPlot.set_position([box.x0, box.y0, box.width * 0.8, box.height])

    # Put a legend to the right of the current axis
    testPlot.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    testPlot.set_ylim(-12, 100)

    ind = 0
    for tick in range(len(testPlot.get_xticklabels())):

        print(tick)
        testPlot.text(tick - .1, -2, str(np.round(number_affecte_vessels[tick][1] / data['3_Out_Tot']['Number_Vessels'] * 100, 2)) + '\%', horizontalalignment='center', color='orange', rotation=70, size=10)
        testPlot.text(tick + .1, -2, str(np.round(number_affecte_vessels[tick][2] / data['4_Out_Tot']['Number_Vessels'] * 100, 2)) + '\%', horizontalalignment='center', color='green', rotation=70, size=10)
        testPlot.text(tick + .3, -2, str(np.round(number_affecte_vessels[tick][3] / data['6_Out_Tot']['Number_Vessels'] * 100, 2)) + '\%', horizontalalignment='center', color='red', rotation=70, size=10)
        testPlot.text(tick - .3, -2, str(np.round(number_affecte_vessels[tick][0] / data['0_Out_Tot']['Number_Vessels'] * 100, 2)) + '\%', horizontalalignment='center', color='blue', rotation=70, size=10)
        ind += 2

    # plt.legend(loc='upper right')
    plt.savefig(
         r'D:\00 Privat\01_Bildung\01_ETH Zürich\MSc\00_Masterarbeit\02_Network_Study_Small\Adjoint\Flow_Change_R100_to_ALL\Activation_Region\flow_diameter_all_types.png')


# Box Plot Flow Capilleries REST


def get_flow_data_for_different_networks_NOT_activated_region_capillaries(percent):

    path = r'D:\00 Privat\01_Bildung\01_ETH Zürich\MSc\00_Masterarbeit\02_Network_Study_Small\Adjoint'

    path_0_graph = r'D:\00 Privat\01_Bildung\01_ETH Zürich\MSc\00_Masterarbeit\02_Network_Study_Small\Networks\0\graph.pkl'
    path_3_graph = r'D:\00 Privat\01_Bildung\01_ETH Zürich\MSc\00_Masterarbeit\02_Network_Study_Small\Networks\3\graph.pkl'
    path_4_graph = r'D:\00 Privat\01_Bildung\01_ETH Zürich\MSc\00_Masterarbeit\02_Network_Study_Small\Networks\4\graph.pkl'
    path_6_graph = r'D:\00 Privat\01_Bildung\01_ETH Zürich\MSc\00_Masterarbeit\02_Network_Study_Small\Networks\6\graph.pkl'

    networks = ['0_Out_', '3_Out_', '4_Out_', '6_Out_']
    # networks = ['0_Out_']
    histogram_data_id = ['R_100', 'R_125', 'R_150', 'R_200', 'All']
    # histogram_data_id = ['All']

    data_network_0 = {}
    data_network_3 = {}
    data_network_4 = {}
    data_network_6 = {}

    data = {'0_Out_': {}, '3_Out_': {}, '4_Out_': {}, '6_Out_': {}, '0_Out_Tot': {}, '3_Out_Tot': {}, '4_Out_Tot': {},
            '6_Out_Tot': {}}

    for network in networks:

        graph_ = ig.Graph()

        if network == '0_Out_':

            graph_ = ig.Graph.Read_Pickle(path_0_graph)

        elif network == '3_Out_':

            graph_ = ig.Graph.Read_Pickle(path_3_graph)

        elif network == '4_Out_':

            graph_ = ig.Graph.Read_Pickle(path_4_graph)

        elif network == '6_Out_':

            graph_ = ig.Graph.Read_Pickle(path_6_graph)

        print(network)

        for radius in histogram_data_id:

            current_input_path = path + '\\' + network + '\\' + radius + '\\' + 'in\\adjointMethod' + '\\'
            current_output_path = path + '\\' + network + '\\' + radius + '\\' + 'out' + '\\'
            meshdata_start = current_output_path + 'meshdata_9999.csv'

            r = csv.reader(open(current_input_path + 'activated_eids.csv'))
            lines = list(r)
            activated_eids = lines[0]
            activated_eids = list(map(int, activated_eids))

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
            data[network + 'Tot']['Number_Vessels'] = len(t_new_array)

            f_start = np.array(df_start['tav_Fplasma'])
            f_final = np.array(df_final['tav_Fplasma'])

            f_start_new = []
            f_end_new = []

            for i in range(len(f_start)):

                if (i % 2) == 0:
                    f_start_new.append(f_start[i])
                    f_end_new.append((f_final[i]))

            f_start_new_array = np.asarray(f_start_new)
            f_end_new_array = np.asarray(f_end_new)

            f_ratio = np.abs(f_start_new_array / f_end_new_array)
            relevant_edges_flow_higher_1_100 = exclude_edges_with_too_small_flows(network)

            for activated_edge in activated_eids:

                try:
                    relevant_edges_flow_higher_1_100.remove(activated_edge)

                except:

                    print("Too Small Baselineflow in this Edge. Already deleted ")

            relevant_edges_capillaries_only = []

            for edge in relevant_edges_flow_higher_1_100:

                if graph_.es[edge]['Type'] == 0 or graph_.es[edge]['Type'] == 3:

                    relevant_edges_capillaries_only.append(edge)

            flow_changes_reacting_edges = f_ratio[relevant_edges_capillaries_only]
            flow_changes_reacting_edges_percent = np.abs((flow_changes_reacting_edges - 1) * 100)

            # HIER KANN BEGRENZUNG EINGEBAUEN WERDEN

            grenze_prozentuela_anderung = percent
            flow_changes_selection_index = np.where(flow_changes_reacting_edges_percent > grenze_prozentuela_anderung)
            flow_changes_selection_FINAL = (flow_changes_reacting_edges_percent[flow_changes_selection_index[0]])

            a = np.where(flow_changes_selection_FINAL < 1000)

            flow_changes_selection_FINAL = flow_changes_selection_FINAL[a]

            # print(diameter_changes_selection_FINAL)
            data[network][radius] = flow_changes_selection_FINAL

    # print(len(data_r_100), data_r_100)
    # print(len(data_r_150), data_r_150)
    # print(len(data_r_200), data_r_200)
    # print(len(data_r_300), data_r_300)
    # print(len(data_r_all), data_r_all)

    return data


def plot_flow_all_vessel_types_NEW_capillaries(prozent):

    print('Start Plotting')

    data = get_flow_data_for_different_networks_NOT_activated_region_capillaries(prozent)

    print('recieved data')
    df = pd.DataFrame(columns=['Group', 'Network 1', 'Network 2', 'Network 3', 'Network 4', ''])

    number_affected_vessels_100 = [len(data['0_Out_']['R_100']), len(data['3_Out_']['R_100']),
                                   len(data['4_Out_']['R_100']), len(data['6_Out_']['R_100'])]

    number_affected_vessels_125 = [len(data['0_Out_']['R_125']), len(data['3_Out_']['R_125']),
                                   len(data['4_Out_']['R_125']), len(data['6_Out_']['R_125'])]

    number_affected_vessels_150 = [len(data['0_Out_']['R_150']), len(data['3_Out_']['R_150']),
                                   len(data['4_Out_']['R_150']), len(data['6_Out_']['R_150'])]

    number_affected_vessels_200 = [len(data['0_Out_']['R_200']), len(data['3_Out_']['R_200']),
                                   len(data['4_Out_']['R_200']), len(data['6_Out_']['R_200'])]

    number_affected_vessels_all = [len(data['0_Out_']['All']), len(data['3_Out_']['All']),
                                   len(data['4_Out_']['All']), len(data['6_Out_']['All'])]

    number_affecte_vessels = [number_affected_vessels_100, number_affected_vessels_125, number_affected_vessels_150,
                              number_affected_vessels_200, number_affected_vessels_all]

    # R_100

    print('collect r100 data')
    for i in range(len(data['0_Out_']['R_100'])):

        x1 = data['0_Out_']['R_100'][i]
        df = df.append({'Group': 'R100', 'Network 1': x1},
                       ignore_index=True)

    for i in range(len(data['3_Out_']['R_100'])):

        x1 = data['3_Out_']['R_100'][i]
        df = df.append({'Group': 'R100', 'Network 2': x1},
                       ignore_index=True)

    for i in range(len(data['4_Out_']['R_100'])):

        x1 = data['4_Out_']['R_100'][i]
        df = df.append({'Group': 'R100', 'Network 3': x1},
                       ignore_index=True)

    for i in range(len(data['6_Out_']['R_100'])):

        x1 = data['6_Out_']['R_100'][i]
        df = df.append({'Group': 'R100', 'Network 4': x1},
                       ignore_index=True)

     # R 125

    print('collect r125 data')

    for i in range(len(data['0_Out_']['R_125'])):
        x1 = data['0_Out_']['R_125'][i]
        df = df.append({'Group': 'R125', 'Network 1': x1},
                       ignore_index=True)

    for i in range(len(data['3_Out_']['R_125'])):
        x1 = data['3_Out_']['R_125'][i]
        df = df.append({'Group': 'R125', 'Network 2': x1},
                       ignore_index=True)

    for i in range(len(data['4_Out_']['R_125'])):
        x1 = data['4_Out_']['R_125'][i]
        df = df.append({'Group': 'R125', 'Network 3': x1},
                       ignore_index=True)

    for i in range(len(data['6_Out_']['R_125'])):
        x1 = data['6_Out_']['R_125'][i]
        df = df.append({'Group': 'R125', 'Network 4': x1},
                       ignore_index=True)

     # R 150

    print('collect r150 data')
    for i in range(len(data['0_Out_']['R_150'])):
        x1 = data['0_Out_']['R_150'][i]
        df = df.append({'Group': 'R150', 'Network 1': x1},
                       ignore_index=True)

    for i in range(len(data['3_Out_']['R_150'])):
        x1 = data['3_Out_']['R_150'][i]
        df = df.append({'Group': 'R150', 'Network 2': x1},
                       ignore_index=True)

    for i in range(len(data['4_Out_']['R_150'])):
        x1 = data['4_Out_']['R_150'][i]
        df = df.append({'Group': 'R150', 'Network 3': x1},
                       ignore_index=True)

    for i in range(len(data['6_Out_']['R_150'])):
        x1 = data['6_Out_']['R_150'][i]
        df = df.append({'Group': 'R150', 'Network 4': x1},
                       ignore_index=True)

    # R 200

    print('collect r200 data')
    for i in range(len(data['0_Out_']['R_200'])):

        x1 = data['0_Out_']['R_200'][i]
        df = df.append({'Group': 'R200', 'Network 1': x1},
                       ignore_index=True)

    for i in range(len(data['3_Out_']['R_200'])):

        x1 = data['3_Out_']['R_200'][i]
        df = df.append({'Group': 'R200', 'Network 2': x1},
                       ignore_index=True)

    for i in range(len(data['4_Out_']['R_200'])):

        x1 = data['4_Out_']['R_200'][i]
        df = df.append({'Group': 'R200', 'Network 3': x1},
                       ignore_index=True)

    for i in range(len(data['6_Out_']['R_200'])):

        x1 = data['6_Out_']['R_200'][i]
        df = df.append({'Group': 'R200', 'Network 4': x1},
                       ignore_index=True)

    # All

    print('collect all data')
    for i in range(len(data['0_Out_']['All'])):
        x1 = data['0_Out_']['All'][i]
        df = df.append({'Group': 'All', 'Network 1': x1},
                       ignore_index=True)

    for i in range(len(data['3_Out_']['All'])):
        x1 = data['3_Out_']['All'][i]
        df = df.append({'Group': 'All', 'Network 2': x1},
                       ignore_index=True)

    for i in range(len(data['4_Out_']['All'])):
        x1 = data['4_Out_']['All'][i]
        df = df.append({'Group': 'All', 'Network 3': x1},
                       ignore_index=True)

    for i in range(len(data['6_Out_']['All'])):
        x1 = data['6_Out_']['All'][i]
        df = df.append({'Group': 'All', 'Network 4': x1},
                       ignore_index=True)

    print(df)
    dd = pd.melt(df, id_vars=['Group'], value_vars=['Network 1', 'Network 2', 'Network 3', 'Network 4'], var_name='Networks')
    plt.figure(figsize=(14, 10))
    testPlot = sns.boxplot(x='Group', y='value', data=dd, hue='Networks', showmeans=True, fliersize=1,
                    meanprops={"marker": "s", "markersize": 10, "markerfacecolor": "black", "markeredgecolor": "black"})

    # Shrink current axis by 20%
    box = testPlot.get_position()
    testPlot.set_position([box.x0, box.y0, box.width * 0.8, box.height])

    # Put a legend to the right of the current axis
    testPlot.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    testPlot.set_ylim(-7, 30)

    ind = 0
    for tick in range(len(testPlot.get_xticklabels())):

        print(tick)
        testPlot.text(tick - .1, -2, str(np.round(number_affecte_vessels[tick][1] / data['3_Out_Tot']['Number_Vessels'] * 100, 2)) + '\%', horizontalalignment='center', color='orange', rotation=70, size=10)
        testPlot.text(tick + .1, -2, str(np.round(number_affecte_vessels[tick][2] / data['4_Out_Tot']['Number_Vessels'] * 100, 2)) + '\%', horizontalalignment='center', color='green', rotation=70, size=10)
        testPlot.text(tick + .3, -2, str(np.round(number_affecte_vessels[tick][3] / data['6_Out_Tot']['Number_Vessels'] * 100, 2)) + '\%', horizontalalignment='center', color='red', rotation=70, size=10)
        testPlot.text(tick - .3, -2, str(np.round(number_affecte_vessels[tick][0] / data['0_Out_Tot']['Number_Vessels'] * 100, 2)) + '\%', horizontalalignment='center', color='blue', rotation=70, size=10)
        ind += 2

    # plt.legend(loc='upper right')
    plt.savefig(
         r'D:\00 Privat\01_Bildung\01_ETH Zürich\MSc\00_Masterarbeit\02_Network_Study_Small\Adjoint\Flow_Change_R100_to_ALL\Rest\flow_diameter_capillaries.png')
    # plt.show()


# Box Plot Flow Arterioles Venules REST


def get_flow_data_for_different_networks_NOT_activated_region_arterioles_and_venules(percent):

    path = r'D:\00 Privat\01_Bildung\01_ETH Zürich\MSc\00_Masterarbeit\02_Network_Study_Small\Adjoint'

    path_0_graph = r'D:\00 Privat\01_Bildung\01_ETH Zürich\MSc\00_Masterarbeit\02_Network_Study_Small\Networks\0\graph.pkl'
    path_3_graph = r'D:\00 Privat\01_Bildung\01_ETH Zürich\MSc\00_Masterarbeit\02_Network_Study_Small\Networks\3\graph.pkl'
    path_4_graph = r'D:\00 Privat\01_Bildung\01_ETH Zürich\MSc\00_Masterarbeit\02_Network_Study_Small\Networks\4\graph.pkl'
    path_6_graph = r'D:\00 Privat\01_Bildung\01_ETH Zürich\MSc\00_Masterarbeit\02_Network_Study_Small\Networks\6\graph.pkl'

    networks = ['0_Out_', '3_Out_', '4_Out_', '6_Out_']
    # networks = ['0_Out_']
    histogram_data_id = ['R_100', 'R_125', 'R_150', 'R_200', 'All']
    # histogram_data_id = ['All']

    data_network_0 = {}
    data_network_3 = {}
    data_network_4 = {}
    data_network_6 = {}

    data = {'0_Out_': {}, '3_Out_': {}, '4_Out_': {}, '6_Out_': {}, '0_Out_Tot': {}, '3_Out_Tot': {}, '4_Out_Tot': {},
            '6_Out_Tot': {}}

    for network in networks:

        graph_ = ig.Graph()

        if network == '0_Out_':

            graph_ = ig.Graph.Read_Pickle(path_0_graph)

        elif network == '3_Out_':

            graph_ = ig.Graph.Read_Pickle(path_3_graph)

        elif network == '4_Out_':

            graph_ = ig.Graph.Read_Pickle(path_4_graph)

        elif network == '6_Out_':

            graph_ = ig.Graph.Read_Pickle(path_6_graph)

        print(network)

        for radius in histogram_data_id:

            current_input_path = path + '\\' + network + '\\' + radius + '\\' + 'in\\adjointMethod' + '\\'
            current_output_path = path + '\\' + network + '\\' + radius + '\\' + 'out' + '\\'
            meshdata_start = current_output_path + 'meshdata_9999.csv'

            r = csv.reader(open(current_input_path + 'activated_eids.csv'))
            lines = list(r)
            activated_eids = lines[0]
            activated_eids = list(map(int, activated_eids))

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
            data[network + 'Tot']['Number_Vessels'] = len(t_new_array)

            f_start = np.array(df_start['tav_Fplasma'])
            f_final = np.array(df_final['tav_Fplasma'])

            f_start_new = []
            f_end_new = []

            for i in range(len(f_start)):

                if (i % 2) == 0:
                    f_start_new.append(f_start[i])
                    f_end_new.append((f_final[i]))

            f_start_new_array = np.asarray(f_start_new)
            f_end_new_array = np.asarray(f_end_new)

            f_ratio = np.abs(f_start_new_array / f_end_new_array)
            relevant_edges_flow_higher_1_100 = exclude_edges_with_too_small_flows(network)

            for activated_edge in activated_eids:

                try:
                    relevant_edges_flow_higher_1_100.remove(activated_edge)

                except:

                    print("Too Small Baselineflow in this Edge. Already deleted ")

            relevant_edges_arterioles_and_venules_only = []

            for edge in relevant_edges_flow_higher_1_100:

                if graph_.es[edge]['Type'] == 1 or graph_.es[edge]['Type'] == 2:

                    relevant_edges_arterioles_and_venules_only.append(edge)

            flow_changes_reacting_edges = f_ratio[relevant_edges_arterioles_and_venules_only]
            flow_changes_reacting_edges_percent = np.abs((flow_changes_reacting_edges - 1) * 100)

            # HIER KANN BEGRENZUNG EINGEBAUEN WERDEN

            grenze_prozentuela_anderung = percent
            flow_changes_selection_index = np.where(flow_changes_reacting_edges_percent > grenze_prozentuela_anderung)
            flow_changes_selection_FINAL = (flow_changes_reacting_edges_percent[flow_changes_selection_index[0]])

            a = np.where(flow_changes_selection_FINAL < 500)

            flow_changes_selection_FINAL = flow_changes_selection_FINAL[a]

            # print(diameter_changes_selection_FINAL)
            data[network][radius] = flow_changes_selection_FINAL

    # print(len(data_r_100), data_r_100)
    # print(len(data_r_150), data_r_150)
    # print(len(data_r_200), data_r_200)
    # print(len(data_r_300), data_r_300)
    # print(len(data_r_all), data_r_all)

    return data


def plot_flow_all_vessel_types_NEW_arterioles_and_venules(prozent):

    print('Start Plotting')

    data = get_flow_data_for_different_networks_NOT_activated_region_arterioles_and_venules(prozent)

    print('recieved data')
    df = pd.DataFrame(columns=['Group', 'Network 1', 'Network 2', 'Network 3', 'Network 4', ''])

    number_affected_vessels_100 = [len(data['0_Out_']['R_100']), len(data['3_Out_']['R_100']),
                                   len(data['4_Out_']['R_100']), len(data['6_Out_']['R_100'])]

    number_affected_vessels_125 = [len(data['0_Out_']['R_125']), len(data['3_Out_']['R_125']),
                                   len(data['4_Out_']['R_125']), len(data['6_Out_']['R_125'])]

    number_affected_vessels_150 = [len(data['0_Out_']['R_150']), len(data['3_Out_']['R_150']),
                                   len(data['4_Out_']['R_150']), len(data['6_Out_']['R_150'])]

    number_affected_vessels_200 = [len(data['0_Out_']['R_200']), len(data['3_Out_']['R_200']),
                                   len(data['4_Out_']['R_200']), len(data['6_Out_']['R_200'])]

    number_affected_vessels_all = [len(data['0_Out_']['All']), len(data['3_Out_']['All']),
                                   len(data['4_Out_']['All']), len(data['6_Out_']['All'])]

    number_affecte_vessels = [number_affected_vessels_100, number_affected_vessels_125, number_affected_vessels_150,
                              number_affected_vessels_200, number_affected_vessels_all]

    # R_100

    print('collect r100 data')
    for i in range(len(data['0_Out_']['R_100'])):

        x1 = data['0_Out_']['R_100'][i]
        df = df.append({'Group': 'R100', 'Network 1': x1},
                       ignore_index=True)

    for i in range(len(data['3_Out_']['R_100'])):

        x1 = data['3_Out_']['R_100'][i]
        df = df.append({'Group': 'R100', 'Network 2': x1},
                       ignore_index=True)

    for i in range(len(data['4_Out_']['R_100'])):

        x1 = data['4_Out_']['R_100'][i]
        df = df.append({'Group': 'R100', 'Network 3': x1},
                       ignore_index=True)

    for i in range(len(data['6_Out_']['R_100'])):

        x1 = data['6_Out_']['R_100'][i]
        df = df.append({'Group': 'R100', 'Network 4': x1},
                       ignore_index=True)

     # R 125

    print('collect r125 data')

    for i in range(len(data['0_Out_']['R_125'])):
        x1 = data['0_Out_']['R_125'][i]
        df = df.append({'Group': 'R125', 'Network 1': x1},
                       ignore_index=True)

    for i in range(len(data['3_Out_']['R_125'])):
        x1 = data['3_Out_']['R_125'][i]
        df = df.append({'Group': 'R125', 'Network 2': x1},
                       ignore_index=True)

    for i in range(len(data['4_Out_']['R_125'])):
        x1 = data['4_Out_']['R_125'][i]
        df = df.append({'Group': 'R125', 'Network 3': x1},
                       ignore_index=True)

    for i in range(len(data['6_Out_']['R_125'])):
        x1 = data['6_Out_']['R_125'][i]
        df = df.append({'Group': 'R125', 'Network 4': x1},
                       ignore_index=True)

     # R 150

    print('collect r150 data')
    for i in range(len(data['0_Out_']['R_150'])):
        x1 = data['0_Out_']['R_150'][i]
        df = df.append({'Group': 'R150', 'Network 1': x1},
                       ignore_index=True)

    for i in range(len(data['3_Out_']['R_150'])):
        x1 = data['3_Out_']['R_150'][i]
        df = df.append({'Group': 'R150', 'Network 2': x1},
                       ignore_index=True)

    for i in range(len(data['4_Out_']['R_150'])):
        x1 = data['4_Out_']['R_150'][i]
        df = df.append({'Group': 'R150', 'Network 3': x1},
                       ignore_index=True)

    for i in range(len(data['6_Out_']['R_150'])):
        x1 = data['6_Out_']['R_150'][i]
        df = df.append({'Group': 'R150', 'Network 4': x1},
                       ignore_index=True)

    # R 200

    print('collect r200 data')
    for i in range(len(data['0_Out_']['R_200'])):

        x1 = data['0_Out_']['R_200'][i]
        df = df.append({'Group': 'R200', 'Network 1': x1},
                       ignore_index=True)

    for i in range(len(data['3_Out_']['R_200'])):

        x1 = data['3_Out_']['R_200'][i]
        df = df.append({'Group': 'R200', 'Network 2': x1},
                       ignore_index=True)

    for i in range(len(data['4_Out_']['R_200'])):

        x1 = data['4_Out_']['R_200'][i]
        df = df.append({'Group': 'R200', 'Network 3': x1},
                       ignore_index=True)

    for i in range(len(data['6_Out_']['R_200'])):

        x1 = data['6_Out_']['R_200'][i]
        df = df.append({'Group': 'R200', 'Network 4': x1},
                       ignore_index=True)

    # All

    print('collect all data')
    for i in range(len(data['0_Out_']['All'])):
        x1 = data['0_Out_']['All'][i]
        df = df.append({'Group': 'All', 'Network 1': x1},
                       ignore_index=True)

    for i in range(len(data['3_Out_']['All'])):
        x1 = data['3_Out_']['All'][i]
        df = df.append({'Group': 'All', 'Network 2': x1},
                       ignore_index=True)

    for i in range(len(data['4_Out_']['All'])):
        x1 = data['4_Out_']['All'][i]
        df = df.append({'Group': 'All', 'Network 3': x1},
                       ignore_index=True)

    for i in range(len(data['6_Out_']['All'])):
        x1 = data['6_Out_']['All'][i]
        df = df.append({'Group': 'All', 'Network 4': x1},
                       ignore_index=True)

    print(df)
    dd = pd.melt(df, id_vars=['Group'], value_vars=['Network 1', 'Network 2', 'Network 3', 'Network 4'], var_name='Networks')
    plt.figure(figsize=(14, 10))
    testPlot = sns.boxplot(x='Group', y='value', data=dd, hue='Networks', showmeans=True, fliersize=1,
                    meanprops={"marker": "s", "markersize": 10, "markerfacecolor": "black", "markeredgecolor": "black"})

    # Shrink current axis by 20%
    box = testPlot.get_position()
    testPlot.set_position([box.x0, box.y0, box.width * 0.8, box.height])

    # Put a legend to the right of the current axis
    testPlot.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    testPlot.set_ylim(-7, 30)

    ind = 0
    for tick in range(len(testPlot.get_xticklabels())):

        print(tick)
        testPlot.text(tick - .1, -2, str(np.round(number_affecte_vessels[tick][1] / data['3_Out_Tot']['Number_Vessels'] * 100, 2)) + '\%', horizontalalignment='center', color='orange', rotation=70, size=10)
        testPlot.text(tick + .1, -2, str(np.round(number_affecte_vessels[tick][2] / data['4_Out_Tot']['Number_Vessels'] * 100, 2)) + '\%', horizontalalignment='center', color='green', rotation=70, size=10)
        testPlot.text(tick + .3, -2, str(np.round(number_affecte_vessels[tick][3] / data['6_Out_Tot']['Number_Vessels'] * 100, 2)) + '\%', horizontalalignment='center', color='red', rotation=70, size=10)
        testPlot.text(tick - .3, -2, str(np.round(number_affecte_vessels[tick][0] / data['0_Out_Tot']['Number_Vessels'] * 100, 2)) + '\%', horizontalalignment='center', color='blue', rotation=70, size=10)
        ind += 2

    # plt.legend(loc='upper right')
    plt.savefig(
         r'D:\00 Privat\01_Bildung\01_ETH Zürich\MSc\00_Masterarbeit\02_Network_Study_Small\Adjoint\Flow_Change_R100_to_ALL\Rest\flow_diameter_arterioles_venules.png')
    # plt.show()


# Box Plot Flow Arterioles Venules ACTIVATED REGION


def get_flow_data_for_different_networks_ACTIVATED_REGION_arterioles_and_venules(percent):

    path = r'D:\00 Privat\01_Bildung\01_ETH Zürich\MSc\00_Masterarbeit\02_Network_Study_Small\Adjoint'

    path_0_graph = r'D:\00 Privat\01_Bildung\01_ETH Zürich\MSc\00_Masterarbeit\02_Network_Study_Small\Networks\0\graph.pkl'
    path_3_graph = r'D:\00 Privat\01_Bildung\01_ETH Zürich\MSc\00_Masterarbeit\02_Network_Study_Small\Networks\3\graph.pkl'
    path_4_graph = r'D:\00 Privat\01_Bildung\01_ETH Zürich\MSc\00_Masterarbeit\02_Network_Study_Small\Networks\4\graph.pkl'
    path_6_graph = r'D:\00 Privat\01_Bildung\01_ETH Zürich\MSc\00_Masterarbeit\02_Network_Study_Small\Networks\6\graph.pkl'

    networks = ['0_Out_', '3_Out_', '4_Out_', '6_Out_']
    # networks = ['0_Out_']
    histogram_data_id = ['R_100', 'R_125', 'R_150', 'R_200', 'All']
    # histogram_data_id = ['All']

    coor = {'x': 200, 'y': 400, 'z': 400}
    rr = 80

    data_network_0 = {}
    data_network_3 = {}
    data_network_4 = {}
    data_network_6 = {}

    data = {'0_Out_': {}, '3_Out_': {}, '4_Out_': {}, '6_Out_': {}, '0_Out_Tot': {}, '3_Out_Tot': {}, '4_Out_Tot': {},
            '6_Out_Tot': {}}

    for network in networks:

        graph_ = ig.Graph()

        if network == '0_Out_':

            graph_ = ig.Graph.Read_Pickle(path_0_graph)

        elif network == '3_Out_':

            graph_ = ig.Graph.Read_Pickle(path_3_graph)

        elif network == '4_Out_':

            graph_ = ig.Graph.Read_Pickle(path_4_graph)

        elif network == '6_Out_':

            graph_ = ig.Graph.Read_Pickle(path_6_graph)

        print(network)

        for radius in histogram_data_id:

            current_input_path = path + '\\' + network + '\\' + radius + '\\' + 'in\\adjointMethod' + '\\'
            current_output_path = path + '\\' + network + '\\' + radius + '\\' + 'out' + '\\'
            meshdata_start = current_output_path + 'meshdata_9999.csv'

            r = csv.reader(open(current_input_path + 'activated_eids.csv'))
            lines = list(r)
            activated_eids = lines[0]
            activated_eids = list(map(int, activated_eids))

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
            data[network + 'Tot']['Number_Vessels'] = len(t_new_array)

            f_start = np.array(df_start['tav_Fplasma'])
            f_final = np.array(df_final['tav_Fplasma'])

            f_start_new = []
            f_end_new = []

            for i in range(len(f_start)):

                if (i % 2) == 0:
                    f_start_new.append(f_start[i])
                    f_end_new.append((f_final[i]))

            f_start_new_array = np.asarray(f_start_new)
            f_end_new_array = np.asarray(f_end_new)

            f_ratio = np.abs(f_start_new_array / f_end_new_array)

            # activated edges from csv files contain capillaries only

            edges_in_activated_region = define_activated_region(graph_, coor, rr)

            arterioles_venules_in_activated_region = []

            for edge in edges_in_activated_region:

                if graph_.es[edge]['Type'] == 1 or graph_.es[edge]['Type'] == 2:

                    arterioles_venules_in_activated_region.append(edge)

            relevant_edges_flow_higher_1_100 = exclude_edges_with_too_small_flows(network)

            activated_eids_relevant = []

            for edge in arterioles_venules_in_activated_region:

                if edge in relevant_edges_flow_higher_1_100:

                    activated_eids_relevant.append(edge)

            flow_changes_reacting_edges = f_ratio[activated_eids_relevant]
            flow_changes_reacting_edges_percent = np.abs((flow_changes_reacting_edges - 1) * 100)

            # HIER KANN BEGRENZUNG EINGEBAUEN WERDEN

            grenze_prozentuela_anderung = percent
            flow_changes_selection_index = np.where(flow_changes_reacting_edges_percent > grenze_prozentuela_anderung)
            flow_changes_selection_FINAL = (flow_changes_reacting_edges_percent[flow_changes_selection_index[0]])

            # print(diameter_changes_selection_FINAL)
            data[network][radius] = flow_changes_selection_FINAL

    # print(len(data_r_100), data_r_100)
    # print(len(data_r_150), data_r_150)
    # print(len(data_r_200), data_r_200)
    # print(len(data_r_300), data_r_300)
    # print(len(data_r_all), data_r_all)

    return data


def plot_flow_all_vessel_types_NEW_ACTIVATED_REGION_arterioles_and_venules(prozent):

    print('Start Plotting')

    data = get_flow_data_for_different_networks_ACTIVATED_REGION_arterioles_and_venules(prozent)

    print('recieved data')
    df = pd.DataFrame(columns=['Group', 'Network 1', 'Network 2', 'Network 3', 'Network 4', ''])

    number_affected_vessels_100 = [len(data['0_Out_']['R_100']), len(data['3_Out_']['R_100']),
                                   len(data['4_Out_']['R_100']), len(data['6_Out_']['R_100'])]

    number_affected_vessels_125 = [len(data['0_Out_']['R_125']), len(data['3_Out_']['R_125']),
                                   len(data['4_Out_']['R_125']), len(data['6_Out_']['R_125'])]

    number_affected_vessels_150 = [len(data['0_Out_']['R_150']), len(data['3_Out_']['R_150']),
                                   len(data['4_Out_']['R_150']), len(data['6_Out_']['R_150'])]

    number_affected_vessels_200 = [len(data['0_Out_']['R_200']), len(data['3_Out_']['R_200']),
                                   len(data['4_Out_']['R_200']), len(data['6_Out_']['R_200'])]

    number_affected_vessels_all = [len(data['0_Out_']['All']), len(data['3_Out_']['All']),
                                   len(data['4_Out_']['All']), len(data['6_Out_']['All'])]

    number_affecte_vessels = [number_affected_vessels_100, number_affected_vessels_125, number_affected_vessels_150,
                              number_affected_vessels_200, number_affected_vessels_all]

    # R_100

    print('collect r100 data')
    for i in range(len(data['0_Out_']['R_100'])):

        x1 = data['0_Out_']['R_100'][i]
        df = df.append({'Group': 'R100', 'Network 1': x1},
                       ignore_index=True)

    for i in range(len(data['3_Out_']['R_100'])):

        x1 = data['3_Out_']['R_100'][i]
        df = df.append({'Group': 'R100', 'Network 2': x1},
                       ignore_index=True)

    for i in range(len(data['4_Out_']['R_100'])):

        x1 = data['4_Out_']['R_100'][i]
        df = df.append({'Group': 'R100', 'Network 3': x1},
                       ignore_index=True)

    for i in range(len(data['6_Out_']['R_100'])):

        x1 = data['6_Out_']['R_100'][i]
        df = df.append({'Group': 'R100', 'Network 4': x1},
                       ignore_index=True)

     # R 125

    print('collect r125 data')

    for i in range(len(data['0_Out_']['R_125'])):
        x1 = data['0_Out_']['R_125'][i]
        df = df.append({'Group': 'R125', 'Network 1': x1},
                       ignore_index=True)

    for i in range(len(data['3_Out_']['R_125'])):
        x1 = data['3_Out_']['R_125'][i]
        df = df.append({'Group': 'R125', 'Network 2': x1},
                       ignore_index=True)

    for i in range(len(data['4_Out_']['R_125'])):
        x1 = data['4_Out_']['R_125'][i]
        df = df.append({'Group': 'R125', 'Network 3': x1},
                       ignore_index=True)

    for i in range(len(data['6_Out_']['R_125'])):
        x1 = data['6_Out_']['R_125'][i]
        df = df.append({'Group': 'R125', 'Network 4': x1},
                       ignore_index=True)

     # R 150

    print('collect r150 data')
    for i in range(len(data['0_Out_']['R_150'])):
        x1 = data['0_Out_']['R_150'][i]
        df = df.append({'Group': 'R150', 'Network 1': x1},
                       ignore_index=True)

    for i in range(len(data['3_Out_']['R_150'])):
        x1 = data['3_Out_']['R_150'][i]
        df = df.append({'Group': 'R150', 'Network 2': x1},
                       ignore_index=True)

    for i in range(len(data['4_Out_']['R_150'])):
        x1 = data['4_Out_']['R_150'][i]
        df = df.append({'Group': 'R150', 'Network 3': x1},
                       ignore_index=True)

    for i in range(len(data['6_Out_']['R_150'])):
        x1 = data['6_Out_']['R_150'][i]
        df = df.append({'Group': 'R150', 'Network 4': x1},
                       ignore_index=True)

    # R 200

    print('collect r200 data')
    for i in range(len(data['0_Out_']['R_200'])):

        x1 = data['0_Out_']['R_200'][i]
        df = df.append({'Group': 'R200', 'Network 1': x1},
                       ignore_index=True)

    for i in range(len(data['3_Out_']['R_200'])):

        x1 = data['3_Out_']['R_200'][i]
        df = df.append({'Group': 'R200', 'Network 2': x1},
                       ignore_index=True)

    for i in range(len(data['4_Out_']['R_200'])):

        x1 = data['4_Out_']['R_200'][i]
        df = df.append({'Group': 'R200', 'Network 3': x1},
                       ignore_index=True)

    for i in range(len(data['6_Out_']['R_200'])):

        x1 = data['6_Out_']['R_200'][i]
        df = df.append({'Group': 'R200', 'Network 4': x1},
                       ignore_index=True)

    # All

    print('collect all data')
    for i in range(len(data['0_Out_']['All'])):
        x1 = data['0_Out_']['All'][i]
        df = df.append({'Group': 'All', 'Network 1': x1},
                       ignore_index=True)

    for i in range(len(data['3_Out_']['All'])):
        x1 = data['3_Out_']['All'][i]
        df = df.append({'Group': 'All', 'Network 2': x1},
                       ignore_index=True)

    for i in range(len(data['4_Out_']['All'])):
        x1 = data['4_Out_']['All'][i]
        df = df.append({'Group': 'All', 'Network 3': x1},
                       ignore_index=True)

    for i in range(len(data['6_Out_']['All'])):
        x1 = data['6_Out_']['All'][i]
        df = df.append({'Group': 'All', 'Network 4': x1},
                       ignore_index=True)

    print(df)
    dd = pd.melt(df, id_vars=['Group'], value_vars=['Network 1', 'Network 2', 'Network 3', 'Network 4'], var_name='Networks')
    plt.figure(figsize=(14, 10))
    testPlot = sns.boxplot(x='Group', y='value', data=dd, hue='Networks', showmeans=True, fliersize=1,
                    meanprops={"marker": "s", "markersize": 10, "markerfacecolor": "black", "markeredgecolor": "black"})

    # Shrink current axis by 20%
    box = testPlot.get_position()
    testPlot.set_position([box.x0, box.y0, box.width * 0.8, box.height])

    # Put a legend to the right of the current axis
    testPlot.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    testPlot.set_ylim(-7, 100)

    ind = 0
    for tick in range(len(testPlot.get_xticklabels())):

        print(tick)
        testPlot.text(tick - .1, -2, str(np.round(number_affecte_vessels[tick][1] / data['3_Out_Tot']['Number_Vessels'] * 100, 2)) + '\%', horizontalalignment='center', color='orange', rotation=70, size=10)
        testPlot.text(tick + .1, -2, str(np.round(number_affecte_vessels[tick][2] / data['4_Out_Tot']['Number_Vessels'] * 100, 2)) + '\%', horizontalalignment='center', color='green', rotation=70, size=10)
        testPlot.text(tick + .3, -2, str(np.round(number_affecte_vessels[tick][3] / data['6_Out_Tot']['Number_Vessels'] * 100, 2)) + '\%', horizontalalignment='center', color='red', rotation=70, size=10)
        testPlot.text(tick - .3, -2, str(np.round(number_affecte_vessels[tick][0] / data['0_Out_Tot']['Number_Vessels'] * 100, 2)) + '\%', horizontalalignment='center', color='blue', rotation=70, size=10)
        ind += 2

    # plt.legend(loc='upper right')
    plt.savefig(
         r'D:\00 Privat\01_Bildung\01_ETH Zürich\MSc\00_Masterarbeit\02_Network_Study_Small\Adjoint\Flow_Change_R100_to_ALL\Activation_Region\flow_diameter_arterioles_venules.png')
    # plt.show()


# Box Plot Flow Capillaries ACTIVATED REGION


def get_flow_data_for_different_networks_ACTIVATED_REGION_capillaries(percent):
    path = r'D:\00 Privat\01_Bildung\01_ETH Zürich\MSc\00_Masterarbeit\02_Network_Study_Small\Adjoint'

    path_0_graph = r'D:\00 Privat\01_Bildung\01_ETH Zürich\MSc\00_Masterarbeit\02_Network_Study_Small\Networks\0\graph.pkl'
    path_3_graph = r'D:\00 Privat\01_Bildung\01_ETH Zürich\MSc\00_Masterarbeit\02_Network_Study_Small\Networks\3\graph.pkl'
    path_4_graph = r'D:\00 Privat\01_Bildung\01_ETH Zürich\MSc\00_Masterarbeit\02_Network_Study_Small\Networks\4\graph.pkl'
    path_6_graph = r'D:\00 Privat\01_Bildung\01_ETH Zürich\MSc\00_Masterarbeit\02_Network_Study_Small\Networks\6\graph.pkl'

    networks = ['0_Out_', '3_Out_', '4_Out_', '6_Out_']
    # networks = ['0_Out_']
    histogram_data_id = ['R_100', 'R_125', 'R_150', 'R_200', 'All']
    # histogram_data_id = ['All']

    data_network_0 = {}
    data_network_3 = {}
    data_network_4 = {}
    data_network_6 = {}

    data = {'0_Out_': {}, '3_Out_': {}, '4_Out_': {}, '6_Out_': {}, '0_Out_Tot': {}, '3_Out_Tot': {}, '4_Out_Tot': {},
            '6_Out_Tot': {}}

    for network in networks:

        graph_ = ig.Graph()

        if network == '0_Out_':

            graph_ = ig.Graph.Read_Pickle(path_0_graph)

        elif network == '3_Out_':

            graph_ = ig.Graph.Read_Pickle(path_3_graph)

        elif network == '4_Out_':

            graph_ = ig.Graph.Read_Pickle(path_4_graph)

        elif network == '6_Out_':

            graph_ = ig.Graph.Read_Pickle(path_6_graph)

        print(network)

        for radius in histogram_data_id:

            current_input_path = path + '\\' + network + '\\' + radius + '\\' + 'in\\adjointMethod' + '\\'
            current_output_path = path + '\\' + network + '\\' + radius + '\\' + 'out' + '\\'
            meshdata_start = current_output_path + 'meshdata_9999.csv'

            r = csv.reader(open(current_input_path + 'activated_eids.csv'))
            lines = list(r)
            activated_eids = lines[0]
            activated_eids = list(map(int, activated_eids))

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
            data[network + 'Tot']['Number_Vessels'] = len(t_new_array)

            f_start = np.array(df_start['tav_Fplasma'])
            f_final = np.array(df_final['tav_Fplasma'])

            f_start_new = []
            f_end_new = []

            for i in range(len(f_start)):

                if (i % 2) == 0:
                    f_start_new.append(f_start[i])
                    f_end_new.append((f_final[i]))

            f_start_new_array = np.asarray(f_start_new)
            f_end_new_array = np.asarray(f_end_new)

            f_ratio = np.abs(f_start_new_array / f_end_new_array)
            relevant_edges_flow_higher_1_100 = exclude_edges_with_too_small_flows(network)

            activated_eids_relevant = []

            for edge in activated_eids:

                if edge in relevant_edges_flow_higher_1_100:
                    activated_eids_relevant.append(edge)

            # for activated_edge in activated_eids:
            #
            #     try:
            #         relevant_edges_flow_higher_1_100.remove(activated_edge)
            #
            #     except:
            #
            #         print("Too Small Baselineflow in this Edge. Already deleted ")

            relevant_edges_capillaries_only = []

            for edge in activated_eids_relevant:

                if graph_.es[edge]['Type'] == 0 or graph_.es[edge]['Type'] == 3:
                    relevant_edges_capillaries_only.append(edge)

            flow_changes_reacting_edges = f_ratio[relevant_edges_capillaries_only]
            flow_changes_reacting_edges_percent = np.abs((flow_changes_reacting_edges - 1) * 100)

            # HIER KANN BEGRENZUNG EINGEBAUEN WERDEN

            grenze_prozentuela_anderung = percent
            flow_changes_selection_index = np.where(flow_changes_reacting_edges_percent > grenze_prozentuela_anderung)
            flow_changes_selection_FINAL = (flow_changes_reacting_edges_percent[flow_changes_selection_index[0]])

            # print(diameter_changes_selection_FINAL)
            data[network][radius] = flow_changes_selection_FINAL

    # print(len(data_r_100), data_r_100)
    # print(len(data_r_150), data_r_150)
    # print(len(data_r_200), data_r_200)
    # print(len(data_r_300), data_r_300)
    # print(len(data_r_all), data_r_all)

    return data


def plot_flow_all_vessel_types_ACTIVATED_REGION_capillaries(prozent):
    print('Start Plotting')

    data = get_flow_data_for_different_networks_ACTIVATED_REGION_capillaries(prozent)

    print('recieved data')
    df = pd.DataFrame(columns=['Group', 'Network 1', 'Network 2', 'Network 3', 'Network 4', ''])

    number_affected_vessels_100 = [len(data['0_Out_']['R_100']), len(data['3_Out_']['R_100']),
                                   len(data['4_Out_']['R_100']), len(data['6_Out_']['R_100'])]

    number_affected_vessels_125 = [len(data['0_Out_']['R_125']), len(data['3_Out_']['R_125']),
                                   len(data['4_Out_']['R_125']), len(data['6_Out_']['R_125'])]

    number_affected_vessels_150 = [len(data['0_Out_']['R_150']), len(data['3_Out_']['R_150']),
                                   len(data['4_Out_']['R_150']), len(data['6_Out_']['R_150'])]

    number_affected_vessels_200 = [len(data['0_Out_']['R_200']), len(data['3_Out_']['R_200']),
                                   len(data['4_Out_']['R_200']), len(data['6_Out_']['R_200'])]

    number_affected_vessels_all = [len(data['0_Out_']['All']), len(data['3_Out_']['All']),
                                   len(data['4_Out_']['All']), len(data['6_Out_']['All'])]

    number_affecte_vessels = [number_affected_vessels_100, number_affected_vessels_125, number_affected_vessels_150,
                              number_affected_vessels_200, number_affected_vessels_all]

    # R_100

    print('collect r100 data')
    for i in range(len(data['0_Out_']['R_100'])):
        x1 = data['0_Out_']['R_100'][i]
        df = df.append({'Group': 'R100', 'Network 1': x1},
                       ignore_index=True)

    for i in range(len(data['3_Out_']['R_100'])):
        x1 = data['3_Out_']['R_100'][i]
        df = df.append({'Group': 'R100', 'Network 2': x1},
                       ignore_index=True)

    for i in range(len(data['4_Out_']['R_100'])):
        x1 = data['4_Out_']['R_100'][i]
        df = df.append({'Group': 'R100', 'Network 3': x1},
                       ignore_index=True)

    for i in range(len(data['6_Out_']['R_100'])):
        x1 = data['6_Out_']['R_100'][i]
        df = df.append({'Group': 'R100', 'Network 4': x1},
                       ignore_index=True)

    # R 125

    print('collect r125 data')

    for i in range(len(data['0_Out_']['R_125'])):
        x1 = data['0_Out_']['R_125'][i]
        df = df.append({'Group': 'R125', 'Network 1': x1},
                       ignore_index=True)

    for i in range(len(data['3_Out_']['R_125'])):
        x1 = data['3_Out_']['R_125'][i]
        df = df.append({'Group': 'R125', 'Network 2': x1},
                       ignore_index=True)

    for i in range(len(data['4_Out_']['R_125'])):
        x1 = data['4_Out_']['R_125'][i]
        df = df.append({'Group': 'R125', 'Network 3': x1},
                       ignore_index=True)

    for i in range(len(data['6_Out_']['R_125'])):
        x1 = data['6_Out_']['R_125'][i]
        df = df.append({'Group': 'R125', 'Network 4': x1},
                       ignore_index=True)

    # R 150

    print('collect r150 data')
    for i in range(len(data['0_Out_']['R_150'])):
        x1 = data['0_Out_']['R_150'][i]
        df = df.append({'Group': 'R150', 'Network 1': x1},
                       ignore_index=True)

    for i in range(len(data['3_Out_']['R_150'])):
        x1 = data['3_Out_']['R_150'][i]
        df = df.append({'Group': 'R150', 'Network 2': x1},
                       ignore_index=True)

    for i in range(len(data['4_Out_']['R_150'])):
        x1 = data['4_Out_']['R_150'][i]
        df = df.append({'Group': 'R150', 'Network 3': x1},
                       ignore_index=True)

    for i in range(len(data['6_Out_']['R_150'])):
        x1 = data['6_Out_']['R_150'][i]
        df = df.append({'Group': 'R150', 'Network 4': x1},
                       ignore_index=True)

    # R 200

    print('collect r200 data')
    for i in range(len(data['0_Out_']['R_200'])):
        x1 = data['0_Out_']['R_200'][i]
        df = df.append({'Group': 'R200', 'Network 1': x1},
                       ignore_index=True)

    for i in range(len(data['3_Out_']['R_200'])):
        x1 = data['3_Out_']['R_200'][i]
        df = df.append({'Group': 'R200', 'Network 2': x1},
                       ignore_index=True)

    for i in range(len(data['4_Out_']['R_200'])):
        x1 = data['4_Out_']['R_200'][i]
        df = df.append({'Group': 'R200', 'Network 3': x1},
                       ignore_index=True)

    for i in range(len(data['6_Out_']['R_200'])):
        x1 = data['6_Out_']['R_200'][i]
        df = df.append({'Group': 'R200', 'Network 4': x1},
                       ignore_index=True)

    # All

    print('collect all data')
    for i in range(len(data['0_Out_']['All'])):
        x1 = data['0_Out_']['All'][i]
        df = df.append({'Group': 'All', 'Network 1': x1},
                       ignore_index=True)

    for i in range(len(data['3_Out_']['All'])):
        x1 = data['3_Out_']['All'][i]
        df = df.append({'Group': 'All', 'Network 2': x1},
                       ignore_index=True)

    for i in range(len(data['4_Out_']['All'])):
        x1 = data['4_Out_']['All'][i]
        df = df.append({'Group': 'All', 'Network 3': x1},
                       ignore_index=True)

    for i in range(len(data['6_Out_']['All'])):
        x1 = data['6_Out_']['All'][i]
        df = df.append({'Group': 'All', 'Network 4': x1},
                       ignore_index=True)

    print(df)
    dd = pd.melt(df, id_vars=['Group'], value_vars=['Network 1', 'Network 2', 'Network 3', 'Network 4'],
                 var_name='Networks')
    plt.figure(figsize=(14, 10))
    testPlot = sns.boxplot(x='Group', y='value', data=dd, hue='Networks', showmeans=True, fliersize=1,
                           meanprops={"marker": "s", "markersize": 10, "markerfacecolor": "black",
                                      "markeredgecolor": "black"})

    # Shrink current axis by 20%
    box = testPlot.get_position()
    testPlot.set_position([box.x0, box.y0, box.width * 0.8, box.height])

    # Put a legend to the right of the current axis
    testPlot.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    testPlot.set_ylim(-10, 100)

    ind = 0
    for tick in range(len(testPlot.get_xticklabels())):
        print(tick)
        testPlot.text(tick - .1, -2, str(
            np.round(number_affecte_vessels[tick][1] / data['3_Out_Tot']['Number_Vessels'] * 100, 2)) + '\%',
                      horizontalalignment='center', color='orange', rotation=70, size=10)
        testPlot.text(tick + .1, -2, str(
            np.round(number_affecte_vessels[tick][2] / data['4_Out_Tot']['Number_Vessels'] * 100, 2)) + '\%',
                      horizontalalignment='center', color='green', rotation=70, size=10)
        testPlot.text(tick + .3, -2, str(
            np.round(number_affecte_vessels[tick][3] / data['6_Out_Tot']['Number_Vessels'] * 100, 2)) + '\%',
                      horizontalalignment='center', color='red', rotation=70, size=10)
        testPlot.text(tick - .3, -2, str(
            np.round(number_affecte_vessels[tick][0] / data['0_Out_Tot']['Number_Vessels'] * 100, 2)) + '\%',
                      horizontalalignment='center', color='blue', rotation=70, size=10)
        ind += 2

    # plt.legend(loc='upper right')
    plt.savefig(
        r'D:\00 Privat\01_Bildung\01_ETH Zürich\MSc\00_Masterarbeit\02_Network_Study_Small\Adjoint\Flow_Change_R100_to_ALL\Activation_Region\flow_diameter_capillaries_.png')
    # plt.show()


########################################################################################################################
#
# Distance Plot
#
########################################################################################################################


# R100_200_300_All with 4 Networks Plot


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
    path_3_graph = r'D:\00 Privat\01_Bildung\01_ETH Zürich\MSc\00_Masterarbeit\02_Network_Study_Small\Networks\3\graph.pkl'
    path_4_graph = r'D:\00 Privat\01_Bildung\01_ETH Zürich\MSc\00_Masterarbeit\02_Network_Study_Small\Networks\4\graph.pkl'
    path_6_graph = r'D:\00 Privat\01_Bildung\01_ETH Zürich\MSc\00_Masterarbeit\02_Network_Study_Small\Networks\6\graph.pkl'

    coord = {'x': 200, 'y': 400, 'z': 400}

    radius_min = 50.0
    radius_max = 400.0
    r_steps = 50
    delta_radius = radius_max - radius_min
    step_radius = delta_radius / r_steps

    radii = np.arange(radius_min, radius_max, step_radius)

    # print(radii)
    # print(len(radii))
    # for radius in radii:
    #     print(radius)
    #     current_selection = define_activated_region(graph_, coord, radius)
    #     print(len(current_selection))
    #     print(current_selection)

    networks = ['0_Out_', '3_Out_', '4_Out_', '6_Out_']
    # networks = ['0_Out_']
    # histogram_data_id = ['R_100', 'R_125', 'R_150', 'R_175', 'R_200', 'R_300', 'All']
    histogram_data_id = ['R_100', 'R_200', 'R_300', 'All']
    # histogram_data_id = ['R_100']

    # Netzwerk 0

    data_r_100_n0 = []
    data_r_200_n0 = []
    data_r_300_n0 = []
    data_r_all_n0 = []

    # Netzwerk 3

    data_r_100_n3 = []
    data_r_200_n3 = []
    data_r_300_n3 = []
    data_r_all_n3 = []

    # Netzwerk 4

    data_r_100_n4 = []
    data_r_200_n4 = []
    data_r_300_n4 = []
    data_r_all_n4 = []

    # Netzwerk 6

    data_r_100_n6 = []
    data_r_200_n6 = []
    data_r_300_n6 = []
    data_r_all_n6 = []

    for case_id in histogram_data_id:

        print(case_id)

        data_r_temp_all = []
        data_r_temp_art = []
        data_r_temp_ven = []
        data_r_temp_cap = []

        for network in networks:

            graph_ = ig.Graph()

            if network == '0_Out_':

                graph_ = ig.Graph.Read_Pickle(path_0_graph)

            elif network == '3_Out_':

                graph_ = ig.Graph.Read_Pickle(path_3_graph)

            elif network == '4_Out_':

                graph_ = ig.Graph.Read_Pickle(path_4_graph)

            elif network == '6_Out_':

                graph_ = ig.Graph.Read_Pickle(path_6_graph)

            print(network)

            # current_input_path = path + '\\' + network + '\\' + r + '\\' + 'in\\adjointMethod' + '\\'
            current_output_path = path + '\\' + network + '\\' + case_id + '\\' + 'out' + '\\'
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

            diameter_change_capillaries = []
            diameter_change_veins = []
            diameter_change_arterioles = []
            diameter_change_all = []

            for radius in radii:

                # print(radius)
                current_selection = define_activated_region(graph_, coord, radius)

                current_selection_arterioles = []
                current_selection_venules = []
                current_selection_capillaries = []

                for edge in current_selection:

                    if graph_.es[edge]['Type'] == 0 or graph_.es[edge]['Type'] == 3:

                        current_selection_capillaries.append(edge)

                    elif graph_.es[edge]['Type'] == 1:

                        current_selection_venules.append(edge)

                    elif graph_.es[edge]['Type'] == 2:

                        current_selection_arterioles.append(edge)

                diameter_change_for_selection = t_new_array[current_selection]
                diameter_change_for_selection_cap = t_new_array[current_selection_capillaries]
                diameter_change_for_selection_art = t_new_array[current_selection_arterioles]
                diameter_change_for_selection_ven = t_new_array[current_selection_venules]

                diameter_change_for_selection_percent = np.abs(diameter_change_for_selection - 1) * 100
                diameter_change_for_selection_percent_cap = np.abs(diameter_change_for_selection_cap - 1) * 100
                diameter_change_for_selection_percent_art = np.abs(diameter_change_for_selection_art - 1) * 100
                diameter_change_for_selection_percent_ven = np.abs(diameter_change_for_selection_ven - 1) * 100

                diameter_change_all.append(np.mean(diameter_change_for_selection_percent))
                diameter_change_capillaries.append(np.mean(diameter_change_for_selection_percent_cap))
                diameter_change_veins.append(np.mean(diameter_change_for_selection_percent_ven))
                diameter_change_arterioles.append(np.mean(diameter_change_for_selection_percent_art))

            if case_id == 'R_100':

                if network == '0_Out_':

                    data_r_100_n0.append(diameter_change_all)

                elif network == '3_Out_':

                    data_r_100_n3.append(diameter_change_all)

                elif network == '4_Out_':

                    data_r_100_n4.append(diameter_change_all)

                elif network == '6_Out_':

                    data_r_100_n6.append(diameter_change_all)

            elif case_id == 'R_200':

                if network == '0_Out_':

                    data_r_200_n0.append(diameter_change_all)

                elif network == '3_Out_':

                    data_r_200_n3.append(diameter_change_all)

                elif network == '4_Out_':

                    data_r_200_n4.append(diameter_change_all)

                elif network == '6_Out_':

                    data_r_200_n6.append(diameter_change_all)

            elif case_id == 'R_300':

                if network == '0_Out_':

                    data_r_300_n0.append(diameter_change_all)

                elif network == '3_Out_':

                    data_r_300_n3.append(diameter_change_all)

                elif network == '4_Out_':

                    data_r_300_n4.append(diameter_change_all)

                elif network == '6_Out_':

                    data_r_300_n6.append(diameter_change_all)

            elif case_id == 'All':

                if network == '0_Out_':

                    data_r_all_n0.append(diameter_change_all)

                elif network == '3_Out_':

                    data_r_all_n3.append(diameter_change_all)

                elif network == '4_Out_':

                    data_r_all_n4.append(diameter_change_all)

                elif network == '6_Out_':

                    data_r_all_n6.append(diameter_change_all)

            # print(diameter_change_all)
            # print(diameter_change_arterioles)
            # print(diameter_change_veins)
            # print(diameter_change_capillaries)

    fig, axs = plt.subplots(2, 2, figsize=(7, 7))

    axs[0, 0].plot(radii, data_r_100_n0[0], label='Network 1', color='blue')
    axs[0, 0].plot(radii, data_r_100_n3[0], label='Network 2', color='orange')
    axs[0, 0].plot(radii, data_r_100_n4[0], label='Network 3', color='green')
    axs[0, 0].plot(radii, data_r_100_n6[0], label='Network 4', color='red')

    axs[0, 0].set_title('R 100')
    axs[0, 0].yaxis.grid(True)
    axs[0, 0].legend(prop={'size': 6})

    axs[0, 1].plot(radii, data_r_200_n0[0], label='Network 1', color='blue')
    axs[0, 1].plot(radii, data_r_200_n3[0], label='Network 2', color='orange')
    axs[0, 1].plot(radii, data_r_200_n4[0], label='Network 3', color='green')
    axs[0, 1].plot(radii, data_r_200_n6[0], label='Network 4', color='red')

    axs[0, 1].set_title('R 200')
    axs[0, 1].yaxis.grid(True)
    axs[0, 1].legend(prop={'size': 6})

    axs[1, 0].plot(radii, data_r_300_n0[0], label='Network 1', color='blue')
    axs[1, 0].plot(radii, data_r_300_n3[0], label='Network 2', color='orange')
    axs[1, 0].plot(radii, data_r_300_n4[0], label='Network 3', color='green')
    axs[1, 0].plot(radii, data_r_300_n6[0], label='Network 4', color='red')

    axs[1, 0].set_title('R 300')
    axs[1, 0].yaxis.grid(True)
    axs[1, 0].legend(prop={'size': 6})

    axs[1, 1].plot(radii, data_r_all_n0[0], label='Network 1', color='blue')
    axs[1, 1].plot(radii, data_r_all_n3[0], label='Network 2', color='orange')
    axs[1, 1].plot(radii, data_r_all_n4[0], label='Network 3', color='green')
    axs[1, 1].plot(radii, data_r_all_n6[0], label='Network 4', color='red')

    axs[1, 1].set_title('All')
    axs[1, 1].yaxis.grid(True)
    axs[1, 1].legend(prop={'size': 6})

    for ax in axs.flat:
        ax.set(ylabel='Change of Diameter in %', xlabel='Distance from activation center [$\mu m$]')

    # Hide x labels and tick labels for top plots and y ticks for right plots.
    for ax in axs.flat:
        ax.label_outer()

    plt.savefig(r'D:\00 Privat\01_Bildung\01_ETH Zürich\MSc\00_Masterarbeit\02_Network_Study_Small\Adjoint\Direct_Comparison\R_100_200_300_All.png')
    plt.clf()

    return None


########################################################################################################################
# START
########################################################################################################################

prozent_grenze_diameter = 1
prozent_grenze_flow = 1

# # FLOW (R100 to ALL)
# plot_flow_all_vessel_types_NEW(prozent_grenze_flow)
# plot_flow_all_vessel_types_NEW_ACTIVATED_REGION(prozent_grenze_flow)
# plot_flow_all_vessel_types_NEW_capillaries(prozent_grenze_flow)
plot_flow_all_vessel_types_NEW_arterioles_and_venules(prozent_grenze_flow)
# plot_flow_all_vessel_types_ACTIVATED_REGION_capillaries(prozent_grenze_flow)
# plot_flow_all_vessel_types_NEW_ACTIVATED_REGION_arterioles_and_venules(prozent_grenze_flow)


# # DIAMETER (R100 to ALL)
# plot_diameter_all_vessel_types_NEW(prozent_grenze_diameter)
# plot_diameter_capillaries_types_NEW(prozent_grenze_diameter)
# plot_diameter_venules_types_NEW(prozent_grenze_diameter)
# plot_diameter_arterioles_and_venules_types_NEW(prozent_grenze_diameter)
# plot_diameter_arterioles_types_NEW(prozent_grenze_diameter)


# # Flow (All Cap, All Art)
# plot_flow_arteries_direct_comp(prozent_grenze_flow)
# plot_flow_caps_direct_comp(prozent_grenze_flow)


# # Diameter (All Cap, All Art)
# plot_diameter_arteries_direct_comp(prozent_grenze_diameter)
# plot_diameter_caps_direct_comp(prozent_grenze_diameter)


# # R_100_200_300_All (2x2) with 4 Networks - Plot
# plot_distance()


# DIAMETER BOXPLOTS - Direct Comparison (All Cap, All Arteries)
plot_diameter_all_caps_all_art_all_Comparison(1)

