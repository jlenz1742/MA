import matplotlib.pyplot as plt
import numpy as np
from matplotlib import colors
from matplotlib.ticker import PercentFormatter
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
import random


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

    a = np.where(np.abs(f_start_new_array) > q_sim_zeitpunkt_9999 / 10)
    relevant_edges = list(a[0])

    return relevant_edges


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

            # print(diameter_changes_selection_FINAL)
            data[network][radius] = flow_changes_selection_FINAL

    print(np.mean(data['0_Out_']['R_100']))
    print(data['0_Out_']['R_100'])
    print(max(data['0_Out_']['R_100']))

    a = np.where(data['0_Out_']['R_100'] > 100)
    print(data['0_Out_']['R_100'][a])

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
    testPlot.set_ylim(-7, 40)

    ind = 0
    for tick in range(len(testPlot.get_xticklabels())):

        print(tick)
        testPlot.text(tick - .1, -2, str(np.round(number_affecte_vessels[tick][1] / data['3_Out_Tot']['Number_Vessels'] * 100, 2)) + '\%', horizontalalignment='center', color='orange', rotation=70, size=10)
        testPlot.text(tick + .1, -2, str(np.round(number_affecte_vessels[tick][2] / data['4_Out_Tot']['Number_Vessels'] * 100, 2)) + '\%', horizontalalignment='center', color='green', rotation=70, size=10)
        testPlot.text(tick + .3, -2, str(np.round(number_affecte_vessels[tick][3] / data['6_Out_Tot']['Number_Vessels'] * 100, 2)) + '\%', horizontalalignment='center', color='red', rotation=70, size=10)
        testPlot.text(tick - .3, -2, str(np.round(number_affecte_vessels[tick][0] / data['0_Out_Tot']['Number_Vessels'] * 100, 2)) + '\%', horizontalalignment='center', color='blue', rotation=70, size=10)
        ind += 2

    # plt.legend(loc='upper right')
    # plt.savefig(
    #      r'D:\00 Privat\01_Bildung\01_ETH Zürich\MSc\00_Masterarbeit\02_Network_Study_Small\Adjoint\Flow_Change_R100_to_ALL\Rest\flow_diameter_capillaries.png')
    plt.show()



plot_flow_all_vessel_types_NEW_capillaries(1)