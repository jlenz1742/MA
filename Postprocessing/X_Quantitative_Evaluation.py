import matplotlib.pyplot as plt
import matplotlib as mpl
import time
import numpy as np
import pandas as pd
import math
from mpl_toolkits.mplot3d.art3d import Line3DCollection
import os
import glob

_path_adjoint_file = r'D:\00 Privat\01_Bildung\01_ETH Zürich\MSc\00_Masterarbeit\00_Simulations\adjointdata.csv'

_start_file = r'D:\00 Privat\01_Bildung\01_ETH Zürich\MSc\00_Masterarbeit\00_Simulations\Cutted\out\meshdata_249.csv'

_end_file_ = r'D:\00 Privat\01_Bildung\01_ETH Zürich\MSc\00_Masterarbeit\00_Simulations\Cutted\out\meshdata_2999.csv'

activated_eids = [1657, 1662,1669,1670,1672,1734,1803,1809,1814,1820,1821,1822,1828,1882,1885,1887,1943,1950,1955,1956,1957,1962,1963,1969,1970,1977,2036,2099,2105,2110,2111,2112,2117,2118,2122,2123,2124,2185,2259,2266,2271,2278,2340,5828,5829,6802,6803,7057,7290,7291,7316,7745,7751,7762,7763,7765,7766,7768,7769,7770,7771,7789,7818,7819,8478,8479,8676,8677]


def adjoint_data_plot(path):

    size_figure = (15, 15)
    fig = plt.figure(figsize=size_figure)
    plt.suptitle("")

    ####################################################################################################################
    #                                                                                                                  #
    #                                                   Rho Plot                                                       #
    #                                                                                                                  #
    ####################################################################################################################

    plt.subplot(2, 1, 1)

    path_adj_data = path
    df = pd.read_csv(path_adj_data)
    it = np.asarray(df.it)
    cost = np.asarray(df.totalCost)
    plt.semilogy(it, cost)

    plt.title("Cost Function", fontsize=10)
    plt.grid(True)
    plt.legend()

    ####################################################################################################################
    #                                                                                                                  #
    #                                               Regularization Rho n                                               #
    #                                                                                                                  #
    ####################################################################################################################

    plt.subplot(2, 1, 2)

    path_adj_data = path
    df = pd.read_csv(path_adj_data)
    it = np.asarray(df.it)
    regularization_rho_n = np.asarray(df.regularization_rho_n)
    rho = np.asarray(df.rho)
    plt.plot(it, rho, label='Rho', linestyle='dashed')
    plt.plot(it, regularization_rho_n, label='Rho_n')

    plt.title("Regularization Rho n", fontsize=10)
    plt.grid(True)
    plt.legend()
    plt.ylim(0, 0.4)

    plt.show()


def compute_flow_change(start_file, end_file, activated_ids):

    df_start = pd.read_csv(start_file)
    df_final = pd.read_csv(end_file)

    # DIAMETER

    d_start = np.array(df_start['D'])
    d_final = np.array(df_final['D'])

    d_start_new = []
    d_end_new = []

    for i in range(len(df_start)):

        if (i % 2) == 0:
            d_start_new.append(d_start[i])
            d_end_new.append((d_final[i]))

    d_start_new_array = np.asarray(d_start_new)
    d_end_new_array = np.asarray(d_end_new)

    # FLOW

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

    f_ratio = f_start_new_array/f_end_new_array

    f_activated = f_ratio[activated_ids]

    # LENGTH

    l_start = np.array(df_start['L'])
    l_start_new = []

    for i in range(len(l_start)):

        if (i % 2) == 0:
            l_start_new.append(l_start[i])

    l_start_new_array = np.asarray(l_start_new)

    l_activated = l_start_new_array[activated_ids]

    # Gewichtete Flussänderungsberechnung

    q_0 = l_activated * np.abs(f_start_new_array[activated_ids])
    q_0_sum = np.sum(q_0)

    dir_ = np.ones(len(activated_ids))
    dir_tool = np.where(f_start_new_array[activated_ids] < 0)

    dir_[dir_tool] = -1

    q_1 = l_activated * dir_ * f_end_new_array[activated_ids]
    q_1_sum = np.sum(q_1)

    print(q_1_sum/q_0_sum)


def compute_diameter_change(start_file, end_file, activated_ids):
    df_start = pd.read_csv(start_file)
    df_final = pd.read_csv(end_file)

    # DIAMETER

    d_start = np.array(df_start['D'])
    d_final = np.array(df_final['D'])

    d_start_new = []
    d_end_new = []

    for i in range(len(df_start)):

        if (i % 2) == 0:
            d_start_new.append(d_start[i])
            d_end_new.append((d_final[i]))

    d_start_new_array = np.asarray(d_start_new)
    d_end_new_array = np.asarray(d_end_new)

    d_ratio = d_start_new_array / d_end_new_array

    d_ratio_activated = d_ratio[activated_ids]

    print(d_ratio_activated)
    mean_alpha = np.mean(d_ratio_activated)
    std_alpha = np.std(d_ratio_activated)

    d_ratio_non_activated = np.delete(d_ratio, activated_ids)
    print()


def plot_3_d_diameter(path_target, meshdata_start, meshdata_final):

    mesh_data_file_start = meshdata_start
    mesh_data_file_final = meshdata_final

    df_start = pd.read_csv(path_target + "\\" + mesh_data_file_start)
    df_final = pd.read_csv(path_target + "\\" + mesh_data_file_final)

    x = df_start['x']
    x_min = min(x)
    x_max = max(x)

    y = df_start['y']
    y_min = min(y)
    y_max = max(y)

    z = df_start['z']
    z_min = min(z)
    z_max = max(z)

    f_plasma_1 = np.array(df_start['Fplasma'])
    f_plasma_2 = np.array(df_final['Fplasma'])

    f_plasma_ratio = f_plasma_2 / f_plasma_1

    test = []

    for i in range(len(f_plasma_ratio)):

        if f_plasma_ratio[i] > 2:

            test.append(1)

        elif f_plasma_ratio[i] < 0:

            test.append(1)

        else:

            test.append(f_plasma_ratio[i])

    f_plasma_ratio_processed = np.asarray(test)

    print('Max and Min Plasma Flow Processed: ', max(f_plasma_ratio_processed), min(f_plasma_ratio_processed))

    t_1 = np.array(df_start['D'])
    t_2 = np.array(df_final['D'])

    t = t_2 / t_1
    t_new = []
    t_2_new = []

    for i in range(len(t)):

        if (i % 2) == 0:
            t_2_new.append(t_2[i] / (4 / math.pow(10, 6)))
            t_new.append(t[i])

    t_new_array = np.asarray(t_new)
    reacting_ids = np.where(t_new_array != 1)

    print(f_plasma_ratio_processed[reacting_ids])

    # generate a list of (x,y,z) points
    points = np.array([x, y, z]).transpose().reshape(-1, 1, 3)

    # set up a list of segments
    segs_temp = np.concatenate([points[:-1], points[1:]], axis=1)
    segs = []

    for i in range(len(segs_temp)):

        if (i % 2) == 0:
            segs.append(segs_temp[i])

    # make the collection of segments

    max_diameter_change = max(t_new_array)-1
    min_diameter_change = np.absolute(np.absolute(min(t_new_array)) - 1)

    a = np.where(1.02 > t_new_array)
    b = np.where(0.98 < t_new_array)
    c = np.intersect1d(a, b)

    transparent_array = np.copy(np.asarray(segs)[c, ])
    transparent_diameters_ratio = np.copy(t_new_array[c, ])
    transparent_diameters = np.copy(np.asarray(t_2_new)[c, ])
    non_transparent_segs = []
    non_transparent_diameters_ratio = []
    non_transparent_diameters = []

    indices_non_transparent = list(range(len(t_new_array)))

    for i in c:

        indices_non_transparent.remove(i)

    for i in indices_non_transparent:

        non_transparent_segs.append(segs[i])
        non_transparent_diameters_ratio.append((t_new_array[i]))
        non_transparent_diameters.append(t_2_new[i])

    if max_diameter_change > min_diameter_change:

        min_change = 1-max_diameter_change

        lc = Line3DCollection(transparent_array, linewidths=transparent_diameters, alpha=0.3, cmap=plt.get_cmap('coolwarm', 5),
                              norm=mpl.colors.Normalize(vmin=min_change, vmax=max(t_new_array)))

        lb = Line3DCollection(non_transparent_segs, linewidths=non_transparent_diameters, alpha=1, cmap=plt.get_cmap('coolwarm', 5),
                              norm=mpl.colors.Normalize(vmin=min_change, vmax=max(t_new_array)))
    else:

        max_change = 1 + min_diameter_change

        lc = Line3DCollection(transparent_array, linewidths=transparent_diameters, alpha=0.3, cmap=plt.get_cmap('coolwarm'),
                              norm=mpl.colors.Normalize(vmin=min(t_new_array), vmax=max_change))

        lb = Line3DCollection(non_transparent_segs, linewidths=non_transparent_diameters, alpha=1, cmap=plt.get_cmap('coolwarm', 5),
                              norm=mpl.colors.Normalize(vmin=min(t_new_array), vmax=max_change))

    lb.set_array(np.asarray(non_transparent_diameters_ratio))
    lc.set_array(transparent_diameters_ratio)  # color the segments by our parameter

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    ax.add_collection3d(lb)
    ax.add_collection3d(lc)

    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    ax.set_zlim(z_min, z_max)

    plt.colorbar(lb)
    plt.show()


def plot_3_d_plasma(path_target, meshdata_start, meshdata_final):

    df_start = pd.read_csv(path_target + "\\" + meshdata_start)
    df_final = pd.read_csv(path_target + "\\" + meshdata_final)

    # DIAMETER

    d_start = np.array(df_start['D'])
    d_final = np.array(df_final['D'])

    d_start_new = []
    d_end_new = []

    for i in range(len(df_start)):

        if (i % 2) == 0:
            d_start_new.append(d_start[i])
            d_end_new.append((d_final[i]))

    d_start_new_array = np.asarray(d_start_new)
    d_end_new_array = np.asarray(d_end_new)

    # FLOW

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

    f_ratio = f_start_new_array/f_end_new_array

    x = df_start['x']
    x_min = min(x)
    x_max = max(x)

    y = df_start['y']
    y_min = min(y)
    y_max = max(y)

    z = df_start['z']
    z_min = min(z)
    z_max = max(z)

    t_1 = np.array(df_start['D'])
    t_2 = np.array(df_final['D'])

    t = t_2 / t_1
    t_new = []
    t_2_new = []

    for i in range(len(t)):

        if (i % 2) == 0:
            t_2_new.append(t_2[i] / (4 / math.pow(10, 6)))
            t_new.append(t[i])

    t_new_array = np.asarray(t_new)

    # generate a list of (x,y,z) points
    points = np.array([x, y, z]).transpose().reshape(-1, 1, 3)

    # set up a list of segments
    segs_temp = np.concatenate([points[:-1], points[1:]], axis=1)
    segs = []

    for i in range(len(segs_temp)):

        if (i % 2) == 0:
            segs.append(segs_temp[i])

    # make the collection of segments

    max_f_ratio_change = max(f_ratio)-1
    min_f_ratio_change = np.absolute(np.absolute(min(f_ratio)) - 1)

    a = np.where(1.02 > t_new_array)
    b = np.where(0.98 < t_new_array)
    c = np.intersect1d(a, b)

    transparent_array = np.copy(np.asarray(segs)[c, ])
    transparent_diameters_ratio = np.copy(t_new_array[c, ])
    transparent_flow_ratio = np.copy(f_ratio[c, ])
    transparent_diameters = np.copy(np.asarray(t_2_new)[c, ])
    non_transparent_segs = []
    non_transparent_diameters_ratio = []
    non_transparent_diameters = []
    non_transpartent_flow_ratio = []

    indices_non_transparent = list(range(len(t_new_array)))

    for i in c:

        indices_non_transparent.remove(i)

    for i in indices_non_transparent:

        non_transparent_segs.append(segs[i])
        non_transparent_diameters_ratio.append((t_new_array[i]))
        non_transparent_diameters.append(t_2_new[i])
        non_transpartent_flow_ratio.append(f_ratio[i])

    if max_f_ratio_change > min_f_ratio_change:

        min_change = 1-max_f_ratio_change

        lc = Line3DCollection(transparent_array, linewidths=transparent_diameters, alpha=0.1, cmap=plt.get_cmap('coolwarm', 5),
                              norm=mpl.colors.Normalize(vmin=0.5, vmax=1.5))

        lb = Line3DCollection(non_transparent_segs, linewidths=non_transparent_diameters, alpha=1, cmap=plt.get_cmap('coolwarm', 5),
                              norm=mpl.colors.Normalize(vmin=0.5, vmax=1.5))
    else:

        max_change = 1 + min_f_ratio_change

        lc = Line3DCollection(transparent_array, linewidths=transparent_diameters, alpha=0.1, cmap=plt.get_cmap('coolwarm'),
                              norm=mpl.colors.Normalize(vmin=0.5, vmax=1.5))

        lb = Line3DCollection(non_transparent_segs, linewidths=non_transparent_diameters, alpha=1, cmap=plt.get_cmap('coolwarm', 5),
                              norm=mpl.colors.Normalize(vmin=0.5, vmax=1.5))

    lb.set_array(np.asarray(non_transpartent_flow_ratio))
    lc.set_array(transparent_flow_ratio)  # color the segments by our parameter

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    ax.add_collection3d(lb)
    ax.add_collection3d(lc)

    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    ax.set_zlim(z_min, z_max)

    plt.colorbar(lb)
    plt.show()


def plot_diameter_change(path_target, meshdata_start, meshdata_final):

    mesh_data_file_start = meshdata_start
    mesh_data_file_final = meshdata_final

    df_start = pd.read_csv(path_target + "\\" + mesh_data_file_start)
    df_final = pd.read_csv(path_target + "\\" + mesh_data_file_final)

    t_1 = np.array(df_start['D'])
    t_2 = np.array(df_final['D'])

    t = t_2 / t_1
    t_new = []
    t_2_new = []

    for i in range(len(t)):

        if (i % 2) == 0:
            t_2_new.append(t_2[i] / (4 / math.pow(10, 6)))
            t_new.append(t[i])

    t_new_array = np.asarray(t_new)

    diameter_change = np.asarray(sorted(np.absolute((t_new_array-1)*100)))
    index = np.where(np.asarray(diameter_change) > 0.1)
    index_list = list(index[0])
    print(index_list)
    diameter_change_new = diameter_change[index_list]

    print(diameter_change)
    n, bins, patches = plt.hist(diameter_change_new, bins='auto', density=False, facecolor='g', alpha=0.75)

    plt.xlabel('Diameter Change in %')
    plt.ylabel('Number of Vessels')
    plt.xlim(0, 10)
    plt.grid(True)
    plt.show()

a = r'D:\00 Privat\01_Bildung\01_ETH Zürich\MSc\00_Masterarbeit\00_Simulations\i1\out'
meshdata_1 = 'meshdata_249.csv'
meshdata_2 = 'meshdata_12999.csv'

# plot_3_d_diameter(a, meshdata_1, meshdata_2)
# adjoint_data_plot(_path_adjoint_file)
# compute_flow_change(_start_file, _end_file_, activated_eids)
# plot_3_d_plasma(a, meshdata_1, meshdata_2)
# plot_diameter_change(a, meshdata_1, meshdata_2)








































