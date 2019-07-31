
import matplotlib.pyplot as plt
import time
import numpy as np
import pandas as pd
import math

_path_adjoint_file = r'D:\00 Privat\01_Bildung\01_ETH Zürich\MSc\00_Masterarbeit\00_Simulations\testcases_janLenz' \
                     r'\jl_001_a\out\adjointdata.csv'

_start_file = r'D:\00 Privat\01_Bildung\01_ETH Zürich\MSc\00_Masterarbeit\00_Simulations\testcases_janLenz' \
              r'\jl_001_a\out\meshdata_249.csv'

_end_file_ = r'D:\00 Privat\01_Bildung\01_ETH Zürich\MSc\00_Masterarbeit\00_Simulations\testcases_janLenz' \
             r'\jl_001_a\out\meshdata_11999.csv'

activated_eids = [1973, 1983, 1984, 1991, 2135, 2220, 2221, 2229,2230,2238,2239,2247,2248,2249,2250,2258,2266,2267,2410,2413,
                  2496,2504,2505,2513,2514,2522,2523,2524,2525,2533,2534,2542,2543,2544,2552,2553,2695,2698,2776,2777,
                  2785,2786,2787,2795,2796,2804,2805,2806,2814,2815,2822,2830,2831,2965,2968,2973,3051,3058,3065,3066,
                  3074,3075,3076,3084,3091,3092,3100,3101,3235,3240,3243,3331,3332,3339,3340,3348,3349,3358,3359,3365,
                  3366,3512,3517,3783,8570,8573,8593,8594,8596,8597,8606,8607,8611,8612,8617,8618,8656,8657,8961,8962,
                  8976,8988,8989,9126,9127,9139,9140,9142,9143,9147,9148,9150,9151,9455,9515,9516,9532,9533,9633,9634,
                  9963,9964,9975,9976,9980,9989,9990,9997,10002,10003]


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

    d_ratio = d_start_new_array/d_end_new_array

    # FLOW

    f_start = np.array(df_start['Fplasma'])
    f_final = np.array(df_final['Fplasma'])

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

    l_start = np.array(df_start['D'])

    l_start_new = []

    for i in range(len(l_start)):

        if (i % 2) == 0:
            l_start_new.append(l_start[i])

    l_start_new_array = np.asarray(d_start_new)

    l_activated = l_start_new_array[activated_ids]
    l_total = np.sum(l_activated)

    # Gewichtete Flussänderungsberechnung

    x = f_activated * l_activated / l_total

    # Ein Element hat einen fluss von -71 -> wird mit diesen beiden zeilen rausgelöscht

    flow_change_final = np.sum(x)

    q_0 = l_activated * np.abs(f_start_new_array[activated_ids])
    q_0_sum = np.sum(q_0)

    dir_ = np.ones(len(activated_ids))
    dir_tool = np.where(f_start_new_array[activated_ids] < 0)

    dir_[dir_tool] = -1

    q_1 = l_activated * dir_ * f_end_new_array[activated_ids]
    q_1_sum = np.sum(q_1)

    print(q_1_sum/q_0_sum)

compute_flow_change(_start_file, _end_file_, activated_eids)










































