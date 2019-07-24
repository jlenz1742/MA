import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time


def execute(codes, main_folder, folders, gammas, radii, epsilon):

    for code in codes:

        if code == 0:

            if len(gammas) == 0:

                continue

            else:

                different_gammas(main_folder, folders, gammas)

        elif code == 1:

            if len(radii) == 0:

                continue

            else:

                different_radii(main_folder, folders, radii)

        elif code == 2:

            if len(epsilon) == 0:

                continue

            else:

                different_epsilon(main_folder, folders, epsilon)

        else:

            return


def different_gammas(main_folder, folders, gammas):

    time_str = time.strftime("_%Y%m%d_%H%M%S")

    label_strings = []

    for i in range(len(gammas)):

        label_strings.append('ID: ' + folders[i] + ', ' + 'Gamma: ' + str(gammas[i]))

    name_group_folder = main_folder

    size_figure = (9, 9)
    fig = plt.figure(figsize=size_figure)
    plt.suptitle("")

    ax1 = fig.add_subplot(111)

    for i in range(len(folders)):

        path_adj_data = name_group_folder + '\\' + folders[i] + r'\out\adjointdata.csv'

        df = pd.read_csv(path_adj_data)

        it = np.asarray(df.it)
        total_cost = np.asarray(df.totalCost)

        plt.semilogy(it, total_cost, label=label_strings[i])

    plt.title("Cost Function", fontsize=10)
    ax1.yaxis.major.formatter._useMathText = True
    plt.grid(True)
    plt.legend()

    name_png = 'Cost_function_different_gammas' + time_str + '.png'

    plt.savefig(name_group_folder + '\\' + name_png)


def different_radii(main_folder, folders, radii):

    time_str = time.strftime("_%Y%m%d_%H%M%S")

    label_strings = []

    for i in range(len(radii)):

        label_strings.append('ID: ' + folders[i] + ', ' + 'Radius: ' + str(radii[i]) + ' $\mu$m')

    name_group_folder = main_folder

    size_figure = (9, 9)
    fig = plt.figure(figsize=size_figure)
    plt.suptitle("")

    ax1 = fig.add_subplot(111)

    for i in range(len(folders)):

        path_adj_data = name_group_folder + '\\' + folders[i] + r'\out\adjointdata.csv'

        df = pd.read_csv(path_adj_data)

        it = np.asarray(df.it)
        total_cost = np.asarray(df.totalCost)

        plt.semilogy(it, total_cost, label=label_strings[i])

    plt.title("Cost Function", fontsize=10)
    ax1.yaxis.major.formatter._useMathText = True
    plt.grid(True)
    plt.legend()

    name_png = 'Cost_function_different_radii' + time_str + '.png'

    plt.savefig(name_group_folder + '\\' + name_png)


def different_epsilon(main_folder, folders, epsilon):

    time_str = time.strftime("_%Y%m%d_%H%M%S")

    label_strings = []

    for i in range(len(epsilon)):

        label_strings.append('ID: ' + folders[i] + ', ' + 'Epsilon: ' + str(epsilon[i]))

    name_group_folder = main_folder

    size_figure = (9, 9)
    fig = plt.figure(figsize=size_figure)
    plt.suptitle("")

    ax1 = fig.add_subplot(111)

    for i in range(len(folders)):

        path_adj_data = name_group_folder + '\\' + folders[i] + r'\out\adjointdata.csv'

        df = pd.read_csv(path_adj_data)

        it = np.asarray(df.it)
        total_cost = np.asarray(df.totalCost)

        plt.semilogy(it, total_cost, label=label_strings[i])

    plt.title("Cost Function", fontsize=10)
    ax1.yaxis.major.formatter._useMathText = True
    plt.grid(True)
    plt.legend()

    name_png = 'Cost_function_different_epsilon' + time_str + '.png'

    plt.savefig(name_group_folder + '\\' + name_png)
