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

    size_figure = (15, 15)
    fig = plt.figure(figsize=size_figure)
    plt.suptitle("")

    ####################################################################################################################
    #                                                                                                                  #
    #                                                   Rho Plot                                                       #
    #                                                                                                                  #
    ####################################################################################################################

    time_str = time.strftime("_%Y%m%d_%H%M%S")

    label_strings = []

    for i in range(len(gammas)):

        label_strings.append('ID: ' + folders[i] + ', ' + 'Gamma: ' + str(gammas[i]))

    plt.subplot(2, 1, 1)

    for i in range(len(folders)):

        path_adj_data = main_folder + '\\' + folders[i] + r'\out\adjointdata.csv'
        df = pd.read_csv(path_adj_data)
        it = np.asarray(df.it)
        rho = np.asarray(df.rho)
        plt.semilogy(it, rho, label=label_strings[i])

    plt.title("Rho", fontsize=10)
    plt.grid(True)
    plt.legend()

    ####################################################################################################################
    #                                                                                                                  #
    #                                               Regularization Rho n                                               #
    #                                                                                                                  #
    ####################################################################################################################

    plt.subplot(2, 1, 2)

    for i in range(len(folders)):

        path_adj_data = main_folder + '\\' + folders[i] + r'\out\adjointdata.csv'
        df = pd.read_csv(path_adj_data)
        it = np.asarray(df.it)
        regularization_rho_n = np.asarray(df.regularization_rho_n)
        plt.semilogy(it, regularization_rho_n, label=label_strings[i])

    plt.title("Regularization Rho n", fontsize=10)
    plt.grid(True)
    plt.legend()

    name_png = 'Regularization_plot_different_gammas' + time_str + '.png'

    plt.savefig(main_folder + '\\' + name_png)


def different_radii(main_folder, folders, radii):

    size_figure = (15, 15)
    fig = plt.figure(figsize=size_figure)
    plt.suptitle("")

    ####################################################################################################################
    #                                                                                                                  #
    #                                                   Rho Plot                                                       #
    #                                                                                                                  #
    ####################################################################################################################

    time_str = time.strftime("_%Y%m%d_%H%M%S")

    label_strings = []

    for i in range(len(radii)):

        label_strings.append('ID: ' + folders[i] + ', ' + 'Radius: ' + str(radii[i]))

    plt.subplot(2, 1, 1)

    for i in range(len(folders)):

        path_adj_data = main_folder + '\\' + folders[i] + r'\out\adjointdata.csv'
        df = pd.read_csv(path_adj_data)
        it = np.asarray(df.it)
        rho = np.asarray(df.rho)
        plt.semilogy(it, rho, label=label_strings[i])

    plt.title("Rho", fontsize=10)
    plt.grid(True)
    plt.legend()

    ####################################################################################################################
    #                                                                                                                  #
    #                                               Regularization Rho n                                               #
    #                                                                                                                  #
    ####################################################################################################################

    plt.subplot(2, 1, 2)

    for i in range(len(folders)):

        path_adj_data = main_folder + '\\' + folders[i] + r'\out\adjointdata.csv'
        df = pd.read_csv(path_adj_data)
        it = np.asarray(df.it)
        regularization_rho_n = np.asarray(df.regularization_rho_n)
        plt.semilogy(it, regularization_rho_n, label=label_strings[i])

    plt.title("Regularization Rho n", fontsize=10)
    plt.grid(True)
    plt.legend()

    name_png = 'Regularization_plot_different_radii' + time_str + '.png'
    plt.savefig(main_folder + '\\' + name_png)


def different_epsilon(main_folder, folders, epsilon):

    size_figure = (15, 15)
    fig = plt.figure(figsize=size_figure)
    plt.suptitle("")

    ####################################################################################################################
    #                                                                                                                  #
    #                                                   Rho Plot                                                       #
    #                                                                                                                  #
    ####################################################################################################################

    time_str = time.strftime("_%Y%m%d_%H%M%S")

    label_strings = []

    for i in range(len(epsilon)):

        label_strings.append('ID: ' + folders[i] + ', ' + 'Epsilon: ' + str(epsilon[i]))

    plt.subplot(2, 1, 1)

    for i in range(len(folders)):

        path_adj_data = main_folder + '\\' + folders[i] + r'\out\adjointdata.csv'
        df = pd.read_csv(path_adj_data)
        it = np.asarray(df.it)
        rho = np.asarray(df.rho)
        plt.semilogy(it, rho, label=label_strings[i])

    plt.title("Rho", fontsize=10)
    plt.grid(True)
    plt.legend()

    ####################################################################################################################
    #                                                                                                                  #
    #                                               Regularization Rho n                                               #
    #                                                                                                                  #
    ####################################################################################################################

    plt.subplot(2, 1, 2)

    for i in range(len(folders)):

        path_adj_data = main_folder + '\\' + folders[i] + r'\out\adjointdata.csv'
        df = pd.read_csv(path_adj_data)
        it = np.asarray(df.it)
        regularization_rho_n = np.asarray(df.regularization_rho_n)
        plt.semilogy(it, regularization_rho_n, label=label_strings[i])

    plt.title("Regularization Rho n", fontsize=10)
    plt.grid(True)
    plt.legend()

    name_png = 'Regularization_plot_different_epsilon' + time_str + '.png'
    plt.savefig(main_folder + '\\' + name_png)
