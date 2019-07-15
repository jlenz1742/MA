import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# Chose the mode
# 0: different gamma -> folder has to be called gamma_x_x
# 1: different folder names -> specify folder name in the corresponding list

mode = 1

# Chose different gammas

gamma = ['0_01', '0_1', '1_0', '10_0']

# Chose different folders

folders = ['0', '1', '2', '3']

# Chose target folder (Data folders have to be in this folder -> Output is stored in this specified folder)

target_folder = r'D:\00 Privat\01_Bildung\01_ETH Zürich\MSc\00_Masterarbeit' \
                r'\Data_Simulation\different_selection_reacting_eids'

####################################################################################################################

size_figure = (9, 9)
fig = plt.figure(figsize=size_figure)
plt.suptitle("")

####################################################################################################################
#                                                                                                                  #
#                                                   Rho Plot                                                       #
#                                                                                                                  #
####################################################################################################################

plt.subplot(2, 1, 1)

if mode == 0:

    for g in gamma:

        path_adj_data = target_folder + '\\' + g + r'\out\adjointdata.csv'

        df = pd.read_csv(path_adj_data)

        it = np.asarray(df.it)
        rho = np.asarray(df.rho)

        plt.semilogy(it, rho, label=g)

elif mode == 1:

    for f in folders:

        path_adj_data = target_folder + '\\' + f + r'\out\adjointdata.csv'

        df = pd.read_csv(path_adj_data)

        it = np.asarray(df.it)
        rho = np.asarray(df.rho)

        plt.semilogy(it, rho, label=f)

plt.title("Rho", fontsize=10)
plt.grid(True)
plt.legend()

####################################################################################################################
#                                                                                                                  #
#                                               Regularization Rho n                                               #
#                                                                                                                  #
####################################################################################################################

plt.subplot(2, 1, 2)

if mode == 0:

    for g in gamma:

        path_adj_data = target_folder + '\\' + g + r'\out\adjointdata.csv'

        df = pd.read_csv(path_adj_data)

        it = np.asarray(df.it)
        regularization_rho_n = np.asarray(df.regularization_rho_n)

        plt.semilogy(it, regularization_rho_n, label=g)

elif mode == 1:

    for f in folders:

        path_adj_data = target_folder + '\\' + f + r'\out\adjointdata.csv'

        df = pd.read_csv(path_adj_data)

        it = np.asarray(df.it)
        regularization_rho_n = np.asarray(df.regularization_rho_n)

        plt.semilogy(it, regularization_rho_n, label=f)

plt.title("Regularization Rho n", fontsize=10)
plt.grid(True)
plt.legend()

plt.savefig(target_folder + r'\Regularization_plot_comparison.png')
