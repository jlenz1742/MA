import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Specify Group Folder

name_group_folder = r'D:\00 Privat\01_Bildung\01_ETH Zürich\MSc\00_Masterarbeit\00_Simulations\1_Results'

# Specify single folders

single_folders = {'1.0': 'Radius = 100', '1.1': 'Radius = 120', '1.3': 'Radius = 160', '1.4': 'Radius = 200'}

# Name output file

name_aid = 'different_radius_gamma_10_rho_0.7'

########################################################################################################################
#                                                                                                                      #
#                                                      Plot                                                            #
#                                                                                                                      #
#######################################################################################################################

size_figure = (9, 9)
fig = plt.figure(figsize=size_figure)
plt.suptitle("")

####################################################################################################################
#                                                                                                                  #
#                                                   Rho Plot                                                       #
#                                                                                                                  #
####################################################################################################################

plt.subplot(2, 1, 1)

for folder in single_folders:


    path_adj_data = name_group_folder + '\\' + folder + r'\adjointdata.csv'

    df = pd.read_csv(path_adj_data)

    it = np.asarray(df.it)
    rho = np.asarray(df.rho)

    plt.semilogy(it, rho, label=single_folders[folder])

plt.title("Rho", fontsize=10)
plt.grid(True)
plt.legend()


####################################################################################################################
#                                                                                                                  #
#                                               Regularization Rho n                                               #
#                                                                                                                  #
####################################################################################################################

plt.subplot(2, 1, 2)

for folder in single_folders:

    path_adj_data = name_group_folder + '\\' + folder + r'\adjointdata.csv'

    df = pd.read_csv(path_adj_data)

    it = np.asarray(df.it)
    regularization_rho_n = np.asarray(df.regularization_rho_n)

    plt.semilogy(it, regularization_rho_n, label=single_folders[folder])

plt.title("Regularization Rho n", fontsize=10)
plt.grid(True)
plt.legend()

name_png = 'regularization_plot_comparison' + name_aid + '.png'

plt.savefig(name_group_folder + '\\' + name_png)
