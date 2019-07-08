import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

''' INPUT '''

gamma = ['0_01', '0_1', '1_0', '10_0']

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
for g in gamma:

    path_adj_data = r'D:\00 Privat\01_Bildung\01_ETH Zürich\MSc\00_Masterarbeit\Data_Simulation\wo_rbcs\gamma_' + g + \
                    r'\out\adjointdata.csv'

    df = pd.read_csv(path_adj_data)

    it = np.asarray(df.it)
    totalCost = np.asarray(df.rho)

    plt.semilogy(it, totalCost, label=g)

plt.title("Rho", fontsize=10)
plt.grid(True)
plt.legend()

####################################################################################################################
#                                                                                                                  #
#                                               Regularization Rho n                                               #
#                                                                                                                  #
####################################################################################################################

plt.subplot(2, 1, 2)
for g in gamma:
    path_adj_data = r'D:\00 Privat\01_Bildung\01_ETH Zürich\MSc\00_Masterarbeit\Data_Simulation\wo_rbcs\gamma_' + g + \
                    r'\out\adjointdata.csv'

    df = pd.read_csv(path_adj_data)

    it = np.asarray(df.it)
    totalCost = np.asarray(df.regularization_rho_n)

    plt.semilogy(it, totalCost, label=g)

plt.title("Regularization Rho n", fontsize=10)
plt.grid(True)
plt.legend()

plt.savefig(r'D:\00 Privat\01_Bildung\01_ETH Zürich\MSc\00_Masterarbeit\Data_Simulation\wo_rbcs'
            r'\Regularization_plot_comparison.png')
