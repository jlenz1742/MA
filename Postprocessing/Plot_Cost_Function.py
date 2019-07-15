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

folders = ['2', '3']

# Chose target folder (Data folders have to be in this folder -> Output is stored in this specified folder)

target_folder = r'D:\00 Privat\01_Bildung\01_ETH Zürich\MSc\00_Masterarbeit' \
                r'\Data_Simulation\different_selection_reacting_eids'

########################################################################################################################
#                                                                                                                      #
#                                                      Plot                                                            #
#                                                                                                                      #
########################################################################################################################


size_figure = (9, 9)
fig = plt.figure(figsize=size_figure)
plt.suptitle("")

ax1 = fig.add_subplot(111)

if mode == 0:

    for g in gamma:

        path_adj_data = target_folder + '\\' + g + r'\out\adjointdata.csv'

        df = pd.read_csv(path_adj_data)

        it = np.asarray(df.it)
        totalCost = np.asarray(df.totalCost)

        plt.semilogy(it, totalCost, label=g)

elif mode == 1:

    for f in folders:

        path_adj_data = target_folder + '\\' + f + r'\out\adjointdata.csv'

        df = pd.read_csv(path_adj_data)

        it = np.asarray(df.it)
        totalCost = np.asarray(df.totalCost)

        plt.semilogy(it, totalCost, label=f)


plt.title("Cost Function", fontsize=10)
ax1.yaxis.major.formatter._useMathText = True
plt.grid(True)
plt.legend()

plt.savefig(target_folder + '\cost_function_comparison.png')
