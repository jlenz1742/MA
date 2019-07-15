import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Specify Group Folder

name_group_folder = r'D:\00 Privat\01_Bildung\01_ETH Zürich\MSc\00_Masterarbeit\00_Simulations\0'

# Specify single folders

single_folders = {'0.3': 'Gamma = 1', '0.7': 'Gamma = 10', '0.11': 'Gamma = 15', '0.15': 'Gamma = 25',
                  '0.19': 'Gamma = 50'}

# Name output file

name_aid = 'all'

########################################################################################################################
#                                                                                                                      #
#                                                      Plot                                                            #
#                                                                                                                      #
########################################################################################################################

size_figure = (9, 9)
fig = plt.figure(figsize=size_figure)
plt.suptitle("")

ax1 = fig.add_subplot(111)

for folder in single_folders:

    path_adj_data = name_group_folder + '\\' + folder + r'\out\adjointdata.csv'

    df = pd.read_csv(path_adj_data)

    it = np.asarray(df.it)
    totalCost = np.asarray(df.totalCost)

    plt.semilogy(it, totalCost, label=single_folders[folder])

plt.title("Cost Function", fontsize=10)
ax1.yaxis.major.formatter._useMathText = True
plt.grid(True)
plt.legend()

name_png = 'cost_function_comparison_' + name_aid + '.png'

plt.savefig(name_group_folder + '\\' + name_png)
