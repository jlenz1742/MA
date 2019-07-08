import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

gamma = ['0_01', '0_1', '1_0', '10_0']

size_figure = (9, 9)
fig = plt.figure(figsize=size_figure)
plt.suptitle("")

ax1 = fig.add_subplot(111)

for g in gamma:

    path_adj_data = r'D:\00 Privat\01_Bildung\01_ETH Zürich\MSc\00_Masterarbeit\Data_Simulation\wo_rbcs\gamma_' + g + r'\out\adjointdata.csv'

    df = pd.read_csv(path_adj_data)

    it = np.asarray(df.it)
    totalCost = np.asarray(df.totalCost)

    plt.semilogy(it, totalCost, label=g)

plt.title("Cost Function", fontsize=10)
ax1.yaxis.major.formatter._useMathText = True
plt.grid(True)
plt.legend()

plt.savefig(r'D:\00 Privat\01_Bildung\01_ETH Zürich\MSc\00_Masterarbeit\Data_Simulation\wo_rbcs\cost_function_comparison.png')
