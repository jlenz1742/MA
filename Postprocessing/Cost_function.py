import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def different_gammas(main_folder, folders, gammas):

    name_group_folder = main_folder

    size_figure = (9, 9)
    fig = plt.figure(figsize=size_figure)
    plt.suptitle("")

    ax1 = fig.add_subplot(111)

    for i in range(len(folders)):

        path_adj_data = name_group_folder + '\\' + folders[i] + r'\adjointdata.csv'

        df = pd.read_csv(path_adj_data)

        it = np.asarray(df.it)
        total_cost = np.asarray(df.totalCost)

        plt.semilogy(it, total_cost, label=gammas[i])

    plt.title("Cost Function", fontsize=10)
    ax1.yaxis.major.formatter._useMathText = True
    plt.grid(True)
    plt.legend()

    name_png = 'Cost_function_different_gammas.png'

    plt.savefig(name_group_folder + '\\' + name_png)
