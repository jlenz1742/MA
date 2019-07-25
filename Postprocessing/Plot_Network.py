import numpy as np
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d.art3d import Line3DCollection
import pandas as pd
import matplotlib.colors as colors

def plot_network_from_meshdata_file():

    df = pd.read_csv(
        r'D:\00 Privat\01_Bildung\01_ETH Zürich\MSc\00_Masterarbeit\00_Simulations\4_Results\4.0\out\meshdata_9999.csv')

    df_final = pd.read_csv(r'D:\00 Privat\01_Bildung\01_ETH Zürich\MSc\00_Masterarbeit\00_Simulations\4_Results\4.0\out\meshdata_7019999.csv')

    headers = list(df.columns.values)

    # print(headers)

    x = df['x']
    x_min = min(x)
    x_max = max(x)

    y = df['y']
    y_min = min(y)
    y_max = max(y)

    z = df['z']
    z_min = min(z)
    z_max = max(z)

    f_plasma_1 = np.array(df['Fplasma'])
    f_plasma_2 = np.array(df_final['Fplasma'])

    f_plasma_ratio = f_plasma_2 / f_plasma_1

    test = []

    for i in range(len(f_plasma_ratio)):

        if f_plasma_ratio[i] > 2:

            test.append(1)

        elif f_plasma_ratio[i] < 0:

            test.append(1)

        else:

            test.append(f_plasma_ratio[i])

    f_plasma_ratio_processed = np.asarray(test)

    print('Max and Min Plasma Flow Processed: ', max(f_plasma_ratio_processed), min(f_plasma_ratio_processed))

    t_1 = np.array(df['D'])
    t_2 = np.array(df_final['D'])

    t = t_2 / t_1
    t_new = []

    for i in range(len(t)):

        if (i % 2) == 0:

            t_new.append(t[i])

    t_new_array = np.asarray(t_new)
    reacting_ids = np.where(t_new_array != 1)

    print(f_plasma_ratio_processed[reacting_ids])

    # generate a list of (x,y,z) points
    points = np.array([x, y, z]).transpose().reshape(-1, 1, 3)

    # set up a list of segments
    segs_temp = np.concatenate([points[:-1], points[1:]], axis=1)
    segs = []

    for i in range(len(segs_temp)):

        if (i % 2) == 0:

            segs.append(segs_temp[i])

    # make the collection of segments
    lc = Line3DCollection(segs, cmap=plt.get_cmap('coolwarm'))
    lc.set_array(t_new_array)  # color the segments by our parameter

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.add_collection3d(lc)
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    ax.set_zlim(z_min, z_max)

    plt.colorbar(lc)
    plt.show()

plot_network_from_meshdata_file()
