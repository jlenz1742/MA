import os
import glob
import numpy as np
import math
import pandas as pd
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d.art3d import Line3DCollection


########################################################################################################################
#                                                                                                                      #
#                                                        INPUT                                                         #
#                                                                                                                      #
########################################################################################################################

# Enter path of output

path_target = r'D:\00 Privat\01_Bildung\01_ETH Zürich\MSc\00_Masterarbeit\00_Simulations\6_Results\6.0\out'
network_id = '4'

########################################################################################################################


def remove_values_from_list(the_list, val):

    return [value for value in the_list if value != val]


# Find CSV Files

mesh_data_file_start = 'meshdata_9999.csv'
mesh_data_files = []
mesh_data_numbers = []
os.chdir(path_target)

for file in glob.glob("meshdata*"):

    mesh_data_files.append(file)

mesh_data_files = remove_values_from_list(mesh_data_files, mesh_data_file_start)

mesh_data_file_final = mesh_data_files[0]


df_start = pd.read_csv(path_target + "\\" + mesh_data_file_start)
df_final = pd.read_csv(path_target + "\\" + mesh_data_file_final)


x = df_start['x']
x_min = min(x)
x_max = max(x)

y = df_start['y']
y_min = min(y)
y_max = max(y)

z = df_start['z']
z_min = min(z)
z_max = max(z)

f_plasma_1 = np.array(df_start['Fplasma'])
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

t_1 = np.array(df_start['D'])
t_2 = np.array(df_final['D'])

t = t_2 / t_1
t_new = []
t_2_new = []

for i in range(len(t)):

    if (i % 2) == 0:
        t_2_new.append(t_2[i] / (4 / math.pow(10, 6)))
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

length_segs = len(segs)
width = [1] * length_segs
print(len(width), len(t_2_new))

# make the collection of segments
lc = Line3DCollection(segs, linewidths=t_2_new, cmap=plt.get_cmap('coolwarm'))
lc.set_array(t_new_array)  # color the segments by our parameter

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.add_collection3d(lc)
ax.set_xlim(x_min, x_max)
ax.set_ylim(y_min, y_max)
ax.set_zlim(z_min, z_max)

plt.colorbar(lc)
plt.show()

















