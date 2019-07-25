import pandas as pd
import glob
import os
import numpy as np

########################################################################################################################
#                                                                                                                      #
#                                                        INPUT                                                         #
#                                                                                                                      #
########################################################################################################################

# Enter path of output

path_target = r'D:\00 Privat\01_Bildung\01_ETH Zürich\MSc\00_Masterarbeit\00_Simulations\4_Results\4.5\out'
network_id = '3'

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

# Load CSV Files

df_start = pd.read_csv(path_target + "\\" + mesh_data_file_start)
df_final = pd.read_csv(path_target + "\\" + mesh_data_file_final)

# # Find reacting IDs
#
# diameters_start = np.array(df_start['D'])
# diameters_final = np.array(df_final['D'])
#
# diameter_change = diameters_final / diameters_start
# diameter_change_updated = []
#
# for i in range(len(diameter_change)):
#
#     if (i % 2) == 0:
#         diameter_change_updated.append(diameter_change[i])
#
# diameter_change_final = np.asarray(diameter_change_updated)
#
# reacting_ids = np.where(diameter_change_final != 1)

# Find activated IDS

path_networks = r'D:\00 Privat\01_Bildung\01_ETH Zürich\MSc\00_Masterarbeit\01_Networks'

path_activated_eids = path_networks + '\\' + network_id + '\\' + 'adjointMethod' + '\\' + 'activated_eids.csv'
df = pd.read_csv(path_activated_eids)
activated_eids = df.columns
activated_eids_updated = []

for edge in activated_eids:

    activated_eids_updated.append(int(edge))

# Find Plasma Flow Change

plasma_flow_start = np.array(df_start['Fplasma'])
plasma_flow_final = np.array(df_final['Fplasma'])

plasma_flow_change = plasma_flow_final / plasma_flow_start
plasma_flow_change_updated = []

for i in range(len(plasma_flow_change)):

    if (i % 2) == 0:
        plasma_flow_change_updated.append(plasma_flow_change[i])

plasma_flow_change_final = np.asarray(plasma_flow_change_updated)

print('\n Relative flow change of the activated Vessels:\n ')

print(plasma_flow_change_final[activated_eids_updated])



