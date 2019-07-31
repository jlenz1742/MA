import Import_Penetrating_Trees
import Plot
import numpy as np
import random

path_arterial_trees = r'D:\00 Privat\01_Bildung\01_ETH Zürich\MSc\00_Masterarbeit\database_penetrating_trees\arteryDB'
path_venous_trees = r'D:\00 Privat\01_Bildung\01_ETH Zürich\MSc\00_Masterarbeit\database_penetrating_trees\veinDB'

target_path_arterial_trees = r'D:\00 Privat\01_Bildung\01_ETH Zürich\MSc\00_Masterarbeit\database_penetrating_trees\00_Plots\arterial'
target_path_venous_trees = r'D:\00 Privat\01_Bildung\01_ETH Zürich\MSc\00_Masterarbeit\database_penetrating_trees\00_Plots\venous'

number_arterial_trees = 58
number_venous_trees = 103

arterial_trees_ids = np.arange(number_arterial_trees)
venous_trees_ids = np.arange(number_venous_trees)

# Plots arterial trees
'''
for _id_ in arterial_trees_ids:

    print(_id_)
    arterial_tree = Import_Penetrating_Trees.get_penetrating_tree_from_pkl_file(path_arterial_trees, _id_, 1, 2)
    name_ = target_path_arterial_trees + '\\' + str(_id_) + '.png'
    Plot.plot_graph_selection_data_base(arterial_tree, name_)

# Plots venous trees

for _id_ in venous_trees_ids:

    print(_id_)
    venous_tree = Import_Penetrating_Trees.get_penetrating_tree_from_pkl_file(path_venous_trees, _id_, 1, 1)
    name_ = target_path_venous_trees + '\\' + str(_id_) + '.png'
    Plot.plot_graph_selection_data_base(venous_tree, name_)
'''




