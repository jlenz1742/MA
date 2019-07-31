import B_Cost_Function
import C_Regularization

########################################################################################################################
#                                                                                                                      #
#                                                        INPUT                                                         #
#                                                                                                                      #
########################################################################################################################

folder = r'D:\00 Privat\01_Bildung\01_ETH Zürich\MSc\00_Masterarbeit\00_Simulations\6_Results'

simulation_ids = ['6.1', '6.4', '6.5']

gammas = [2000, 5000, 10000]
radii = [150, 250, 350]
epsilons = [0.3, 0.5, 0.7, 0.9]

codes = [1]

# 0: Cost function / Regularization with different gammas
# 1: Cost function / Regularization with different radii
# 2: Cost function / Regularization with different epsilon

########################################################################################################################

# COST FUNCTION --------------------------------------------------------------------------------------------------------

main_folder = folder
single_folders = simulation_ids

B_Cost_Function.execute(codes, main_folder, single_folders, gammas, radii, epsilons)

# REGULARIZATION -------------------------------------------------------------------------------------------------------

main_folder = folder
single_folders = simulation_ids

C_Regularization.execute(codes, main_folder, single_folders, gammas, radii, epsilons)

# ----------------------------------------------------------------------------------------------------------------------















