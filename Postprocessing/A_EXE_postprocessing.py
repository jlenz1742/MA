import B_Cost_Function
import C_Regularization

########################################################################################################################
#                                                                                                                      #
#                                                        INPUT                                                         #
#                                                                                                                      #
########################################################################################################################

folder = r'D:\00 Privat\01_Bildung\01_ETH Zürich\MSc\00_Masterarbeit\00_Simulations\4_Results'

simulation_ids = ['4.0', '4.1', '4.2', '4.3', '4.4', '4.5']

gammas = [50, 75, 100, 150, 200, 500]
radii = [150, 250]
epsilons = []

codes = [0]

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















