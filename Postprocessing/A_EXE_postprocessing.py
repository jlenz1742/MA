import B_Cost_Function
import C_Regularization

########################################################################################################################
#                                                                                                                      #
#                                                        INPUT                                                         #
#                                                                                                                      #
########################################################################################################################

folder = r'D:\00 Privat\01_Bildung\01_ETH Zürich\MSc\00_Masterarbeit\00_Simulations\3_Results'

simulation_ids = ['3.2', '3.5']

gammas = []
radii = [150, 250]
epsilons = []

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















