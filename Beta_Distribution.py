import numpy as np
import math
import matplotlib.pyplot as plt


def get_value_from_beta_distribution(mu, sigma):

    alpha = mu ** 2 * ((1 - mu) / sigma ** 2 - 1 / mu)

    beta = alpha * (1 / mu - 1)

    value = np.random.beta(alpha, beta)

    return value

########################################################################################################################
#                                                    Test Plot                                                         #
########################################################################################################################

# data = []

# for i in range(1000):
#
#     data.append(get_value_from_beta_distribution(4 / math.pow(10, 6), 1 / math.pow(10, 6)))

# fig, ax = plt.subplots()
# ax.hist(data, 50)
# props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
# plt.show()
