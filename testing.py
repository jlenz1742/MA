import datetime
import os
import time
import json
import pandas as pd
import glob
import re
import numpy as np

a = np.array([1, 0, 1, 1, 1, 1, 1, 1, 1, 0])
b = np.array([1, 9])

c = np.delete(a, b)

print(c)