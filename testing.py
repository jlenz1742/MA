import datetime
import os
import time
import json
import pandas as pd
import glob
import re

test =[1,0,0,0,0,0,1]

indices = [i for i in range(len(test)) if test[i] == 1]

print(indices)

a = [[1,0],[],[3, 0],[5]]

print(a[-1][0])