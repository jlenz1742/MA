import datetime
import os
import time
print(type(datetime.datetime.now()))

a = str(datetime.datetime.now())

time_str = time.strftime("%Y%m%d_%H%M%S")
os.makedirs('test_' + time_str)