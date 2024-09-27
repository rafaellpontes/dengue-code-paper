import os

for  time in range(0,10):
    for forecasting_time in range(11,12):
        os.system("python3 main.py " + str(forecasting_time) + ' ' + str(time))  
