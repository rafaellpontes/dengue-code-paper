import os

for  time in range(0,5):
    for forecasting_time in range(15,19):
        os.system("python3 main.py " + str(forecasting_time) + ' ' + str(time))  
