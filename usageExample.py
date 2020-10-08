from src.WrfEvaluation import *
import datetime
import pandas as pd

path1 = "/data/co2flux/common/rsegura/Outputs/HeatWave-2015/BouLac_4days_Reference_corrected_Copernicus_corLU_noMORPH/wrfout_d03/"
label1 = "BL"

paths = [path1]
labels = [label1]
path_to_outputs = "./outputs/"
path_to_radiosonde = "./input/2015/"
day1 = 4
day2 = 5
month1 = 7
month2 = 7
year1 = 2015
year2 = 2015

initial_date = datetime.datetime(year1, month1, day1)
final_date = datetime.datetime(year2, month2, day2)

simulation = final_date - initial_date
dates = []
for i in range(0, simulation.days):
    day = initial_date + datetime.timedelta(i)
    dates.append(day.strftime("%Y-%m-%d"))

for date in dates:
     print(date)
#     for hour in ['00','12']:
     for hour in ['12']:
        WRFevaluations = []
        for i, path in enumerate(paths): 
            we = WRFEvaluation()
            we.maxHeight = 2500
            we.compareVerticalProfile(path_to_radiosonde+date[2:4]+date[5:7]+date[8:10]+hour+".txt", path+"wrfout_d03_"+date+"_00.nc", int(hour), path_to_outputs+labels[i]+"_"+date[2:4]+date[5:7]+date[8:10]+hour, False, True, labels[i])
            WRFevaluations.append(we)
            #bias.append((we.dataFrame['wrf_mixratio']-we.dataFrame['ftr_mixratio']).mean())
            #bias.append(np.sqrt(((we.dataFrame['wrf_mixratio'] - we.dataFrame['ftr_mixratio'])* (we.dataFrame['wrf_mixratio'] - we.dataFrame['ftr_mixratio'])).mean()))
        WRFevaluations[0].minT = 0
        WRFevaluations[0].maxT = 25
        Models_comparison(WRFevaluations, 'ftr_WF', 'wrf_wspeed', 'Wind speed ($\mathrm{m\, s^{-1}}$)', 100, path_to_outputs+"WS_"+date[2:4]+date[5:7]+date[8:10]+hour)
        WRFevaluations[0].minT = 298
        WRFevaluations[0].maxT = 318
        Models_comparison(WRFevaluations, 'ftr_theta', 'wrf_theta', 'Potential temperature ($\mathrm{K}$)', 100, path_to_outputs+"THETA_"+date[2:4]+date[5:7]+date[8:10]+hour)
        WRFevaluations[0].minT = 0
        WRFevaluations[0].maxT = 18
        Models_comparison(WRFevaluations, 'ftr_mixratio', 'wrf_mixratio', 'Mixing ratio ($\mathrm{g\, kg^{-1}}$)', 100, path_to_outputs+"QVAPOR_"+date[2:4]+date[5:7]+date[8:10]+hour)

#print(radiosounding_PBLH)
#print(wrf_PBLH)
#print(bias)
