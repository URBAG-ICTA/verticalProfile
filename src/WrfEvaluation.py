import numpy as np
import math
import pandas as pd
from netCDF4 import Dataset
import matplotlib.pyplot as plt
import inspect
import time;

class WrfEvaluation:

    minT = 0
    maxT = 0
    P0 = 0 # Standard reference pressure in pascals
    columnsFromRadiosondatge = [] 
    # Other avaliable columns in radiosondatge files ftr_DP, ftr_WF, ftr_WD, ftr_VEF, ftr_VNF
    dataFrame = pd.DataFrame()
    maxAltura = 0
    
    
    def __init__(self):
        self.minT = -10
        self.maxT = 100
        self.P0 = 100000 # Standard reference pressure in pascals
        self.columnsFromRadiosondatge = ['ftr_LAT','ftr_LON','ftr_alt','ftr_temp', 'ftr_pres', 'ftr_hum'] 
        # Other avaliable columns in radiosondatge files ftr_DP, ftr_WF, ftr_WD, ftr_VEF, ftr_VNF
        self.dataFrame = pd.DataFrame()
        self.maxAltura = 2500

    def compareVerticalProfile(self, radiosondatgeFile, wrf_file, wrf_cell_Time, outputFile, plot, save):
        startTime = time.time()
        call = inspect.currentframe().f_code.co_name+"('"+radiosondatgeFile+"', '"+wrf_file+"', "+str(wrf_cell_Time)+", '"+outputFile+"', "+str(plot)+", "+str(save)+")"
        wrf_file = Dataset(wrf_file, "r", format="NETCDF3")
        self.dataFrame = self.getColumnsFromRadiosondatgeFile( radiosondatgeFile, self.columnsFromRadiosondatge)
        self.cutDataFrame()
        self.addVirtualPotentialTemperature()
        self.addWrfClosestCell(wrf_file, wrf_cell_Time)
        self.addWrfTemperaturePredictionTodataFrame(wrf_file, wrf_cell_Time)
        self.addPBLHtoDataFrame(wrf_file, wrf_cell_Time)
        pblHFromRadiosondatge = self.computePBLHeightOnRadiosondatge()

        pblHFrom_wrf = self.dataFrame['wrf_pblh'].mean()
        if plot == True:
            self.plotVerticalTemperatureProfile(
                self.dataFrame['ftr_alt'],
                self.dataFrame['ftr_temp'],
                self.dataFrame['wrf_temp'],
                pblHFrom_wrf,
                self.dataFrame['Potential_Temperature'],
                100,
                pblHFromRadiosondatge,
                outputFile,
                save
            )
        
        endTime = time.time()
        return {
                "Temp mean square error" : self.getMeanSquareError(self.dataFrame),
                "difference of PBL in metters" : pblHFromRadiosondatge-pblHFrom_wrf,
                "call to repeat calculations" : call,
                "execution time in seconds" : endTime-startTime
                }

    def cutDataFrame(self):
        #data2500 = self.dataFrame[self.dataFrame.loc[:,'ftr_alt'] < self.maxAltura]        
        self.dataFrame = self.dataFrame[self.dataFrame['ftr_alt'] < self.maxAltura]
        
    
    #given a point by lat and lon return the wrf indexes where the point is
    def getClosestCellInWrfFile(self, wrf_file, lat, lon, wrf_cell_Time):
        timeCell = wrf_cell_Time
        XLONG = np.array(wrf_file.variables['XLONG'][timeCell])
        XLAT = np.array(wrf_file.variables['XLAT'][timeCell])
    
        distance = abs(XLONG-lon)+abs(XLAT-lat)
        minimumValue = np.amin(distance)
        res = np.where(distance == minimumValue)
        return [res[0][0], res[1][0]]
    
    # given a randiosondatge File and a list of fields returns a dataset with those fields/columns
    def getColumnsFromRadiosondatgeFile(self, file, columnsList):
        taulaRadiosondatge = pd.read_csv(file, sep="\t")
        return taulaRadiosondatge[columnsList]
    


    def addWrfClosestCell(self, wrf_dataset, wrf_cell_Time):
        wrf_indexes = []
        for index, row in self.dataFrame.iterrows():
            wrf_indexes.append(self.getClosestCellInWrfFile(wrf_dataset, row["ftr_LAT"], row["ftr_LON"], wrf_cell_Time))
    
        wrf_indexes = np.array(wrf_indexes)
        self.dataFrame['wrf_index_lat'] = wrf_indexes[:,0]
        self.dataFrame['wrf_index_lon'] = wrf_indexes[:,1]
        return 1
    
    def getCenteredHight(self, lower_hight):
        centered_hight = []
        for i in range(len(lower_hight)):
            if i + 1 >= len(lower_hight):
                break
            centered_hight.append( (lower_hight[i] + lower_hight[i+1])/2 )
        return centered_hight
    
    def getLowerInterestLayer(self, h, centered_hight):
        for i in range(len(centered_hight)):
            if centered_hight[i] < h and h < centered_hight[i+1]:
                return i
        return False

    def getTemperatureFromWrfAt(self, h,lowh,Tlowh,upperh,Tupperh):
        tD = abs(lowh-upperh) #totalDistance
        Th = (abs(upperh-h)/tD)*Tlowh + (abs(h-lowh)/tD)*Tupperh
        return Th
    
    def fromWrfToCelcius(self, T, PB, P):
        theta = T + 300
        ptot = PB + P
        temp = theta*math.pow((ptot/self.P0),(2/7))
        return temp-273.15
    
    def getWrfTempInCelFromLatILonIandH(self, latI, lonI, h, wrf_file, timeCell):
        lat = latI
        lon = lonI
    
        #wrf_temp = wrf_file.variables['T'][0,:,lat,lon]
        wrf_PH = wrf_file.variables['PH'][0,:,lat,lon]
        wrf_PHB = wrf_file.variables['PHB'][0,:,lat,lon]
        hight = (wrf_PH + wrf_PHB)/9.8
    
        centered_hight = self.getCenteredHight(hight)
    
        lowerLayer = self.getLowerInterestLayer(h, centered_hight)
    
        lowh = centered_hight[lowerLayer]
        upperh = centered_hight[lowerLayer+1]
    
        Tlowh = self.fromWrfToCelcius( 
            wrf_file.variables['T'][timeCell,lowerLayer,lat,lon],
            wrf_file.variables['PB'][timeCell,lowerLayer,lat,lon],
            wrf_file.variables['P'][timeCell,lowerLayer,lat,lon]
        )
    
        Tupperh = self.fromWrfToCelcius( 
            wrf_file.variables['T'][timeCell,lowerLayer+1,lat,lon],
            wrf_file.variables['PB'][timeCell,lowerLayer+1,lat,lon],
            wrf_file.variables['P'][timeCell,lowerLayer+1,lat,lon]
        )
    
        wrf_prediction = self.getTemperatureFromWrfAt(h,lowh,Tlowh,upperh,Tupperh)
    
        return wrf_prediction

    def addWrfTemperaturePredictionTodataFrame(self, wrf_dataset, wrf_cell_Time):
        wrf_temperatures = []
        
        for index, row in self.dataFrame.iterrows():
            wrf_temperatures.append(
                self.getWrfTempInCelFromLatILonIandH(
                    row['wrf_index_lat'], 
                    row['wrf_index_lon'], 
                    row['ftr_alt'], 
                    wrf_dataset,
                    wrf_cell_Time)
            )
        self.dataFrame['wrf_temp'] = wrf_temperatures
        return 1 #radiosondatgeDataFrame

    def addPBLHtoDataFrame(self, wrf_dataset, wrf_cell_Time):
        wrf_PBLH = []
        for index, row in self.dataFrame.iterrows():
            wrf_PBLH.append(
                wrf_dataset.variables['PBLH'][ wrf_cell_Time ,row['wrf_index_lat'], row['wrf_index_lon'] ]
                    )
        self.dataFrame['wrf_pblh'] = wrf_PBLH
        return 1 #radiosondatgeDataFrame  

    def plotLine(self, point1, point2):
        x_values = [point1[0], point2[0]]
        y_values = [point1[1], point2[1]]
        plt.plot(x_values, y_values)

    def getMeanSquareError(self, dataFrame):
        Real = dataFrame['ftr_temp']
        Pred = dataFrame['wrf_temp']
        X = (Real-Pred)**2
        return X.mean()

    def computePBLHeightOnRadiosondatge(self):
        # Detecta la primera inversio tèrmica de més d'un grau
        firstMinimum = 100
        for index, row in self.dataFrame.iterrows():
            if row['ftr_temp'] <= firstMinimum:
                firstMinimum = row['ftr_temp']
            if row['ftr_temp'] > firstMinimum and abs(row['ftr_temp'] - firstMinimum) > 1:
                return row['ftr_alt']
        return 0

    def plotVerticalTemperatureProfile(self, heights, temperatures, wrf_temperatures, wrf_pblh, potentialTemperature , dpi, pblHFromRadiosondatge, outputFile, save):
        plt.figure(figsize=[8, 8])
        self.plotLine([self.minT, wrf_pblh], [self.maxT, wrf_pblh])
        self.plotLine([self.minT, pblHFromRadiosondatge], [self.maxT, pblHFromRadiosondatge])
        plt.plot(temperatures, heights, 'bo',markersize=1)
        plt.plot(wrf_temperatures, heights, 'ro',markersize=1)
        plt.plot(potentialTemperature, heights, 'ro',markersize=1)        
        plt.ylim(0,2500)
        plt.xlim(self.minT,self.maxT)
        plt.ylabel('Height (m)', fontsize=16)
        plt.xlabel('Temperature ($^\circ$)', fontsize=16)
        plt.grid(axis='y')
        plt.tight_layout()
        if save == True:
            plt.savefig(outputFile, dpi=dpi)

    def potentialTemperature(self, Temperature, Pressure):
        Theta = Temperature*(self.P0/Pressure)**(2/7)
        return Theta

    def addVirtualPotentialTemperature(self):
        potentialTemperature = []
        for index, row in self.dataFrame.iterrows():
            potentialTemperature.append(self.potentialTemperature(row['ftr_temp'],  row['ftr_pres']))
        self.dataFrame['Potential_Temperature'] = potentialTemperature
            
