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
    maxHeight = 0
    pblHfromRadiosondatge = 0
    pblHFrom_wrf = 0
    label = ''
    
    def __init__(self):
        self.minT = -10
        self.maxT = 100
        self.P0 = 100000 # Standard reference pressure in pascals
        self.columnsFromRadiosondatge = ['ftr_LAT','ftr_LON','ftr_alt','ftr_temp', 'ftr_pres', 'ftr_hum'] 
        # Other avaliable columns in radiosondatge files ftr_DP, ftr_WF, ftr_WD, ftr_VEF, ftr_VNF
        self.dataFrame = pd.DataFrame()
        self.maxHeight = 2500
        self.pblHfromRadiosondatge = 0
        self.pblHFrom_wrf = 0
        self.label = ''

    def compareVerticalProfile(self, radiosondatgeFile, wrf_file, wrf_cell_Time, outputFile, plot, save, label):
        startTime = time.time()
        self.label = label
        call = inspect.currentframe().f_code.co_name+"('"+radiosondatgeFile+"', '"+wrf_file+"', "+str(wrf_cell_Time)+", '"+outputFile+"', "+str(plot)+", "+str(save)+", "+label+")"
        wrf_file = Dataset(wrf_file, "r", format="NETCDF3")
        XLONG = np.array(wrf_file.variables['XLONG'][wrf_cell_Time])
        XLAT = np.array(wrf_file.variables['XLAT'][wrf_cell_Time])
        THETA = np.array(wrf_file.variables['T'][wrf_cell_Time])
        PH = np.array(wrf_file.variables['PH'][wrf_cell_Time])
        PHB = np.array(wrf_file.variables['PHB'][wrf_cell_Time])
        PB = np.array(wrf_file.variables['PB'][wrf_cell_Time])
        P = np.array(wrf_file.variables['P'][wrf_cell_Time])
        PBLH = np.array(wrf_file.variables['PBLH'][wrf_cell_Time])
        self.dataFrame = self.getColumnsFromRadiosondatgeFile( radiosondatgeFile, self.columnsFromRadiosondatge)
        self.cutDataFrame()
        self.addVirtualPotentialTemperature()
        self.addWrfClosestCell(XLONG, XLAT)
        self.addWrfTemperaturePredictionTodataFrame(THETA, PH, PHB, P, PB)
        self.addPBLHtoDataFrame(PBLH)
        self.pblHFromRadiosondatge = self.computePBLHeightOnRadiosondatgeWithGradient()

        self.pblHFrom_wrf = self.dataFrame['wrf_pblh'].mean()
        if plot == True:
            self.plotVerticalTemperatureProfile(
                self.dataFrame['ftr_alt'],
                self.dataFrame['ftr_temp'],
                self.dataFrame['wrf_temp'],
                self.pblHFrom_wrf,
                self.dataFrame['Potential_Temperature'],
                100,
                self.pblHFromRadiosondatge,
                outputFile,
                save
            )
        
        endTime = time.time()
        return {
                "Temp mean square error" : self.getMeanSquareError(self.dataFrame),
                "difference of PBL in metters" : self.pblHFromRadiosondatge-self.pblHFrom_wrf,
                "call to repeat calculations" : call,
                "execution time in seconds" : endTime-startTime
                }

    def cutDataFrame(self):
        #data2500 = self.dataFrame[self.dataFrame.loc[:,'ftr_alt'] < self.maxHeight]        
        self.dataFrame = self.dataFrame[self.dataFrame['ftr_alt'] < self.maxHeight]
        
    
    #given a point by lat and lon return the wrf indexes where the point is
    def getClosestCellInWrfFile(self, XLONG, XLAT, lat, lon):
        distance = abs(XLONG-lon)+abs(XLAT-lat)
        minimumValue = np.amin(distance)
        res = np.where(distance == minimumValue)
        return [res[0][0], res[1][0]]
    
    # given a randiosondatge File and a list of fields returns a dataset with those fields/columns
    def getColumnsFromRadiosondatgeFile(self, file, columnsList):
        taulaRadiosondatge = pd.read_csv(file, sep="\t")
        return taulaRadiosondatge[columnsList]
    


    def addWrfClosestCell(self, XLONG, XLAT):
        wrf_indexes = []
        for index, row in self.dataFrame.iterrows():
            wrf_indexes.append(self.getClosestCellInWrfFile(XLONG, XLAT, row["ftr_LAT"], row["ftr_LON"]))
    
        wrf_indexes = np.array(wrf_indexes)
        self.dataFrame['wrf_index_lat'] = wrf_indexes[:,0]
        self.dataFrame['wrf_index_lon'] = wrf_indexes[:,1]
        return 1
    
    def getCenteredHeight(self, lower_height):
        centered_height = []
        for i in range(len(lower_height)):
            if i + 1 >= len(lower_height):
                break
            centered_height.append( (lower_height[i] + lower_height[i+1])/2 )
        return centered_height
    
    def getLowerInterestLayer(self, h, centered_height):
        for i in range(len(centered_height)):
            if centered_height[i] < h and h < centered_height[i+1]:
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
    
    def getWrfTempInCelFromLatILonIandH(self, latI, lonI, h, THETA, PH, PHB, P, PB):
        lat = int(latI)
        lon = int(lonI)

        wrf_PH = PH[:,lat,lon]
        wrf_PHB = PHB[:,lat,lon]
        height = (wrf_PH + wrf_PHB)/9.81
    
        centered_height = self.getCenteredHeight(height)
    
        lowerLayer = self.getLowerInterestLayer(h, centered_height)
    
        lowh = centered_height[lowerLayer]
        upperh = centered_height[lowerLayer+1]
    
        Tlowh = self.fromWrfToCelcius( 
            THETA[lowerLayer,lat,lon],
            PB[lowerLayer,lat,lon],
            P[lowerLayer,lat,lon]
        )
    
        Tupperh = self.fromWrfToCelcius( 
            THETA[lowerLayer+1,lat,lon],
            PB[lowerLayer+1,lat,lon],
            P[lowerLayer+1,lat,lon]
        )
    
        wrf_prediction = self.getTemperatureFromWrfAt(h,lowh,Tlowh,upperh,Tupperh)
        wrf_thetaprediction = self.getTemperatureFromWrfAt(h, lowh, THETA[lowerLayer,lat,lon]+300, upperh, THETA[lowerLayer+1,lat,lon]+300)
        return wrf_prediction, wrf_thetaprediction
    
    def addWrfTemperaturePredictionTodataFrame(self, THETA, PH, PHB, P, PB):
        wrf_temperatures = []
        wrf_thetas = []
        for index, row in self.dataFrame.iterrows():
            temp, theta = self.getWrfTempInCelFromLatILonIandH(
                    row['wrf_index_lat'], 
                    row['wrf_index_lon'], 
                    row['ftr_alt'], 
                    THETA, PH, PHB, P, PB)
            wrf_temperatures.append(temp)
            wrf_thetas.append(theta -273.15)
        self.dataFrame['wrf_temp'] = wrf_temperatures
        self.dataFrame['wrf_theta'] = wrf_thetas
        return 1 #radiosondatgeDataFrame

    def addPBLHtoDataFrame(self, PBLH):
        wrf_PBLH = []
        for index, row in self.dataFrame.iterrows():
            wrf_PBLH.append(
                PBLH[ int(row['wrf_index_lat']), int(row['wrf_index_lon']) ]
                    )
        self.dataFrame['wrf_pblh'] = wrf_PBLH
        return 1 #radiosondatgeDataFrame  

    def plotLine(self, point1, point2, lab):
        x_values = [point1[0], point2[0]]
        y_values = [point1[1], point2[1]]
        plt.plot(x_values, y_values, label = lab)

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
   
    def computePBLHeightOnRadiosondatgeWithGradient(self):
        # Detecta el máxim del gradient de temperatura potencial
        MaxGradient = 0
        PBLH = 0
        for index, row in self.dataFrame.iterrows():
            if index < len(self.dataFrame) - 1:             
                gradient = (self.dataFrame.loc[index+1,'Potential_Temperature'] - self.dataFrame.loc[index,'Potential_Temperature'])/(self.dataFrame.loc[index+1,'ftr_alt'] - self.dataFrame.loc[index,'ftr_alt'])
                if gradient > MaxGradient:
                    MaxGradient = gradient
                    PBLH = self.dataFrame.loc[index,'ftr_alt']
        return PBLH

    def plotVerticalTemperatureProfile(self, heights, temperatures, wrf_temperatures, wrf_pblh, virtualPotentialTemperature , dpi, pblHFromRadiosondatge, outputFile, save):
        plt.figure(figsize=[8, 8])
        self.plotLine([self.minT, wrf_pblh], [self.maxT, wrf_pblh], "PBLH "+self.label)
        self.plotLine([self.minT, pblHFromRadiosondatge], [self.maxT, pblHFromRadiosondatge], "PBLH radisonde")
        plt.plot(temperatures, heights, 'bo',markersize=1)
        plt.plot(wrf_temperatures, heights, 'ro',markersize=1)
        plt.plot(virtualPotentialTemperature, heights, 'rv',markersize=1)        
        plt.ylim(0,self.maxHeight)
        plt.xlim(self.minT,self.maxT)
        plt.ylabel('Height (m)', fontsize=16)
        plt.xlabel('Temperature ($^\circ$C)', fontsize=16)
        plt.grid(axis='y')
        plt.tight_layout()
        if save == True:
            plt.savefig(outputFile, dpi=dpi)

    def potentialTemperature(self, Temperature, Pressure):
        Theta = (Temperature + 273.15)*(self.P0/(Pressure*100))**(2/7)
        return Theta
    
    def virtualPotentialTemperature(self, potentialTemperature, mixingRatio):
        vTheta = (potentialTemperature + 273.15) * (1 + 0.61*mixingRatio)
        return vTheta
    
    def virtualTemperature(self, Temperature, mixingRatio):
        vTemperature = (Temperature + 273.15) * (1 + 0.61*mixingRatio)
        return vTemperature
    
    def mixingRatio(self, HR, Temperature, Pressure):
        eps = 0.622
        es = 6.112 * math.exp(17.67*Temperature/(Temperature + 243.5))
        ws = eps * es /(Pressure - (1 - eps)*es)
        w = (HR/100)*ws
        return w

    def addVirtualPotentialTemperature(self):
        potentialTemperature = []
        for index, row in self.dataFrame.iterrows():
            potentialTemperature.append(self.potentialTemperature(row['ftr_temp'],  row['ftr_pres']) - 273.15)
        self.dataFrame['Potential_Temperature'] = potentialTemperature
        virtualPotentialTemperature = []
        for index, row in self.dataFrame.iterrows():
            virtualPotentialTemperature.append(self.virtualPotentialTemperature(row['Potential_Temperature'], self.mixingRatio(row['ftr_hum'], row['ftr_temp'], row['ftr_pres'])) - 273.15)
        self.dataFrame['Virtual_Potential_Temperature'] = virtualPotentialTemperature
        virtualTemperature = []
        for index, row in self.dataFrame.iterrows():
            virtualTemperature.append(self.virtualTemperature(row['ftr_temp'], self.mixingRatio(row['ftr_hum'], row['ftr_temp'], row['ftr_pres'])) - 273.15)
        self.dataFrame['Virtual_Temperature'] = virtualTemperature
        
def Models_comparison(we1, we2, variable_radio, variable_wrf, variable_label, dpi, outputFile):
    plt.figure(figsize=[8, 8])
    we1.plotLine([we1.minT, we1.pblHFromRadiosondatge], [we1.maxT, we1.pblHFromRadiosondatge], "PBLH radiosonde")
    we1.plotLine([we1.minT, we1.pblHFrom_wrf], [we1.maxT, we1.pblHFrom_wrf], "PBLH "+we1.label)
    we2.plotLine([we1.minT, we2.pblHFrom_wrf], [we1.maxT, we2.pblHFrom_wrf], "PBLH "+we2.label)
    plt.plot(we1.dataFrame[variable_radio], we1.dataFrame['ftr_alt'], 'bo',markersize=1, label='Radiosonde')
    plt.plot(we1.dataFrame[variable_wrf], we1.dataFrame['ftr_alt'], 'ro',markersize=1, label=we1.label)
    plt.plot(we2.dataFrame[variable_wrf], we1.dataFrame['ftr_alt'], 'go',markersize=1, label=we2.label)        
    plt.ylim(0,we1.maxHeight)
    plt.xlim(we1.minT,we1.maxT)
    plt.legend(fontsize='medium')
    plt.ylabel('Height (m)', fontsize=16)
    plt.xlabel(variable_label, fontsize=16)
    plt.grid(axis='y')
    plt.tight_layout()
    plt.savefig(outputFile, dpi=dpi)
    
    
        
    