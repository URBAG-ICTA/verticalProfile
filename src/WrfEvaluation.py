import numpy as np
import math
import pandas as pd
from netCDF4 import Dataset
import matplotlib.pyplot as plt
import inspect
import time;
import matplotlib

class WRFEvaluation:

    minT = 0
    maxT = 0
    P0 = 0 # Standard reference pressure in pascals
    columnsFromRadiosondatge = [] 
    # Other avaliable columns in radiosondatge files ftr_DP, ftr_WF, ftr_WD, ftr_VEF, ftr_VNF
    dataFrame = pd.DataFrame()
    maxHeight = 0
    pblHfromRadiosondatge = 0
    pblHfromWRF = 0
    label = ''
    
    def __init__(self):
        self.minT = 0
        self.maxT = 0.015
        self.P0 = 100000 # Standard reference pressure in pascals
        self.columnsFromRadiosondatge = ['ftr_LAT','ftr_LON','ftr_alt','ftr_temp', 'ftr_pres', 'ftr_hum', 'ftr_WF', 'ftr_WD'] 
        # Other avaliable columns in radiosondatge files ftr_DP, ftr_VEF, ftr_VNF
        self.dataFrame = pd.DataFrame()
        self.maxHeight = 2500
        self.pblHfromRadiosondatge = 0
        self.pblHfromWRF = 0
        self.label = ''

    def compareVerticalProfile(self, radiosondatgeFile, wrf_file, wrf_cell_Time, outputFile, plot, save, label):
        startTime = time.time()
        self.label = label
        call = inspect.currentframe().f_code.co_name+"('"+radiosondatgeFile+"', '"+wrf_file+"', "+str(wrf_cell_Time)+", '"+outputFile+"', "+str(plot)+", "+str(save)+", "+label+")"
        wrf_file = Dataset(wrf_file, "r", format="NETCDF3")
        XLONG = np.array(wrf_file.variables['XLONG'][wrf_cell_Time])
        XLAT = np.array(wrf_file.variables['XLAT'][wrf_cell_Time])
        XLONG_U = np.array(wrf_file.variables['XLONG_U'][wrf_cell_Time])
        XLAT_U = np.array(wrf_file.variables['XLAT_U'][wrf_cell_Time])
        XLONG_V = np.array(wrf_file.variables['XLONG_V'][wrf_cell_Time])
        XLAT_V = np.array(wrf_file.variables['XLAT_V'][wrf_cell_Time])
        THETA = np.array(wrf_file.variables['T'][wrf_cell_Time])
        PH = np.array(wrf_file.variables['PH'][wrf_cell_Time])
        PHB = np.array(wrf_file.variables['PHB'][wrf_cell_Time])
        PB = np.array(wrf_file.variables['PB'][wrf_cell_Time])
        P = np.array(wrf_file.variables['P'][wrf_cell_Time])
        U = wrf_file.variables['U'][wrf_cell_Time]
        V = wrf_file.variables['V'][wrf_cell_Time]
        COSALPHA = wrf_file.variables['COSALPHA'][wrf_cell_Time]
        SINALPHA = wrf_file.variables['SINALPHA'][wrf_cell_Time]
        U_UNST = 0.5 * (U[:,:,:-1] + U[:,:,1:])
        V_UNST = 0.5 * (V[:,:-1,:] + V[:,1:,:])
        U = U_UNST * COSALPHA - V_UNST * SINALPHA
        V = V_UNST * COSALPHA + U_UNST * SINALPHA
        U = np.array(U)
        V = np.array(V)
        PBLH = np.array(wrf_file.variables['PBLH'][wrf_cell_Time])
        QVAPOR = np.array(wrf_file.variables['QVAPOR'][wrf_cell_Time])
        wrf_file.close()
        self.dataFrame = self.getColumnsFromRadiosondatgeFile(radiosondatgeFile, self.columnsFromRadiosondatge)
        self.cutDataFrame()
        self.addMixingRatio()
        self.addBulkRichardsonNumber()
        self.addWrfClosestCell(XLONG, XLAT)
        self.addWrfTemperaturePredictionTodataFrame(THETA, PH, PHB, P, PB, QVAPOR)
        self.addWrfWindSpeedPredictionTodataFrame(U, V, PH, PHB)
        self.addWrfMixingRationPredictiontodataFrame(QVAPOR, PH, PHB)
        self.addPBLHtoDataFrame(PBLH)
        self.pblHfromWRF = self.computePBLHeightOnWRFModel()
        
        #Seleccionar si es Stable Boundary Layer (SBL) or Convective Boundary Layer (CBL)
        self.pblHfromRadiosondatge = self.computePBLHeightOnRadiosondatgeForCBL()
        #self.pblHfromRadiosondatge = self.computePBLHeightOnRadiosondatgeForSBL() 
        self.pblHfromWRF = self.dataFrame['wrf_pblh'].mean()
        print('Planetary boundary layer predicted from WRF: ', self.pblHfromWRF)
        print('Planetery boundary layer computed from radiosounding: ', self.pblHfromRadiosondatge)
        if plot == True:
            self.plotVerticalTemperatureProfile(
                self.dataFrame['ftr_alt'],
                self.dataFrame['ftr_temp'],
                self.dataFrame['wrf_temp'],
                self.pblHfromWRF,
                self.dataFrame['Potential_Temperature'],
                100,
                self.pblHfromRadiosondatge,
                outputFile,
                save
            )
        
        endTime = time.time()
        return {
                "Temp mean square error" : self.getMeanSquareError(self.dataFrame),
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
    
    def addMixingRatio(self):
        MixingRatio = []
        for i, row in self.dataFrame.iterrows():
            MixingRatio.append(self.mixingRatio(row['ftr_hum'], row['ftr_temp'], row['ftr_pres']))
        self.dataFrame['ftr_mixratio'] = MixingRatio

    def addBulkRichardsonNumber(self):
        Ri = []
        Theta = []
        Thetav = []
        for i, row in self.dataFrame.iterrows():
            if i == 0:
                z0 = row['ftr_alt']
                theta = self.potentialTemperature(row['ftr_temp'], row['ftr_pres'])
                Theta.append(theta )
                thetav0 = self.virtualPotentialTemperature(theta - 273.15, row['ftr_mixratio'])
                Thetav.append(thetav0 - 273.15)
                Ri.append(0)
            else:
                theta = self.potentialTemperature(row['ftr_temp'], row['ftr_pres'])
                Theta.append(theta)
                thetav = self.virtualPotentialTemperature(theta - 273.15, row['ftr_mixratio'])
                Thetav.append(thetav - 273.15)
                Ri.append(self.bulkRichardsonNumber(thetav, thetav0, row['ftr_alt'], z0, row['ftr_WF']))
        self.dataFrame['ftr_theta'] = Theta
        self.dataFrame['ftr_thetav'] = Thetav
        self.dataFrame['ftr_Ri'] = Ri

    def addWrfClosestCell(self, XLONG, XLAT, suf = ''):
        wrf_indexes = []
        for index, row in self.dataFrame.iterrows():
            if index == 0:
               pos = self.getClosestCellInWrfFile(XLONG, XLAT, row["ftr_LAT"], row["ftr_LON"])
            wrf_indexes.append(pos)
    
        wrf_indexes = np.array(wrf_indexes)
        self.dataFrame['wrf_index_lat'+suf] = wrf_indexes[:,0]
        self.dataFrame['wrf_index_lon'+suf] = wrf_indexes[:,1]
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

    def getVariableFromWrfAtH(self, h,lowh,Tlowh,upperh,Tupperh):
        tD = abs(lowh-upperh) #totalDistance
        Th = (abs(upperh-h)/tD)*Tlowh + (abs(h-lowh)/tD)*Tupperh
        return Th
    
    def fromWrfToCelcius(self, T, PB, P):
        theta = T + 300
        ptot = PB + P
        temp = theta*math.pow((ptot/self.P0),(2/7))
        return temp-273.15
    
    def getWrfTempInCelFromLatILonIandH(self, latI, lonI, h, THETA, PH, PHB, P, PB, QVAPOR):
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
    
        wrf_TempPrediction = self.getVariableFromWrfAtH(h,lowh,Tlowh,upperh,Tupperh)
        wrf_ThetaPrediction = self.getVariableFromWrfAtH(h, lowh, THETA[lowerLayer, lat, lon]+300, upperh, THETA[lowerLayer+1, lat, lon]+300)
        wrf_QvaporPrediction = self.getVariableFromWrfAtH(h, lowh, QVAPOR[lowerLayer, lat, lon], upperh, QVAPOR[lowerLayer+1, lat, lon])
        wrf_vThetaPrediction = self.virtualPotentialTemperature(wrf_ThetaPrediction -273.15, wrf_QvaporPrediction)
        return wrf_TempPrediction, wrf_ThetaPrediction, wrf_vThetaPrediction
    
    def getWrfWSpeedInCelFromLatILonIandH(self, latI, lonI, h, U, V, PH, PHB):
        lat = int(latI)
        lon = int(lonI)

        wrf_PH = PH[:,lat,lon]
        wrf_PHB = PHB[:,lat,lon]
        height = (wrf_PH + wrf_PHB)/9.81

        centered_height = self.getCenteredHeight(height)
        lowerLayer = self.getLowerInterestLayer(h, centered_height)
        lowh = centered_height[lowerLayer]
        upperh = centered_height[lowerLayer + 1]
        WSlowh = np.sqrt((U[lowerLayer, lat, lon]**2)+ (V[lowerLayer, lat, lon]**2))
        WSupperh = np.sqrt((U[lowerLayer+1, lat, lon]**2)+ (V[lowerLayer+1, lat, lon]**2))
        wrf_wspeed = self.getVariableFromWrfAtH(h, lowh, WSlowh, upperh, WSupperh)
        return wrf_wspeed
    
    def getWrfMixingInCelFromLatILonIandH(self, latI, lonI, h, QVAPOR, PH, PHB):
        lat = int(latI)
        lon = int(lonI)

        wrf_PH = PH[:,lat,lon]
        wrf_PHB = PHB[:,lat,lon]
        height = (wrf_PH + wrf_PHB)/9.81

        centered_height = self.getCenteredHeight(height)
        lowerLayer = self.getLowerInterestLayer(h, centered_height)
        lowh = centered_height[lowerLayer]
        upperh = centered_height[lowerLayer+1]

        Qlowh = QVAPOR[lowerLayer,lat,lon]
        Qupperh = QVAPOR[lowerLayer+1,lat,lon]

        wrf_QvaporPrediction = self.getVariableFromWrfAtH(h, lowh, Qlowh, upperh, Qupperh)
        return wrf_QvaporPrediction*1000
    
    def addWrfTemperaturePredictionTodataFrame(self, THETA, PH, PHB, P, PB, QVAPOR):
        wrf_temp = []
        wrf_theta = []
        wrf_thetav = []
        for index, row in self.dataFrame.iterrows():
            temp, theta, thetav = self.getWrfTempInCelFromLatILonIandH(
                    row['wrf_index_lat'], 
                    row['wrf_index_lon'], 
                    row['ftr_alt'], 
                    THETA, PH, PHB, P, PB, QVAPOR)
            wrf_temp.append(temp)
            wrf_theta.append(theta)
            wrf_thetav.append(thetav -273.15)
        self.dataFrame['wrf_temp'] = wrf_temp
        self.dataFrame['wrf_theta'] = wrf_theta
        self.dataFrame['wrf_thetav'] = wrf_thetav
        return 1 #radiosondatgeDataFrame
    
    def addWrfWindSpeedPredictionTodataFrame(self, U, V, PH, PHB):
        wrf_wspeed = []
        for index, row in self.dataFrame.iterrows():
            wspeed = self.getWrfWSpeedInCelFromLatILonIandH(row['wrf_index_lat'], row['wrf_index_lon'], 
                    row['ftr_alt'], U, V, PH, PHB)
            wrf_wspeed.append(wspeed)
        self.dataFrame['wrf_wspeed'] = wrf_wspeed
        return 1

    def addWrfMixingRationPredictiontodataFrame(self, QVAPOR, PH, PHB):
        wrf_mixratio = []
        for index, row in self.dataFrame.iterrows():
            mixratio = self.getWrfMixingInCelFromLatILonIandH(row['wrf_index_lat'], row['wrf_index_lat'],
                    row['ftr_alt'], QVAPOR, PH, PHB)
            wrf_mixratio.append(mixratio)
        self.dataFrame['wrf_mixratio'] = wrf_mixratio
        return 1

    def addPBLHtoDataFrame(self, PBLH):
        wrf_PBLH = []
        for index, row in self.dataFrame.iterrows():
            wrf_PBLH.append(
                PBLH[ int(row['wrf_index_lat']), int(row['wrf_index_lon']) ]
                    )
        self.dataFrame['wrf_pblh'] = wrf_PBLH
        return 1 #radiosondatgeDataFrame  

    def plotLine(self, point1, point2, lab, colorLine):
        x_values = [point1[0], point2[0]]
        y_values = [point1[1], point2[1]]
        plt.plot(x_values, y_values, label = lab, color = colorLine)

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
   
    def computePBLHeightOnRadiosondatgeForCBL(self):
        # Detecta el máxim del gradient de temperatura potencial
        #First search
        k = 2
        for index, row in self.dataFrame.iterrows():
            if (index == 0):
                theta0 = row['ftr_theta']
            else:
                if (row['ftr_theta'] - theta0 >= 0.5):
                    k = index
                    break
        MinGradient = 0.004
        PBLH = 0
        for index, row in self.dataFrame.iterrows():
            if index >= k and index < (len(self.dataFrame)-1):             
                gradient = (self.dataFrame.loc[index+1,'ftr_theta'] - self.dataFrame.loc[index,'ftr_theta'])/(self.dataFrame.loc[index+1,'ftr_alt'] - self.dataFrame.loc[index,'ftr_alt'])
                if gradient >= MinGradient:
                    PBLH = self.dataFrame.loc[index,'ftr_alt']
                    break
        return PBLH
    
    def computePBLHeightOnRadiosondatgeForSBL(self):
        # Detecta el máxim del gradient de temperatura potencial
        #First search
        mingrad = 1000
        k = 2
        for index, row in self.dataFrame.iterrows():
            if (index < 2):
                continue
            else:
                grad = (self.dataFrame.loc[index,'ftr_theta'] - self.dataFrame.loc[index-1,'ftr_theta'])/(self.dataFrame.loc[index,'ftr_alt'] - self.dataFrame.loc[index-1,'ftr_alt'])
                if grad < mingrad:
                    mingrad = grad
                else:
                    k = index -1
                    break
        MinGradient = -0.04
        PBLH = 0
        for index, row in self.dataFrame.iterrows():
            if index >= k and index < (len(self.dataFrame)-1):
                grad = (self.dataFrame.loc[index,'ftr_theta'] - self.dataFrame.loc[index-1,'ftr_theta'])/(self.dataFrame.loc[index,'ftr_alt'] - self.dataFrame.loc[index-1,'ftr_alt'])
                grad_prev = (self.dataFrame.loc[index-1,'ftr_theta'] - self.dataFrame.loc[index-2,'ftr_theta'])/(self.dataFrame.loc[index-1,'ftr_alt'] - self.dataFrame.loc[index-2,'ftr_alt'])
                grad_next = (self.dataFrame.loc[index+1,'ftr_theta'] - self.dataFrame.loc[index,'ftr_theta'])/(self.dataFrame.loc[index+1,'ftr_alt'] - self.dataFrame.loc[index,'ftr_alt'])
                grad_next2 = (self.dataFrame.loc[index+2,'ftr_theta'] - self.dataFrame.loc[index+1,'ftr_theta'])/(self.dataFrame.loc[index+2,'ftr_alt'] - self.dataFrame.loc[index+1,'ftr_alt'])
                if (grad - grad_prev < MinGradient) or (grad_next < 0.004 and grad_next2 < 0.004):
                    PBLH = self.dataFrame.loc[index,'ftr_alt']
                    break
        return PBLH


    def computePBLHeightOnWRFModel(self):
        k = 2
        for index, row in self.dataFrame.iterrows():
           if (index == 0):
               theta0 = row['wrf_theta']
           else:
               if (row['wrf_theta'] - theta0 >= 0.5):
                    k = index
                    break
        MinGradient = 0.004
        PBLH = 0
        for index, row in self.dataFrame.iterrows():
            if index >= k and index < (len(self.dataFrame)-1):
                gradient = (self.dataFrame.loc[index+1,'wrf_theta'] - self.dataFrame.loc[index,'wrf_theta'])/(self.dataFrame.loc[index+1,'ftr_alt'] - self.dataFrame.loc[index,'ftr_alt'])
                if gradient >= MinGradient:
                    PBLH = self.dataFrame.loc[index,'ftr_alt']
                    break
        return PBLH

    def plotVerticalTemperatureProfile(self, heights, temperatures, wrf_temperatures, wrf_pblh, virtualPotentialTemperature , dpi, pblHfromRadiosondatge, outputFile, save):
        plt.figure(figsize=[8, 8])
        self.plotLine([self.minT, wrf_pblh], [self.maxT, wrf_pblh], "PBLH "+self.label)
        self.plotLine([self.minT, pblHfromRadiosondatge], [self.maxT, pblHfromRadiosondatge], "PBLH radisonde")
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
        eps = 0.622 #kg/kg
        e0 = 6.11 #hPa
        b = 17.2694
        es = e0 * math.exp(b*Temperature/(Temperature +237.29))
        qs = eps * es /(Pressure - es*(1-eps))
        q = (HR/100)*qs
        w = q/(1-q)
        return w*1000
    
    def bulkRichardsonNumber(self, thetav, thetav0, z, z0, wspeed):
        Ri = ((9.81/thetav0)*(thetav - thetav0)*(z-z0))/(wspeed**2)
        return Ri

def Models_comparison(we, variable_radio, variable_wrf, variable_label, dpi, outputFile):
    colors = ['b', 'r', 'g', 'm', 'tab:brown', 'c','tab:orange', 'tab:olive']
    linestyles = ['-', '-', '-.', '-.', '-', '-', '-.', '-.']
    matplotlib.rc('xtick', labelsize=14) 
    matplotlib.rc('ytick', labelsize=14) 
    fig, ax = plt.subplots(figsize=[7, 7]) #best is 15,7 for legend
    ax.plot(we[0].dataFrame[variable_radio], we[0].dataFrame['ftr_alt'],'black',linestyle='-', markersize=2, label='Radiosonde')
    for i, evaluation in enumerate(we):
        ax.plot(evaluation.dataFrame[variable_wrf], evaluation.dataFrame['ftr_alt'], color = colors[i],marker= 'o', linestyle=linestyles[i],linewidth=1.5, markersize=3,markevery=10, label=evaluation.label)
    ax.set_ylim(0,np.max([we[0].maxHeight,we[0].pblHfromRadiosondatge+100, we[0].pblHfromWRF+100]))
    ax.set_xlim(we[0].minT,we[0].maxT)
    #plt.legend(fontsize='x-large', ncol = 9, bbox_to_anchor=(0.5, 1.1), loc='upper center')
    ax.set_ylabel('Height (m)', fontsize=18)
    ax.set_xlabel(variable_label, fontsize=18)
    major_ticks = np.arange(0, 3000, 500)
    minor_ticks = np.arange(0, 2500, 100)
    ax.set_yticks(major_ticks)
    ax.set_yticks(minor_ticks, minor=True)
    major_ticks = np.arange(298, 320, 2)
    ax.set_xticks(major_ticks)
    #ax.grid(which='both', axis='y')
    plt.tight_layout()
    plt.savefig(outputFile, dpi=dpi)
    plt.close()
    
    
        
    
