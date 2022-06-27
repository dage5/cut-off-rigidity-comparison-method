import data as data
import numpy as np
import math, sys

import matplotlib.pyplot as plt
import astropy.time as atime
import numpy as np
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from cartopy.mpl.ticker import (LongitudeFormatter, LatitudeFormatter,
                                LatitudeLocator)
import matplotlib.ticker as mticker
from cartopy.util import add_cyclic_point
import copy
from scipy import stats

np.set_printoptions(precision=2)
np.set_printoptions(suppress=True)

def loadtext_without_columns(filePath, skipcols=[], delimiter=" "):
    with open(filePath) as f:
       n_cols = len(f.readline().split(delimiter))

    #define a range from 0 to n_cols
    usecols = np.arange(0, n_cols)
    #remove the indices found in skipcols
    usecols = set(usecols) - set(skipcols)
    #sort the new indices in ascending order
    usecols = sorted(usecols)
    #print((usecols))
    #load the file and retain indices found in usecols
    data = np.loadtxt(filePath, delimiter = delimiter, usecols = usecols)
    return data

def plot(values, filename, title = "", colorBarText = "Rigidity [GV]"):
	startLat, endLat, startLon, endLon, stepLat, stepLon, latLen, lonLen, latNum, lonNum = (-90, 90, 0, 360, 5, 15, 180, 360, 1, 1)

	if(stepLat != 0):
		latNum = int(latLen/stepLat)+1
	if(stepLon != 0):
		lonNum = int(lonLen/stepLon)+1
	if( (endLon == 0. and startLon > 0)):
		endLon = 360.

	latGrid = np.linspace(startLat, endLat, latNum)
	lonGrid = np.linspace(startLon, endLon, lonNum)
	stLat, enLat, stLon, enLon, epLat, epLon = list(map(str, [startLat, endLat, startLon, endLon, stepLat, stepLon]))

	global_transform_instance = ccrs.PlateCarree()
	X, Y = np.meshgrid(lonGrid, latGrid)
	if (endLon==360 and startLon==0):#https://scitools.org.uk/cartopy/docs/v0.15/cartopy/util/util.html
		lons = np.arange(0, 360, stepLon)
		valuesR = np.reshape(values, (X.shape[0], X.shape[1]-1 ))
		valuesR, cyclic_lons = add_cyclic_point(valuesR, coord=lons)
		#print(valuesR_old[:-1,:])
		"""for row in valuesR:
			print(row[10:14])"""
		#print(valuesR)
		"""print(cyclic_lons[0:2:1], cyclic_lons[-2:])
			
		
		print(lonGrid.shape, latGrid.shape)
		print(X.shape,Y.shape)"""
	else:	
		valuesR = np.reshape(values, X.shape)
		
	"""print(lonGrid)
	lonGrid = np.roll(lonGrid, 10)
	print(lonGrid)"""
	
	fig = plt.figure(tight_layout = True, dpi = 100, figsize=[15,10])
	#ax = plt.gca()
	ax = fig.add_axes(rect = [1,1,1,1.5], projection=global_transform_instance,ymargin = 1)
	ax.set_title(title, fontsize=15, wrap = True)
	
	if(latLen > 45 or lonLen > 90):
		ax.set_global()
	else:
		(lonSt, latSt) = (stepLat, stepLon)
		(sLon, eLon, sLat, eLat) = (startLon-10, endLon+10, startLat-10, endLat+10)
		ax.set_extent([sLon, eLon, sLat, eLat], crs=global_transform_instance)
	
	ax.coastlines(resolution='110m')#https://scitools.org.uk/cartopy/docs/v0.13/matplotlib/geoaxes.html
	gl = ax.gridlines(crs=global_transform_instance, draw_labels=True, alpha = 0.75, linestyle='--') #https://scitools.org.uk/cartopy/docs/v0.13/matplotlib/gridliner.html
	gl.top_labels = None
	gl.right_labels = None
	gl.xformatter = LongitudeFormatter()
	gl.yformatter = LatitudeFormatter()
	gl.xlocator = mticker.LinearLocator(numticks=9) #https://matplotlib.org/3.2.1/api/ticker_api.html
	gl.ylocator = mticker.LinearLocator(numticks=7)

#	levels = 8#np.linspace(values.min(), values.max(), coeff)
	if (latLen != 0 and lonLen != 0):
		#cs = plt.contourf(X,Y, valuesR, 40, transform=global_transform_instance, cmap = "YlGn")
		#plt.contour(X,Y, valuesR,  transform=global_transform_instance, colors= "black", alpha = 0.2, width = 0.1)
		cs = plt.pcolor( lonGrid, latGrid, valuesR, transform=global_transform_instance, shading='auto')#, shading='nearest', edgecolors=None)
		"""if(True):
			valuesR[valuesR != 0] = np.nan
			plt.pcolor( lonGrid, latGrid, valuesR, transform=global_transform_instance, shading='auto', alpha = 0, edgecolors = 'black')"""

	cbar = fig.colorbar(cs, ax=ax, orientation="horizontal", pad=0.02, aspect = 60, shrink = 1)
	cbar.ax.get_xaxis().labelpad = 10
	cbar.ax.tick_params(labelsize=15)
	cbar.ax.set_xlabel(colorBarText, labelpad=0.02, rotation=0, fontsize = 15)
	plt.savefig(filename, bbox_inches='tight',pad_inches=0.21)
	plt.close()
	return

def getAreaWhereNAN(data):
	data = copy.deepcopy(np.reshape(data, (37, 24)) )
	#print(data)
	R = 6371.0
	area = 0
	infArea = 0
	areaEarth = (4 * math.pi * R*R)
	for i in range(0,37):
		for j in range(0, 24):
			firstLat = 90 - i * 5 - 2.5
			secondLat = 90 - i * 5 + 2.5
			if firstLat < -90:
				firstLat = -90
			if secondLat > 90:
				secondLat = 90
			if np.isnan(data[i][j]):#calculates area where the difference is bigger than the treshold, areas where zero in atleast one dataset occured are excluded
				A = (math.pi/180) * (R*R)*abs(np.sin(math.radians(firstLat)) - (np.sin(math.radians(secondLat) ) ) ) * (15)
				area = area + abs(A)
			elif np.isinf(data[i][j]):#calculates area where locations were marked infinite, e.g. locations where atleast in one of the original datasets, value was zero
				A = (math.pi/180) * (R*R)*abs(np.sin(math.radians(firstLat)) - (np.sin(math.radians(secondLat) ) ) ) * (15)
				infArea = infArea + abs(A)
	badPoints = 0
	for i in range(0,37):
		for j in range(0, 24):
			if np.isnan(data[i][j]):
				badPoints = badPoints + 1
	#print("{:.2f} ".format(area), "{:.2f} ".format(areaEarth), "{:.2f} ".format(100 - area / (areaEarth/100) ), "{:.2f} ".format(100 - area / ((areaEarth-infArea)/100) ), "{:.2f} ".format(infArea) )
	return "{:.2f} ".format(100 - 100*area/areaEarth ) + "{:.2f} ".format(100 - 100 * area/(areaEarth-infArea) ) + "{:.2f} ".format(area) + "{:.2f} ".format(areaEarth) + "{:.2f} ".format(infArea)

TESTED_VALUE = 0 #float(sys.argv[1])

def rollAll(data, rangeMax, shift):
	for i in range(0,rangeMax):
		data[i] = np.roll(data[i], shift)
def prepData(theirData2, ourData2, filename, divide, threshold = 100, title = "", treatZeroAsNoDiff = False, shift = -11):
	theirData = copy.deepcopy(theirData2)
	ourData = copy.deepcopy(ourData2)
	
	sameDim = theirData.shape == ourData.shape
	
	rollAll(theirData, 37, shift)
	if(sameDim):
		rollAll(ourData, 37, shift)

	#!!!Najst kde su min a max z datasetu a aka je hodnota odrezavacej rigidity!!!
	diff_list = []
	minL = {"value":1000000, "lat":0, "lon":0, "L":0, "R":0}
	maxL = {"value":-1000, "lat":0, "lon":0, "L":0, "R":0}
#	minR = {"value":10, "lat":0, "lon":0, "L":0, "R":0}
#	maxR = {"value":0, "lat":0, "lon":0, "L":0, "R":0}
	for i in range(0,37):
		for j in range(0, 24):
			theirCell = theirData[36 - i][j]
			if(sameDim):
				ourCell = ourData[36 - i][j]
			else:
				ourCell = ourData[i * 5][j * 15]
			computedVal = 0
			if divide:
				if (theirCell <= TESTED_VALUE or ourCell <= TESTED_VALUE) and treatZeroAsNoDiff:
					computedVal = np.inf
				else:
					computedVal = (theirCell / ourCell - 1 ) * 100
			else:
				computedVal = theirCell - ourCell
			diff_list.append(computedVal)
			if (not divide):
				#print(abs(computedVal))
				if (computedVal) < (minL["value"]):#(not np.isinf(computedVal)) and  
					minL["value"] = computedVal
					minL["lat"] = i * 5 - 90
					minL["lon"] = j * 15
					minL["L"] = theirCell
					minL["R"] = ourCell
				if (computedVal) > (maxL["value"]):
					maxL["value"] = computedVal
					maxL["lat"] = i * 5 - 90
					maxL["lon"] = j * 15
					maxL["L"] = theirCell
					maxL["R"] = ourCell
	if (not divide):
		print(minL)
		print(maxL)
	diff_list = (np.array(diff_list))
	retStr = ""
	if divide:
		###
		data_ = diff_list.flatten()
		data_[ np.isnan(data_)] = 0 #handle division by zero, only applies when treatZeroAsNoDiff = False, when True, this does nothing
		data_ = data_[~ np.isinf(data_)]#remove ratio where atleast one of values from either dataset was zero
		retStr = retStr + "{:.2f} ".format(np.median(np.abs(data_))) + "{:.2f} ".format(np.average(np.abs(data_)))
		###
		diff_list[np.isnan(diff_list)] = 0 #handle division by zero, only applies when treatZeroAsNoDiff = False, when True, this does nothing
		diff_list[ (abs(diff_list) > threshold) & (~ np.isinf(diff_list)) ] = np.nan #include locations where diff is bigger than treshold but it is not infinite, infinites are preserved and are excluded
		retStr = retStr + getAreaWhereNAN(diff_list) + "\n"
		plot(diff_list, filename + "_ratio", title, "Relative difference [%]")
		print(filename + "_ratio")
	else:
		diff_list[abs(diff_list) > threshold] = np.nan
		plot(diff_list, filename + "_diff", title)
		print(filename + "_diff")
	return retStr
	
if __name__ == '__main__':
	igrf_1_1_2010 = loadtext_without_columns("2010-001-00-00-00.000_-90.0_90.0_0.0_360.0_1.0_1.0_R.png.txt", [360])
	igrf_1_1_2015 = loadtext_without_columns("2015-001-00-00-00.000_-90.0_90.0_0.0_360.0_1.0_1.0_R.png.txt", [360])
	
	table = ""
	fns = ["2015_smartshea_gerontidou", "2015_gerontidou_cor", "2015_smartshea_cor"]
	desc_abs = "Difference between "
	desc_rel = "Relative difference between "
	ref = ["Smart & Shea (2019) and Gerontidou et al. (2021)", "Gerontidou et al. (2021) and COR", "Smart & Shea (2019) and COR"]
	desc_sffx = " for 2015 epoch"
	#data.smartshea_2015
	table = table + prepData(data.smartshea_2015_2, data.gerontidou_2015, fns[0], False, 100, desc_abs+ref[0]+desc_sffx)
	table = table + prepData(data.gerontidou_2015, igrf_1_1_2015, fns[1], False, 100, desc_abs+ref[1]+desc_sffx)
	table = table + prepData(data.smartshea_2015_2 , igrf_1_1_2015, fns[2], False, 100, desc_abs+ref[2]+desc_sffx)
	
	
	table = table + prepData(data.smartshea_2015_2, data.gerontidou_2015, fns[0], True, 5, desc_rel+ref[0]+desc_sffx)
	table = table + prepData(data.gerontidou_2015, igrf_1_1_2015, fns[1], True, 5,  desc_rel+ref[1]+desc_sffx)
	table = table + prepData(data.smartshea_2015_2 , igrf_1_1_2015, fns[2], True, 5,  desc_rel+ref[2]+desc_sffx)

	table = table + "------------------------------------------\nValues smaller than "+str(TESTED_VALUE)+" in either dataset are ignored:\n"
	
	suffix = "_special_zero"
	table = table + "|"
	table = table + prepData(data.smartshea_2015_2, data.gerontidou_2015, fns[0] + suffix, True, 5,  desc_rel+ref[0]+desc_sffx, True)
	table = table + "|"
	table = table + prepData(data.gerontidou_2015, igrf_1_1_2015, fns[1] + suffix, True, 5,  desc_rel+ref[1]+desc_sffx, True)
	table = table + "|"
	table = table + prepData(data.smartshea_2015_2 , igrf_1_1_2015, fns[2] + suffix, True, 5,  desc_rel+ref[2]+desc_sffx, True)
	
	
	print("median;", "average;", "% A <5% diff;",  "% A <5% diff - 0 ignored;", "A;", "Earth A;",  "ignored A;")
	print(table)
	
	





