#libraries importing

import numpy as np
import pandas
import matplotlib.pyplot as plt
import seaborn as sns

#importing the libraries  for clustering and curve fit

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from scipy.optimize import curve_fit
from itertools import cycle, islice
from pandas.plotting import parallel_coordinates

#importing the data
data = pandas.read_csv('API_19_DS2_en_csv_v2_4700503.csv', skiprows=4)
data


data=data.drop(data[['Unnamed: 66']],axis=1)
data


# ## Transpose

'''This  function is created to transpose the value of the variable in which  dataset is stored. For transposing the dataset, here .T is  used and for reading  the csv files pandas used here. '''
#transposig the data
def transpose():
    dataTrans    = pandas.read_csv('API_19_DS2_en_csv_v2_4700503.csv', skiprows=4, index_col=0, header=None).T
    dataTrans


# ## Handling Null Values

#checking for null values
data.isnull().sum()


#replacing null values with 0
data.fillna(value = 0, inplace = True)
data.isnull().sum()


data.describe()


#renaming the column
data.rename(columns = {'Country Name':'CountryName','Country Code':'CountryCode','Indicator Name':'IndicatorName','Indicator Code':'IndicatorCode'},inplace=True)


indi = data.groupby(['CountryName']).first()
indi1 = indi.head(10)
indi1


# # Urban Population

# creating  a new dataframe for graph
indi1 = indi1.reset_index()
indi1


#plotting the graph for urban population percentage over the years
X = ['Afghanistan','Africa Eastern and Southern','Africa Western and Central','Albania','Algeria','American Samoa', 'Andorra', 'Angola', 'Antigua and Barbuda', 'Arab World']
fig = plt.figure(figsize = (30,8))
X_axis = np.arange(len(X))
plt.bar(X_axis - 0.6, indi1['2016'], 0.4, label = '1960')
plt.bar(X_axis - 0.4, indi1['2017'], 0.4, label = '1980')
plt.bar(X_axis - 0.2, indi1['2018'], 0.4, label = '2000')
plt.bar(X_axis + 0.2, indi1['2019'], 0.4, label = '2010')
plt.bar(X_axis + 0.4, indi1['2020'], 0.4, label = '2020')
plt.bar(X_axis + 0.6, indi1['2021'], 0.4, label = '2021')

#giving the titles and labels
plt.xticks(X_axis, X, rotation = 'vertical')
plt.xlabel("Countries")
plt.ylabel("Urban Population Percentage")
plt.title("Urban Population percentage VS Country")
plt.legend()
plt.show()


data.head(44)


finalData = data.groupby(['CountryName','CountryCode','IndicatorName', 'IndicatorCode','1960', '1961', '1962', '1963', '1964', '1965', '1966', '1967', '1968', '1969', '1970', '1971', '1972', '1973', '1974', '1975', '1976', '1977', '1978', '1979', '1980', '1981', '1982', '1983', '1984', '1985', '1986', '1987', '1988', '1989', '1990', '1991', '1992', '1993', '1994', '1995', '1996', '1997', '1998', '1999', '2000', '2001', '2002', '2003', '2004', '2005', '2006', '2007', '2008', '2009', '2010', '2011', '2012', '2013', '2014', '2015', '2016', '2017', '2018', '2019', '2020', '2021']).size().reset_index()
finalData.head()


# # Foreign Direct Investment

# creating the datafrmae for the foreign investment 
finalData.set_index("IndicatorName", inplace=True)
value = "Foreign direct investment, net inflows (% of GDP)"
foreign = finalData.loc[value]
foreign = foreign.drop(["CountryCode","IndicatorCode"],axis =1)


foreign.head()


#reducing the dataset 
foreignPlot = foreign[::30]
foreignPlot = foreignPlot[["CountryName",'1990', '2000', '2010', '2012', '2014', '2018', '2021']]
foreignPlot = foreignPlot.reset_index(drop=True)
foreignPlot = foreignPlot.set_index(['CountryName'])
foreignPlot


#creating the heatmap
plt.figure(figsize = (14,9))
sns.heatmap(foreignPlot,annot=True,linewidth=2,cmap="Greens")
plt.title('Country VS Foreign Direct investment(% of GDP)')


# # CO2 emission


#taking out the CO2 emmision data
co2DataValue="CO2 emissions from liquid fuel consumption (kt)"
co2Data = finalData.loc[co2DataValue]
co2Data = co2Data.drop(["CountryCode","IndicatorCode"],axis = 1)
co2Data.head()


#Plotting the graph for co2 emmision
xyear=['1990', '2000', '2010', '2012', '2014', '2018', '2021']
co2Data = co2Data[:30]
co2Data.plot(x = "CountryName", y = xyear, figsize = (20,12))
plt.title("CO2 emmission by liquid fuel")
plt.ylabel("Amount of CO2 in power of e")


# # Growth  of Population


#  data  for  the growth  of  the population
growthValue = "Population growth (annual %)"
growth = finalData.loc[growthValue]
growth.head()


#plotting the grpah for  the population growth
growthYear = ['1960', '1980', '1970', '1990', '2000', '2010', '2020', '2021']
growth = growth[:10]
growth.plot(x = "CountryName", y = xyear, kind = "bar", figsize = (25,12))
plt.title("Growth of population yearly")
plt.ylabel("Growth in  percentage")


# # K means clustering

foreignPlot.head(10)


#scaling the features
X = StandardScaler().fit_transform(foreignPlot)
X


#applying the k means 
kmeans = KMeans(n_clusters=9)
model = kmeans.fit(X)
print("model\n", model)


centers = model.cluster_centers_
centers



#Creates a DataFrame with a column for the Cluster Number and makes it available for use.
def pd_centers(featuresUsed, centers):
    colNames = list(featuresUsed)
    colNames.append('prediction')
    # Zip with a column called 'prediction' (index)
    Z = [np.append(A, index) for index, A in enumerate(centers)]
    # Convert to pandas data frame for plotting
    P = pandas.DataFrame(Z, columns = colNames)
    P['prediction'] = P['prediction'].astype(int)
    return P



#creating functioins for  creating the parallel plots
def parallel_plot(data):
    my_colors = list(islice(cycle(['b', 'r', 'g', 'y', 'k']), None, len(data)))
    plt.figure(figsize = (15,8)).gca().axes.set_ylim([-3,+3])
    parallel_coordinates(data, 'prediction', color = my_colors, marker = 'o')



#defining the features
features = ['1990', '2000', '2010', '2012', '2014', '2018', '2021']



P = pd_centers(features, centers)
P


# ## for 1990


#parallel plot for the 1990
parallel_plot(P[P['1990'] < -0.5])


# ## For 2018


#function for the parallel plot
parallel_plot(P[P['2000'] > 0.5])


# # Model Fitting


''' This block went on to generate the plot for the curve fit depending on the query after first receiving the data, then developing the logistic function, and then fitting the data to the logistic function. 
Using the "err ranges" tool, I am currently calculating the minimum and maximum values of the confidence range as well as making a prediction about the values that will be present in ten years. 
After graphing the data, the logistic function that best matches the data, the confidence interval, and the predicted value, the procedure is finished by formatting the plot and plotting the graph.'''
#Giving the Data
growth_xdata = growth['1990']
growth_ydata = growth['2000']

#Logistic Function
def logistic(x,a,b,c):
    return c/(1 + np.exp(-a*(x-b)))

params, cov = curve_fit(logistic, growth_xdata, growth_ydata)

print("params", '\n', params)
#defining  the  err_ranges function for confindence range
def err_ranges(x, y, params):
    growth_y_fit = logistic(x, *params)
    growth_y_res = y - growth_y_fit
    sse = np.sum(growth_y_res**2)
    var = sse / (len(x) - len(params))
    
    ci = 1.96 # 95% confidence interval
    err = ci * np.sqrt(np.diag(cov)*var)
    return np.array([growth_y_fit - err, growth_y_fit + err])

# here the values are predicting
growth_x_pred = 20
growth_y_pred = logistic(growth_x_pred, *params)
plt.figure(figsize = (13,7))

#plotting the confidence ranges 
plt.plot(growth_xdata, growth_ydata, 'o', label = 'Data Points')
growth_x_fit = np.linspace(0,20,1000)
print(growth_x_fit.shape)
growth_y_fit = logistic(growth_x_fit, *params)
print(growth_y_fit.shape)
plt.plot(growth_x_fit, growth_y_fit, label = 'Best Fit')

err_range = err_ranges(growth_x_fit, growth_y_fit, params)
print(err_range)
plt.fill_between(growth_x_fit, err_range[0], err_range[1], color = 'gray', alpha = 0.5, label = 'Confidence Range')
plt.scatter(growth_x_pred, growth_y_pred, color = 'red', label = 'Predicted Value')

#pot  formation
plt.xlabel('1990')
plt.ylabel('2000')
plt.title('Confidence ranges')
plt.legend()
plt.show()

