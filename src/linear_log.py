# -*- coding: utf-8 -*-
"""
Created on Tue may 16 17:48:18 2020

@author: Priya Maini
"""
# =============================================================================
# New data with Linear regression
#considering aasian non asian as mapped 0 and 1 and passengers deleted. and latest dates
#LATEST DATA 
# =============================================================================
import numpy as np
import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from scipy.optimize import curve_fit
data_covid=pd.read_excel(r'D:/open source/NYcovid19/final data/final_paper_asian_nonAsian.xlsx')
data_covid.describe()



data_covid1=data_covid.drop(['country_name'],axis=1)
data_covid2=data_covid1.drop(['s_no'],axis=1)
data_covid3=data_covid2
#changing colums to date
data_covid3['peak_date'] = pd.to_datetime(data_covid3['peak_date'], format='%d/%m/%Y')
data_covid3['end_date_500'] = pd.to_datetime(data_covid3['end_date_500'], format='%d/%m/%Y')
data_covid3['start_date_500'] = pd.to_datetime(data_covid3['start_date_500'], format='%d/%m/%Y')
#taking time difference
data_covid3['peak_day']=(data_covid3['peak_date']-data_covid3['start_date_500']).dt.days
data_covid3['days_to_recover']=(data_covid3['end_date_500']-data_covid3['start_date_500']).dt.days
data_covid3['region']=data_covid3['region'].map({'asian': 0 ,'non asian': 1})
data_no_missingv=data_covid3.dropna(axis=0) 
sns.distplot(data_no_missingv['days_to_recover'])
#not normal data
# so lets use quarantile to make it normal 
data_new=data_no_missingv

plt.scatter(data_new['total_tests_done_per_million'],data_new['days_to_recover'])
plt.xlabel('total_tests_done_per_million',fontsize=20)
plt.ylabel('total days to recover',fontsize=20)

plt.scatter(data_new['positive_cases_count_per_milion'],data_new['days_to_recover'])
plt.xlabel('positive_cases_count_per_milion',fontsize=20)
plt.ylabel('total days to recover',fontsize=20)

data_new=data_new.drop(['end_cases_count'],axis=1)
data_new=data_new.drop(['total_tests_done_per_million'],axis=1)
data_new=data_new.drop(['positive_cases_count_per_milion'],axis=1)

#data_new[' passengers_carried_2018']=data_new[' passengers_carried_2018'].astype(int)

plt.scatter(data_new['region'],data_new['days_to_recover'])
plt.xlabel('Asian/Non-Asian',fontsize=20)
plt.ylabel('total days to recover',fontsize=20)

# checking multicollinearity
 #corr=data_new.corr()
 #display(corr)
 
data_new=data_new.drop(['start_date_500'],axis=1)
data_new=data_new.drop(['end_date_500'],axis=1)
data_new=data_new.drop(['peak_date'],axis=1)
data_new=data_new.drop(['region'],axis=1)
data_new=data_new.drop(['start_cases_count'],axis=1)
#data_new=data_new.drop(['start_cases_count'],axis=1) # just observations
#data_new=data_new.drop([' passengers_carried_2018'],axis=1)

# SPLIT INTO Training AND TESTING 
target=data_new['days_to_recover']
inputs=data_new.drop(['days_to_recover'],axis=1)

x1,x2,x3,x4 = inputs

#def func(X,a,b1,b2,b3,b4,c1,c2,c3,c4):
 #   x1,x2,x3,x4 = X
  #  return a+b1*X[x1]+b2*X[x2]+b3*X[x3]+b4*X[x4]+c1*np.log(X[x1])+c2*np.log(X[x2])+c3*np.log(X[x3])+c4*np.log(X[x4])

def func(X,a,c1,c2,c3,c4):
    x1,x2,x3,x4 = X
    return a+c1*np.log(X[x1])+c2*np.log(X[x2])+c3*np.log(X[x3])+c4*np.log(X[x4])
#x1,x2,x3,x4 = inputs

popt,pcov=curve_fit(func,inputs,target)
print(popt)
y_hat=popt[0]+np.log(inputs[x1])*popt[1]+np.log(inputs[x2])*popt[2]+np.log(inputs[x3])*popt[3]+np.log(inputs[x4])*popt[4]
data_new['predicted']=y_hat

plt.scatter(data_new['days_to_recover'], y_hat)
plt.xlabel('Actual values', size=18)
plt.ylabel('Predictions(y_hat)', size=18)
plt.ylim(10,100)
plt.show()


sns.distplot(data_new['days_to_recover']- y_hat)
plt.title("Residuals PDF",size=18)

from sklearn.metrics import r2_score
R2_score=r2_score(target,y_hat) #R^2: 0.517738774453164162
adjustedR2=1-((1-R2_score)*(33-1)/(33-4-1))
print(adjustedR2) #0.4488719792483282


columns=['peak_cases_count','total_density_per_sq_km','passengers_carried_2018','peak_day']
#valuesIndia = [78,130103,385,164035638]
valuesIndia = [130103,385,164035638,78] 
#valuesRussia = [89,11000,376,1640357.5] 
frame = { 'Columns': pd.Series(columns), 'Values': pd.Series(valuesIndia) }
CuntryDf = pd.DataFrame(frame).set_index('Columns').transpose() 
#CuntryDf.append([89,11000,376,1640357.5])

India = func(CuntryDf,popt[0],popt[1],popt[2],popt[3],popt[4])
print(India)
# 137th day since start of 500 cases i.e 1th april i.e August 16 2020 india will see 500 new cases only 

====================