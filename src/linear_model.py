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
data_covid=pd.read_excel(r'D:/priya maths/march2020/NYcovid19/final data/final_paper_asian_nonAsian.xlsx')
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
#data_new=data_new.drop(['start_cases_count'],axis=1) # just observations
#data_new=data_new.drop([' passengers_carried_2018'],axis=1)

# SPLIT INTO Training AND TESTING 
target=data_new['days_to_recover']
inputs=data_new.drop(['days_to_recover'],axis=1)


#df_normalised=pd.DataFrame(target)
#scaler=preprocessing.MinMaxScaler()

#y_scaled =scaler.fit_transform(df_normalised)
#df_norm=pd.DataFrame(y_scaled)
#x=data_with_dummies.iloc[:,1:5]
#scaler_x=StandardScaler()
#x=scaler_x.fit_transform(x)



#sns.distplot(y_test- y_hat)
#plt.title("Residuals PDF",size=18)

#sns.distplot(y_scaled)
#plt.title("y_scaled",size=18)



#plt.scatter(data_new['peak_day'],df_norm)
#plt.xlabel('peak_day',fontsize=20)
#plt.ylabel('total days to recover',fontsize=20)



from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(inputs,target,test_size=0.2,random_state=1)
from sklearn.preprocessing import StandardScaler
from sklearn import preprocessing
reg = LinearRegression()
reg.fit(x_train,y_train)
#GRAB The intercept and coefficient
intercept = reg.intercept_
coef = reg.coef_
for cf in zip(inputs.columns,coef):
    print("the coefficient for {} is {:.2}".format(cf[0],cf[1]))

y_hat = reg.predict(x_test)

plt.scatter(y_test, y_hat)
plt.xlabel('Targets(y_test)', size=18)
plt.ylabel('Predictions(y_hat)', size=18)
plt.ylim(20,90)
plt.show()
y_hat
# new data frame to store predictions and original y

df_pf =pd.DataFrame(y_hat, columns=['Predictions'])
df_pf
df_pf['Target'] =y_test.values
df_pf['residual']=df_pf['Target']-df_pf['Predictions']
df_pf['Difference'] = np.absolute(df_pf['residual']/df_pf['Target']*100)
df_pf.describe()
    # # another test is resisuals 
# =============================================================================
sns.distplot(y_test- y_hat)
plt.title("Residuals PDF",size=18)
# rqauare value
score=reg.score(x_train,y_train)
score # R2 is 0.69.66






# lets apply OLS model
x1=sm.add_constant(inputs)

model = sm.OLS(target,x1)
esst =model.fit()

esst.summary()
coeff=esst.params
print(coeff['Peak day '])
#creating the function with OLS Coeff
def funcEndDt(startCases,PeakCasesCount,totalDensity,isAsian,passengers,peak_day):
    return coeff[0]+startCases*coeff[1]+PeakCasesCount*coeff[2]+totalDensity*coeff[3]+isAsian*coeff[4]+passengers*coeff[5]+peak_day*coeff[6]

IndiaEndDate = funcEndDt(601,13103,385,0,164035638,78)     
print(IndiaEndDate)
## 104 days after start of 500 cases i.e 1st april i.e 
# 15th july 2020 india will see 500 new cases only 





====================