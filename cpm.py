import numpy as np
import seaborn as sns
import pandas as pd
#data fetching from csv crop dataset file
df=pd.read_csv('cropds.csv',na_values='=')
#structure,missing values,intial data of the dataframe
df.info()
df.isnull().sum()
df.head(6)
df.columns
data2=df.copy()
data2=data2.dropna()
data2.head()
data2.info()
data2.isnull().sum()
data2=data2.head(800000)       #limiting just incase for faster results
#month to season
str=data2["Date"][1]
str2=str.split('-')
Dict={1:"January",2:"February",3:"March",4:"April",5:"May",6:"June",7:"July",8:"August",9:"September",10:"October",11:"November",12:"December"}
month_column=[]
for rr in data2["Date"]:
    str=rr
    str2=str.split('-')
    month_column.append(Dict[int(str2[1])])
data2["month_column"]=month_column
data2["month_column"].unique()
season_names=[]
for tt in data2["month_column"]:
    if tt=="January" or tt=="February":
        season_names.append("winter")
    elif tt=="March" or tt=="April":
        season_names.append("spring")
    elif tt=="May" or tt=="June":
        season_names.append("summer")
    elif tt=="July" or tt=="August":
        season_names.append("monsoon")
    elif tt=="September" or tt=="October":
        season_names.append("autumn")
    elif tt=="November" or tt=="December":
        season_names.append("pre winter")
data2["season_names"]=season_names
data2.head()
#date->day of week
import pandas as pd
df=pd.Timestamp("2019-04-12")
day_of_week=[]
for rr in data2["Date"]:
    str=rr
    df=pd.Timestamp(rr)
    day=df.dayofweek
    day_of_week.append(day)
data2["day"]=day_of_week
data2.tail()
data2 = data2.drop('Date', axis=1)#date column dropped
data2.columns
data2=data2.head(100000)
import seaborn as sns
import matplotlib.pyplot as plt
#data cleaning
sns.boxplot(data2['Modal Price'])
plt.show()
data2.shape
data2['Modal Price']       #visually finding outliers
#inter quartile range (IQR) statistical data cleaning method
q1=np.percentile(data2['Modal Price'],25,interpolation="midpoint") #below 25%ile
q3=np.percentile(data2['Modal Price'],75,interpolation='midpoint')    #below 75%ile
iqr=q3-q1
upper=np.where(data2['Modal Price']>=(q3+1.5*iqr))   #values above upper bound=outlier
lower=np.where(data2['Modal Price']<=(q1-1.5*iqr))    #values below lower bound=outlier
data2.drop(upper[0],inplace=True)
data2.drop(lower[0],inplace=True)
sns.boxplot(data2['Modal Price'])   #post cleansing data
plt.show()
df=data2.copy()
data2.columns
#plotting bunch of relations graph
import plotly.express as px
import plotly.io as pio
pio.renderers.default = 'browser'
sns.relplot(data=df,x="State",y="Modal Price", hue="season_names",kind="line")
plt.show()
sns.relplot(data=df,x="District",y="Modal Price", hue="season_names",kind="line")
plt.show()
fig=px.bar(df,x="District",y="Modal Price",color="season_names",height=400)
fig.show()
sns.relplot(data=df,x="season_names",y="Modal Price",hue="season_names",kind="line")
plt.show()
sns.relplot(data=df,x="day", y="Modal Price",hue="season_names",kind="line")
plt.show()
data2.columns
sns.relplot(data=df,x="Market",y="Modal Price",hue="season_names",kind="line")
plt.show()
sns.relplot(data=df,x="month_column",y="Modal Price",hue="season_names",kind="line")
plt.show()
#numerazing non number data types for Model training
dist=(data2['Commodity_name'])
distset=set(dist)
dd=list(distset)
dictofwords={dd[i]:i for i in range(0,len(dd))}
data2['Commodity_name']=data2['Commodity_name'].map(dictofwords)
dist=(data2['State'])
distset=set(dist)
dd=list(distset)
dictofwords={dd[i]:i for i in range(0,len(dd))}
data2['State']=data2['State'].map(dictofwords)
dist=(data2['District'])
distset=set(dist)
dd=list(distset)
dictofwords={dd[i]:i for i in range(0,len(dd))}
data2['District']=data2['District'].map(dictofwords)
dist=(data2['Market'])
distset=set(dist)
dd=list(distset)
dictofwords={dd[i]:i for i in range(0,len(dd))}
data2['Market']=data2['Market'].map(dictofwords)
dist=(data2['month_column'])
distset=set(dist)
dd=list(distset)
dictofwords={dd[i]:i for i in range(0,len(dd))}
data2['month_column']=data2['month_column'].map(dictofwords)
dist=(data2['season_names'])
distset=set(dist)
dd=list(distset)
dictofwords={dd[i]:i for i in range(0,len(dd))}
data2['season_names']=data2['season_names'].map(dictofwords)
data2.info()
dataplot=sns.heatmap(data2.corr(),cmap="YlGnBu",annot=True)  #heat map relates each data label with others to check dependencies
plt.show()
data2.columns
features=data2[['Commodity_name','State','District','Market','month_column','season_names','day']]  #features
labels=data2['Modal Price'] #target
from sklearn.model_selection import train_test_split
Xtrain,Xtest,Ytrain,Ytest=train_test_split(features,labels,test_size=0.2,random_state=2)
from sklearn.metrics import classification_report
from sklearn import metrics
from sklearn import tree
from sklearn.ensemble import RandomForestRegressor
from sklearn.datasets import make_regression
regr=RandomForestRegressor(max_depth=1000,random_state=0)
regr.fit(Xtrain,Ytrain)
Xtest[0:1]
y_pred=regr.predict(Xtest)
from sklearn.metrics import r2_score
print(r2_score(Ytest,y_pred))  #accuracy equivalent ie how much of test relates with train
print(y_pred)
data2.columns
print(Xtest[0:1]) #fetches an input cause still we are using numerized features as inputs and not string

user_input=[[166,24,155,954,1,0,6]]  #numerized input of features
print(regr.predict(user_input)) #predicted modal price




