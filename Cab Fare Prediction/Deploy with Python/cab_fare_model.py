# Importing the libraries
#Importing required libraries
#import os #getting access to input files
import pandas as pd # Importing pandas for performing EDA
import numpy as np  # Importing numpy for Linear Algebric operations
import matplotlib.pyplot as plt # Importing for Data Visualization
import seaborn as sns # Importing for Data Visualization
from collections import Counter 
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression #ML algorithm
from sklearn.model_selection import train_test_split #splitting dataset
from sklearn.metrics import mean_squared_error


#%matplotlib inline
import pickle



train_cab  = pd.read_csv("train_cab.csv")

# lets convert our fare_amount variable from object to numeric data type
####EDA
train_cab['fare_amount'] = pd.to_numeric(train_cab['fare_amount'], errors = "coerce") 
# when we tried convert pickup_datetime variable to date format it was throwing error coz of a starnge value in the variable
# So first treat it as NA and drop 

train_cab.loc[train_cab['pickup_datetime'] == '43' ,'pickup_datetime'] = np.nan  

train_cab = train_cab.drop(train_cab[train_cab['pickup_datetime'].isnull()].index, axis = 0)
# Here pickup_datetime variable is in object so we need to change its data type to datetime
train_cab['pickup_datetime'] =  pd.to_datetime(train_cab['pickup_datetime'], format='%Y-%m-%d %H:%M:%S UTC')
### we will saperate the Pickup_datetime column into separate field like year, month, day of the week, etc

train_cab['year'] = train_cab['pickup_datetime'].dt.year
train_cab['Month'] = train_cab['pickup_datetime'].dt.month
train_cab['Date'] = train_cab['pickup_datetime'].dt.day
train_cab['Day'] = train_cab['pickup_datetime'].dt.dayofweek
train_cab['Hour'] = train_cab['pickup_datetime'].dt.hour

###MISSING VALUES ANALYSIS
#first we removing passanger_count missing values rows
train_cab = train_cab.drop(train_cab[train_cab['passenger_count'].isnull()].index, axis=0)

#second we will removing fare_amount missing values
# eliminating rows for which value of "fare_amount" is missing
train_cab = train_cab.drop(train_cab[train_cab['fare_amount'].isnull()].index, axis=0)

###OUTLIER ANALYSIS
#here passenger count has totally 77 outliers where 58 variables are 0 and 19 variables are above 6.
#there is no use of having these data.hence we drop it
train_cab = train_cab.drop(train_cab[train_cab['passenger_count'] > 7].index, axis=0)
train_cab = train_cab.drop(train_cab[train_cab['passenger_count'] < 1].index, axis=0)

#from the above inferences,we can say that the fare amount for a cab cant be in negative as well as it cannot exceed 454
#above and below the values will be considered as outliers and drop it.
train_cab = train_cab.drop(train_cab[train_cab['fare_amount'] > 454].index, axis=0)
train_cab = train_cab.drop(train_cab[train_cab['fare_amount'] < 1].index, axis=0)

train_cab = train_cab.drop((train_cab[train_cab['pickup_latitude']<-90]).index, axis=0)
train_cab = train_cab.drop((train_cab[train_cab['pickup_latitude']>90]).index, axis=0)

#SOME PREPROCESING OPERATIONS
#As we know that we have given pickup longitute and latitude values and same for drop. 
#So we need to calculate the distance Using the haversine formula and we will create a new variable called distance
from math import radians, cos, sin, asin, sqrt

def haversine(a):
    lon1=a[0]
    lat1=a[1]
    lon2=a[2]
    lat2=a[3]
    """
    Calculate the great circle distance between two points 
    on the earth (specified in decimal degrees)
    """
    # convert decimal degrees to radians 
    lon1, lat1, lon2, lat2 = map(radians, [lon1, lat1, lon2, lat2])
    
    # haversine formula 
    dlon = lon2 - lon1 
    dlat = lat2 - lat1 
    a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
    c =  2 * asin(sqrt(a))
    # Radius of earth in kilometers is 6371
    km = 6371* c
    return km

#Applying haversine formula to trained data
train_cab['distance'] = train_cab[['pickup_longitude','pickup_latitude','dropoff_longitude','dropoff_latitude']].apply(haversine,axis=1)
#Now we will remove the rows whose distance value is zero and more than 130
#For taining dataset
train_cab = train_cab.drop(train_cab[train_cab['distance']== 0].index, axis=0)
train_cab = train_cab.drop(train_cab[train_cab['distance'] > 130 ].index, axis=0)

#FEATURE SELECTION
drop = ['pickup_datetime', 'pickup_longitude', 'pickup_latitude','dropoff_longitude', 'dropoff_latitude']
train_cab = train_cab.drop(drop, axis = 1)

#As in test daatset passanger_count data type is int64 and in train dataset we found passanger_cout type as float64
#so for avoiding any on mismatch further oprations we conver it into int64
train_cab['passenger_count'] = train_cab['passenger_count'].astype('int64')

##Correlation analysis AND ANOVA TESTING WE WILL REMOVED DATE AND DAY VARIABLE FROM OUR DATASET
train_cab = train_cab.drop('Date', axis = 1)
train_cab = train_cab.drop('Day', axis = 1)

#FETURE SCALLING
#since skewness of target variable is high, apply log transform to reduce the skewness-
train_cab['fare_amount'] = np.log1p(train_cab['fare_amount'])

#since skewness of distance variable is high, apply log transform to reduce the skewness-
train_cab['distance'] = np.log1p(train_cab['distance'])
#train_cab['distance'] = train_cab['distance'].astype('int64')

print(train_cab.dtypes)


#DATA MODELING
x= train_cab.drop(['fare_amount'],axis=1)
y= train_cab['fare_amount']

# Now Split the data into train and test using train_test_split function
X_train, X_test, y_train, y_test = train_test_split( x, y, test_size=0.2, random_state=101)
#Since we have a very small dataset, we will train our model with all availabe data.

fit_RF = RandomForestRegressor(n_estimators = 200).fit(X_train,y_train)

# Saving model to disk
pickle.dump(fit_RF, open('model.pkl','wb'))

# Loading model to compare the results
model = pickle.load(open('model.pkl','rb'))
print(model.predict([[1, 2009, 6, 17, 0.708]]))