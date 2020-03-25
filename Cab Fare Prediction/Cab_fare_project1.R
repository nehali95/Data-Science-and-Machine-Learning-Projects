# Clean the environment
rm(list=ls())

# Set working directory
setwd("D:/EdwisorDS/Assignment/Cab_Fare_Prediction_Project1")

# Load required Libraries for analysis  ----------------------------------
x = c("ggplot2", "corrgram","dplyr", "caret", "randomForest", "e1071", "rpart",'DataCombine')
#DMwR,unbalanced,C50 ,dummies,Information,MASS,gbm,ROSE,sampling
#install.packages(x)
#NOW TO LOAD MULTIPLE PACKAGES AT TIME
lapply(x, require, character.only = TRUE)
rm(x)
# Load the data -----------------------------------------------------------
Train_Cab = read.csv("train_cab.csv")

#Let's Understand dataset
# Check class of the data
cat("type of Train dataset",class(Train_Cab))

#Check the dimensions(no of rows and no of columns)
cat("Shape of Train Dataset",dim(Train_Cab))


#Check top(first) rows of dataset 
head(Train_Cab)


#Check structure of dataset(data structure of each variable)
str(Train_Cab)

#Check summary of dataset 
summary(Train_Cab)


#-------------------------------------------------------------------------------------------------------------------------------------------
                    ##### DATA PREPROCESSING ON TRAIN AND TEST DATA SIMULTANEOUSLY #####
#-------------------------------------------------------------------------------------------------------------------------------------------
 ###------------------ 1] Explolatory Data Analytics ------------------------------- #### 

#As we have seen while finding data type of train cab dataset that fare_amount variable showing as a object,
#due to tat while finding description of that data set it will not able to do an arithmatic operation on fare_amount.
#Therefore futher process we have to convert fare_amount to numeric.

Train_Cab$fare_amount = as.numeric(as.character(Train_Cab$fare_amount))
str(Train_Cab$fare_amount)  

# We need to change pickup_datetime from factor to datetime
# But first, let's replace UTC in pickup_datetime variable with ''(space)

Train_Cab$pickup_datetime[Train_Cab$pickup_datetime== '43' ]= NA
Missing_val = sum(is.na(Train_Cab$pickup_datetime))
Missing_val
#na.omit(Train_Cab$pickup_datetime)

Train_Cab$pickup_datetime = gsub('// UTC','',Train_Cab$pickup_datetime)

 #### For Train dataset ###
# Now convert variable pickup_dattime to date time format by creating
# new variable with name Date 

Train_Cab$date = as.Date(Train_Cab$pickup_datetime)
# Lets split this new variable Date into year,month,weekday 
# Extract the year
Train_Cab$year = substr(as.character(Train_Cab$date),1,4)

# Extract the month
Train_Cab$month =substr(as.character(Train_Cab$date),6,7)

# Extract the weekday 
Train_Cab$day = weekdays(as.POSIXct(Train_Cab$date),abbreviate = F)

# Extract the date 
Train_Cab$date = substr(as.character(Train_Cab$date),9,10)

# Extract the time / hour
Train_Cab$hour = substr(as.factor(Train_Cab$pickup_datetime),12,13)


str(Train_Cab)


#Recheck dimensions of test and train data
dim(Train_Cab)


###------------------------- 2] MISSING VALUE ANALYSIS---------------------------------###

# Total number of missing values present in whole datset 
Missing_val = sum(is.na(Train_Cab)) 
Missing_val #86


#here we are removing all missing values from train datasets
Train_Cab = na.omit(Train_Cab)
Missing_val = sum(is.na(Train_Cab)) 
Missing_val


#lets check the dimension after removing missing values
dim(Train_Cab)



###-------------------------- 3] OUTLIER ANALYSIS ---------------------------------###

#########For passanger_count variable###########
summary(Train_Cab$passenger_count)

#Here we can seen maximum passenger count is 5345 which not possible in cab service. 
#And also we can seen that minimus passenger count is 0 which is also not valid for count.
#So we remove manually remove this outlier from passenger variable by considering maximum passenger count is 6 and minimum is not less than 1.
Train_Cab[Train_Cab$passenger_count < 1,"passenger_count"] = NA
#Train_Cab[Train_Cab$passenger_count > 6,"passenger_count"] <- NA
# When I applying above statements it was giving me error 
#ERROR: missing values are not allowed in subscripted assignments of data frames
#sO WE GO BY BELOW SOLUTION
Train_Cab$passenger_count<- ifelse(Train_Cab$passenger_count > 6, NA,Train_Cab$passenger_count )

summary(Train_Cab$passenger_count)

###########For fare_amount count variable###########
summary(Train_Cab$fare_amount)
sort(Train_Cab$fare_amount,decreasing = TRUE)
#In order of fare amount we can see that there is a huge difference in 1st 2nd and 3rd position 
#so we will remove the rows having fare amounting more that 454 as considering them as outliers

#And also, from the above description,we can say that the fare amount for a cab cant be in negative 
#as well as it cannot exceed 500 above and below the values will be considered as outliers and drop it.
Train_Cab$fare_amount[Train_Cab$fare_amount < 1] = NA
Train_Cab$fare_amount[Train_Cab$fare_amount >454] = NA

summary(Train_Cab$fare_amount)

#REMOVING ALL NAN VALUES AFTER OVERCOMING OUTLIERS OF FARE_AMOUNT AND PASSANGER_COUNT
sum(is.na(Train_Cab)) #77+
Train_Cab = na.omit(Train_Cab)

#recheck dimension of Train dataset
dim(Train_Cab)

############# For variable pickup_latitude , dropoff_latitude and dropoff_longitude , dropoff_longitude #############

#There sholud me Latitudes range from -90 to 90 so we can remove below -90 and above +90 observation by considering its an outlier.
#There sholud me Longitude range from -180 to 180 so we can remove below -180 and above +180 observation by considering its an outlier.

Train_Cab[Train_Cab$pickup_longitude < -180 ,"pickup_longitude"] = NA
Train_Cab[Train_Cab$pickup_longitude > 180 ,"pickup_longitude"] = NA
sum(is.na(Train_Cab))

Train_Cab[Train_Cab$pickup_latitude < -90,"pickup_latitude"] = NA
Train_Cab[Train_Cab$pickup_latitude > 90,"pickup_latitude"] = NA
sum(is.na(Train_Cab))#1

Train_Cab[Train_Cab$dropoff_longitude < -180,"dropoff_longitude"] = NA
Train_Cab[Train_Cab$dropoff_longitude > 180,"dropoff_longitude"] = NA
sum(is.na(Train_Cab))# 1 found

Train_Cab$dropoff_latitude<- ifelse(Train_Cab$dropoff_latitude < -90, NA,Train_Cab$dropoff_latitude )
Train_Cab$dropoff_latitude<- ifelse(Train_Cab$dropoff_latitude > 90, NA,Train_Cab$dropoff_latitude )
sum(is.na(Train_Cab))#1 found

Train_Cab = na.omit(Train_Cab)
sum(is.na(Train_Cab))

#after missing value analysis structure of train will be
dim(Train_Cab)

#As per above condition there are no oulier found in test datset
#so we will not apply above techniques to test dataset

----------------------------------------------------------------------------------------------------------------------
  ###################### Some preprocessing operation before moving towards further process.#######################
----------------------------------------------------------------------------------------------------------------------
#Calculating distance based on given varibales:
#As we know that we have given pickup longitute and latitude values and same for drop. 
#So we need to calculate the distance Using the haversine formula and we will create a new variable called distance




#y = install.packages(c('purrr','geosphere','rlist'))
library(purrr) #for functional programing map()
library(geosphere) #Spherical trigonometry for geographic applications. That is, compute distances and related measures for angular (longitude/latitude) locations
library(rlist) #Provides a set of functions for data manipulation with list objects, including mapping, filtering, grouping, sorting, updating, searching, and other useful functions. Most functions are designed to be pipeline friendly so that data processing with lists can be chained.
get_geo_distance = function(long1, lat1, long2, lat2) {
  loadNamespace("purrr")
  loadNamespace("geosphere")
  longlat1 = purrr::map2(long1, lat1, function(x,y) c(x,y))
  longlat2 = purrr::map2(long2, lat2, function(x,y) c(x,y))
  distance_list = purrr::map2(longlat1, longlat2, function(x,y) geosphere::distHaversine(x, y))
  distance_m = list.extract(distance_list, position = 1)
  #if (units == "km") {
  distance = distance_m / 1000.0;
  
  distance
}


# Applying distance formula for train data
for(i in (1:nrow(Train_Cab)))
{
  Train_Cab$distance[i]= get_geo_distance(Train_Cab$pickup_longitude[i],Train_Cab$pickup_latitude[i],Train_Cab$dropoff_longitude[i],Train_Cab$dropoff_latitude[i])
}

# Lets check data after distance variable creation
head(Train_Cab)


# Lets check whether distance variables has any outlier 
sort(Train_Cab$distance,decreasing = TRUE)
summary(Train_Cab$distance)
dim(Train_Cab)
#As we can see that top 23 values in the distance variables are very high i.e more than 8000 Kms 
#distance they have travelled Also just after 23rd value from the top, 
#the distance goes down to 129, which means these values are showing some outliers 
#We need to remove these values. Also values could not be 0
Train_Cab[Train_Cab$distance == 0,'distance'] <- NA
Train_Cab$distance<- ifelse(Train_Cab$distance > 130, NA,Train_Cab$distance )
sum(is.na(Train_Cab))# 478 found
Train_Cab = na.omit(Train_Cab)

#RECHECK DIMENSION OF TEST AND TRAIN DATASET
dim(Train_Cab)

##------------------------------------- 4 DATA VISUALIZATION -------------------------------------##
library(ggplot2)

#FARE AMOUNT
ggplot(Train_Cab, aes(x = factor(fare_amount))) +
  geom_bar(fill = "coral") +
  theme_classic()
ggplot(Train_Cab, aes_string(x = Train_Cab$fare_amount)) + 
  geom_histogram(fill="skyblue", colour = "black") + geom_density() +
  theme_bw() + xlab("Fare amount") + ylab("Frequency")+ggtitle("distribution of fare_amount")

# Visualization between fare_amount and passenger count.
ggplot(data = Train_Cab, aes(x = passenger_count,y = fare_amount))+
  geom_bar(stat = "identity",color ="green")+
  labs(title = "Fare Amount Vs. year", x = "Passenger Count", y = "Fare")+
  theme(plot.title = element_text(hjust = 0.5, face = "bold"))+
  theme(axis.text.x = element_text( color="green", size=6, angle=45))
# We can see, in year 2013 there were rides which got high fare_amount

# Visualization between fare_amount and date
ggplot(data = Train_Cab, aes(x = date, y = fare_amount))+
  geom_bar(stat = "identity",color = "yellow")+
  labs(title = "Fare Amount Vs.hour", x = "Date", y = "Fare")+
  theme(plot.title = element_text(hjust = 0.5, face = "bold"))+
  theme(axis.text.x = element_text( color="black", size=6, angle=45))

# Visualization between fare_amount and months.
ggplot(Train_Cab, aes(x = month,y = fare_amount))+ 
  geom_bar(stat = "identity",color = "Red")+
  labs(title = "Fare Amount Vs. Month", x = "Month", y = "Fare")+
  theme(axis.text.x = element_text( color="red", size=8))
# We can see month March collects the highest fare_amount


# Visualization between fare_amount and day.
ggplot(data = Train_Cab, aes(x = day,y = fare_amount))+
geom_bar(stat = "identity",color = "green")+
labs(title = "Fare Amount Vs. day", x = "Days of the week", y = "Fare")+
theme(plot.title = element_text(hjust = 0.5, face = "bold"))+
theme(axis.text.x = element_text( color="black", size=6, angle=45))
# We can see that  Thursday to Saturday rides has the highest fare_amount 

# Visualization between fare_amount and time.
ggplot(data = Train_Cab, aes(x = hour, y = fare_amount))+
  geom_bar(stat = "identity",color = "yellow")+
  labs(title = "Fare Amount Vs.time", x = "Time", y = "Fare")+
  theme(plot.title = element_text(hjust = 0.5, face = "bold"))+
  theme(axis.text.x = element_text( color="black", size=6, angle=45))
# Rides taken during 6 pm to 10 pm gives highest fare_amount
# Lets plot scatter plot for target and continous variables 

# Visualization between fare_amount and distance.
ggplot(data = Train_Cab, aes(x = distance, y = fare_amount))+
  geom_bar(stat = "identity",color = "yellow")+
  labs(title = "Fare Amount Vs. distance", x = "Distance", y = "Fare")+
  theme(plot.title = element_text(hjust = 0.5, face = "bold"))+
  theme(axis.text.x = element_text( color="black", size=6, angle=45))

  ##-------------------------------------5 FEATURE SELECTION ------------------------------------------##

#1. We have splitted the pickup_datetime variable into different varaibles like month, year, day etc so
#now we dont need to have that pickup_Date variable now. Hence we can drop that,
#2. Also we have created distance varible using pickup and drop longitudes and latitudes 
#so we will also drop pickup and drop longitudes and latitudes variables.
library(dplyr)
Train_Cab[ ,c('pickup_datetime', 'pickup_longitude','pickup_latitude','dropoff_longitude','dropoff_latitude')] <- list(NULL)
dim(Train_Cab)
head(Train_Cab)
str(Train_Cab)
#Train_Cab=Train_Cab %>%mutate(day = factor(day, order = TRUE, labels = c(1:7)))
#Train_Cab=Train_Cab %>% mutate_if(is.character, as.numeric)
#str(Train_Cab)

#CORRELATION ANALYSIS
numeric_index = sapply(Train_Cab,is.numeric)# Selecting only numeric 
numeric_index #fare_amount, passenger_count, distance
numeric_data =Train_Cab[,numeric_index]
cnames = colnames(numeric_data)
cnames
# Correlation Plot for to select significant continous variables 

#correlation matrix
correlation_matrix = cor(Train_Cab[,cnames])
correlation_matrix

library(corrgram)
#correlation plot
corrgram(Train_Cab[,numeric_index],order = F,upper.panel = panel.pie,
         text.panel = panel.txt,main = 'Correlation plot')

# We can say distance variable is moderately correlated with fare amount 
# rest of the variables also correlated positively and negative but we can 
# say them as weakly correlated we can use them in model 

# Anova Test is performed between cat_var (categorical independent variables) & fare_amount (continuous target variable) 
str(Train_Cab)
names(Train_Cab)
cat_var =c("date","year","month","day","hour")

# aov(Train_Cab$fare_amount~Train_Cab$year)
# for all categorical variables
for(i in cat_var){
  print(i)
  Anova_test_result = summary(aov(formula = fare_amount~Train_Cab[,i],Train_Cab))
  print(Anova_test_result)
}
names(Train_Cab)
# From the anova result, we can observe Date and day 
# has p value > 0.05, so delete this variable not consider in model.
# lets delete date and day variable
Train_Cab$day = NULL
Train_Cab$date = NULL

head(Train_Cab)
dim(Train_Cab)

 


##---------------------------------- 6] FEATURE SCALLING ----------------------------
# In our dataset fare amount and distance are the two continous
# variables whose disribution is slightly skewed

# Checking distance variable distribution using histogram
ggplot(Train_Cab, aes_string(x = Train_Cab$distance)) + 
  geom_histogram(fill="skyblue", colour = "black") + geom_density() +
  theme_bw() + xlab("distance") + ylab("Frequency")+ggtitle(" distribution of distance ")
# this variable is right skewed 

ggplot(Train_Cab, aes_string(x = Train_Cab$fare_amount)) + 
  geom_histogram(fill="skyblue", colour = "black") + geom_density() +
  theme_bw() + xlab("distance") + ylab("Frequency")+ggtitle(" distribution of distance ")

# Lets take log transformation to remove skewness
# Lets define function for log transformation of variables
signedlog10 = function(x) {
  ifelse(abs(x) <= 1, 0, sign(x)*log1p(abs(x)))
}

# Applying log function to distance variable
Train_Cab$distance = signedlog10(Train_Cab$distance)
#Train_Cab$distance=log1p(Train_Cab$distance)

# Checking distance distribution after applying function
ggplot(Train_Cab, aes_string(x = Train_Cab$distance)) + 
  geom_histogram(fill="skyblue", colour = "green") + geom_density() +
  theme_bw() + xlab("distance") + ylab("Frequency")+ggtitle(" distribution of distance after log transformation")

# Applying log function to fare_amount variable
Train_Cab$fare_amount = signedlog10(Train_Cab$fare_amount)
#Train_Cab$fare_amount=log1p(Train_Cab$fare_amount)

# Checking fare_amount distribution after applying function
ggplot(Train_Cab, aes_string(x = Train_Cab$fare_amount)) + 
  geom_histogram(fill="skyblue", colour = "green") + geom_density() +
  theme_bw() + xlab("distance") + ylab("Frequency")+ggtitle(" distribution of distance after log transformation")



head(Train_Cab)

#------------------------------------------------------------------------------------------------------------
       ######################### MODEL DEVELOPMENT ###############################            
#-----------------------------------------------------------------------------------------------------------
#As per observing on dataset we can conclude that our dataset is in continuous format. Our target variable has a integer value which is not refering to a classification or descrete class variable.

#Therefore, While applying modelling technique we choosed to go with regression modeling techniques which have task of predicting a continuous quantity :
  
  #We apply:
    #1. Linear Regression Model
    #2. Decision Tree regression Model
    #3. Random Forest Regression Model

library(DataCombine)
library(caret)
rmExcept("Train_Cab")
# Split the data set into train and test 
set.seed(1234)
train.index = createDataPartition(Train_Cab$fare_amount, p = .80, list = FALSE)
train_data = Train_Cab[train.index,]
test_data  = Train_Cab[-train.index,]
dim(train_data)
dim(test_data)
#There are 3 types of reggression matrix we have seen:
  #1. MAE (Mean Absolute Error)
  #2. MAPE (Mean Absolute Percentage Error)
  #3. RSME (Root squered Mean Error)
#As RSME we use when we have time series type data. and MAPE we use 
#when we just want percentage of deifference betweewn actual and predicted outcome.

MAPE = function(y, y1){
  mean(abs((y - y1)/y))
}

#------------------------------ 1] Linear Regression model ------------------------------

# fit linear regression model
# we will use the lm() function in the stats package
lm_model = lm(fare_amount ~.,data = train_data)
summary(lm_model)

#Residual standard error: 0.1226 on 12297 degrees of freedom
#Multiple R-squared:  0.7757,	Adjusted R-squared:  0.775
#F-statistic:   1013 on 42 and 12297 DF,  p-value: < 2.2e-16
#lets predict for train and test data

Predictions_LR_train = predict(lm_model,train_data)
Predictions_LR_test = predict(lm_model,test_data)


#let us check performance of our model

#mape calculation
LR_train_mape = MAPE(Predictions_LR_train,train_data[,1])
LR_test_mape = MAPE(test_data[,1],Predictions_LR_test)
print(LR_train_mape)#0.0896


#------------------------------------ 2] Decision tree regression ------------------------------------
library(rpart)
DT_model = rpart(fare_amount ~ ., data = train_data, method = "anova")
DT_model


# Lets predict for train and test data
predictions_DT_train= predict(DT_model,train_data)
predictions_DT_test= predict(DT_model,test_data)

# MAPE calculation
DT_train_mape = MAPE(train_data[,1],predictions_DT_train)
DT_test_mape = MAPE(test_data[,1],predictions_DT_test)
print(DT_train_mape)#0.084


# ---------------------------------- 3] RANDOM FOREST ------------------------------------
library(randomForest)
#library(e1071)
library(dplyr)

#While considering charachter variable in random forest modelling it will giving an error "foreign function call"
#so we have to convert that character variable into numeric or factor type
train_data=train_data %>% mutate_if(is.character, as.factor)
str(train_data)
test_data=test_data %>% mutate_if(is.character, as.factor)
str(test_data)
#lets build the random forest model
RF_model = randomForest(fare_amount~.,data = train_data, n.trees = 200)
print(RF_model)

# randomForest(formula = fare_amount ~ ., data = train_data, n.trees = 500) 
#Type of random forest: regression
#Number of trees: 500
#No. of variables tried at each split: 1
#Mean of squared residuals: 0.09194799 Var explained: 68.73

#lets predict for both train and test data
predictions_RF_train = predict(RF_model,train_data)
predictions_RF_test = predict(RF_model,test_data)

#MAPE calculation
RF_train_mape = MAPE(predictions_RF_train,train_data[,1])
RF_test_mape = MAPE(predictions_RF_test,test_data[,1])
print(RF_train_mape)#0.081


#---------------------------------------------------------------------------------------------------------------------------------
 #########################  MODEL EVALUATION ON TEST DATASET ###############################
#--------------------------------------------------------------------------------------------------------------------------------
#All steps for training data preproccessing we will follow for Test data
# Reading Test dataset
Test_Cab = read.csv("test.csv")

#Understanding Test data
class(Test_Cab)
dim(Test_Cab)
head(Test_Cab)
str(Test_Cab)
summary(Test_Cab)

#EDA
Test_Cab$pickup_datetime = gsub('// UTC','',Test_Cab$pickup_datetime)
#### Now same for Test Dataset ####
Test_Cab$date = as.Date(Test_Cab$pickup_datetime)

Test_Cab$year = substr(as.character(Test_Cab$date),1,4)

Test_Cab$month =substr(as.character(Test_Cab$date),6,7)

Test_Cab$day = weekdays(as.POSIXct(Test_Cab$date),abbreviate = F)

Test_Cab$date = substr(as.character(Test_Cab$date),9,10)

Test_Cab$hour = substr(as.factor(Test_Cab$pickup_datetime),12,13)

str(Test_Cab)
dim(Test_Cab)

#MISSING VALUE ANALYSIS
Missing_val_test= sum(is.na(Test_Cab)) 
Missing_val_test
#there are no missing values in test data

#OUTLIER ANALYSIS
#########For passanger_count variable###########
summary(Test_Cab$passenger_count)
Test_Cab[Test_Cab$passenger_count < 1,"passenger_count"] = NA
Test_Cab[Test_Cab$passenger_count > 6,"passenger_count"] = NA

summary(Train_Cab$passenger_count)

#recheck dimension 
dim(Test_Cab)
Test_Cab[Test_Cab$pickup_longitude < -180 ,"pickup_longitude"] = NA
Test_Cab[Test_Cab$pickup_longitude > 180 ,"pickup_longitude"] = NA
sum(is.na(Test_Cab))
Test_Cab[Test_Cab$pickup_latitude < -90,"pickup_latitude"] = NA
Test_Cab[Test_Cab$pickup_latitude > 90,"pickup_latitude"] = NA
sum(is.na(Test_Cab))
Test_Cab[Test_Cab$dropoff_longitude < -180,"dropoff_longitude"] = NA
Test_Cab[Test_Cab$dropoff_longitude > 180,"dropoff_longitude"] = NA
sum(is.na(Test_Cab))# 1 found
Test_Cab$dropoff_latitude<- ifelse(Test_Cab$dropoff_latitude < -90, NA,Test_Cab$dropoff_latitude )
Test_Cab$dropoff_latitude<- ifelse(Test_Cab$dropoff_latitude > 90, NA,Test_Cab$dropoff_latitude )
sum(is.na(Test_Cab))#2 found
Test_Cab = na.omit(Test_Cab)
sum(is.na(Test_Cab))
#after missing value analysis structure of train will be
dim(Test_Cab)


#Some preprocessing technique
#Haversine formla
#y = install.packages(c('purrr','geosphere','rlist'))

get_geo_distance = function(long1, lat1, long2, lat2) {
  loadNamespace("purrr")
  loadNamespace("geosphere") 
  longlat1 = purrr::map2(long1, lat1, function(x,y) c(x,y))
  longlat2 = purrr::map2(long2, lat2, function(x,y) c(x,y))
  distance_list = purrr::map2(longlat1, longlat2, function(x,y) geosphere::distHaversine(x, y))
  distance_m = list.extract(distance_list, position = 1)
  #if (units == "km") {
  distance = distance_m / 1000.0;
  
  distance
}


for(i in (1:nrow(Test_Cab)))
{
  Test_Cab$distance[i]= get_geo_distance(Test_Cab$pickup_longitude[i],Test_Cab$pickup_latitude[i],Test_Cab$dropoff_longitude[i],Test_Cab$dropoff_latitude[i])
}
head(Test_Cab)

Test_Cab[Test_Cab$distance == 0,'distance'] <- NA
sort(Test_Cab$distance, decreasing = FALSE)
Test_Cab$distance<- ifelse(Test_Cab$distance > 130, NA,Test_Cab$distance )
sum(is.na(Test_Cab))# 85 found
Test_Cab = na.omit(Test_Cab)
dim(Test_Cab)

#FEATURE SELECTION
Test_Cab[ ,c('pickup_datetime', 'pickup_longitude','pickup_latitude','dropoff_longitude','dropoff_latitude')] <- list(NULL)
dim(Test_Cab)
head(Test_Cab)
str(Test_Cab)
Test_Cab$day = NULL
Test_Cab$date = NULL

signedlog10 = function(x) {
  ifelse(abs(x) <= 1, 0, sign(x)*log1p(abs(x)))
}

Test_Cab$distance=signedlog10(Test_Cab$distance)


#MODEL DEVELOPMENT
#Now we will do model devlopment on test data
#As training data modeldevelopment we get good accuracy on Random Forest model
#So here we will fit random forest on Test_Cab
library(randomForest)
library(e1071)
library(dplyr)
Test_Cab=Test_Cab %>% mutate_if(is.character, as.factor)
RF_model = randomForest(fare_amount~.,data = train_data, n.trees = 200)
print(RF_model)
RFTest_Cab = predict(RF_model, Test_Cab)
RFTest_Cab
summary(RFTest_Cab)
Test_Cab$Fare_amount<-RFTest_Cab
print(head(Test_Cab))
