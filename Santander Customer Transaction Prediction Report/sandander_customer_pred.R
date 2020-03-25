# Clean the environment
rm(list=ls())


# Set working directory
setwd("D:/EdwisorDS/Assignment/Santandar_Prediction_Project2")
x = c("ggplot2", "corrgram", "DMwR", "caret", "randomForest", "e1071", "rpart",  'DataCombine', 'inTrees','dplyr','class','pROC')

#C50,unbalanced,dummies,Mass,"gbm", "ROSE", 'sampling', "Information"

#install.packages(x)
#NOW TO LOAD MULTIPLE PACKAGES AT TIME
lapply(x, require, character.only = TRUE)
rm(x)
#---------------------------------------------------------------------------------------
#                                Load the data 
#-----------------------------------------------------------------------------------------
Train_data = read.csv("train.csv")
dim(Train_data)
str(Train_data)
summary(Train_data)
#Bar plot for count of target classes in train data:-
plot1=ggplot(Train_data,aes(target))+theme_bw()+geom_bar(stat='count',fill='lightgreen')
plot1


#-------------------------------------------------------------------------------------------------
#                             Data Visualization 
#-------------------------------------------------------------------------------------------------

#Mean value per rows and columns in train & test dataset:-
#Applying the function to find mean values per row in train and test data.
train_mean<-apply(Train_data[,-c(1,2)],MARGIN=1,FUN=mean)
test_mean<-apply(Test_data[,-c(1)],MARGIN=1,FUN=mean)
ggplot()+
  #Distribution of mean values per row in train data
  geom_density(data=Train_data[,-c(1,2)],aes(x=train_mean),kernel='gaussian',show.legend=TRUE,color='blue')+theme_classic()+
  #Distribution of mean values per row in test data
  geom_density(data=Test_data[,-c(1)],aes(x=test_mean),kernel='gaussian',show.legend=TRUE,color='green')+
  labs(x='mean values per row',title="Distribution of mean values per row in train and test dataset")

#Applying the function to find mean values per column in train and test data.
train_mean<-apply(Train_data[,-c(1,2)],MARGIN=2,FUN=mean)
test_mean<-apply(Test_data[,-c(1)],MARGIN=2,FUN=mean)
ggplot()+
  #Distribution of mean values per column in train data
  geom_density(aes(x=train_mean),kernel='gaussian',show.legend=TRUE,color='blue')+theme_classic()+
  #Distribution of mean values per column in test data
  geom_density(aes(x=test_mean),kernel='gaussian',show.legend=TRUE,color='green')+
  labs(x='mean values per column',title="Distribution of mean values per column in train and test dataset")


#Applying the function to find standard deviation values per row in train and test data.
train_sd<-apply(Train_data[,-c(1,2)],MARGIN=1,FUN=sd)
test_sd<-apply(Test_data[,-c(1)],MARGIN=1,FUN=sd)
ggplot()+
  #Distribution of sd values per row in train data
  geom_density(data=Train_data[,-c(1,2)],aes(x=train_sd),kernel='gaussian',show.legend=TRUE,color='red')+theme_classic()+
  #Distribution of sd values per row in test data
  geom_density(data=Test_data[,-c(1)],aes(x=test_sd),kernel='gaussian',show.legend=TRUE,color='blue')+
  labs(x='sd values per row',title="Distribution of sd values per row in train and test dataset")

#Applying the function to find sd values per column in train and test data.
train_sd<-apply(Train_data[,-c(1,2)],MARGIN=2,FUN=sd)
test_sd<-apply(Test_data[,-c(1)],MARGIN=2,FUN=sd)
ggplot()+
  #Distribution of sd values per column in train data
  geom_density(aes(x=train_sd),kernel='gaussian',show.legend=TRUE,color='red')+theme_classic()+
  #Distribution of sd values per column in test data
  geom_density(aes(x=test_sd),kernel='gaussian',show.legend=TRUE,color='blue')+
  labs(x='sd values per column',title="Distribution of std values per column in train and test dataset")

#Applying the function to find skewness values per row in train and test data.
train_skew<-apply(Train_data[,-c(1,2)],MARGIN=1,FUN=skewness)
test_skew<-apply(Test_data[,-c(1)],MARGIN=1,FUN=skewness)
ggplot()+
  #Distribution of skewness values per row in train data
  geom_density(aes(x=train_skew),kernel='gaussian',show.legend=TRUE,color='green')+theme_classic()+
  #Distribution of skewness values per column in test data
  geom_density(aes(x=test_skew),kernel='gaussian',show.legend=TRUE,color='blue')+
  labs(x='skewness values per row',title="Distribution of skewness values per row in train and test dataset")

#Applying the function to find skewness values per column in train and test data.
train_skew<-apply(Train_data[,-c(1,2)],MARGIN=2,FUN=skewness)
test_skew<-apply(Test_data[,-c(1)],MARGIN=2,FUN=skewness)
ggplot()+
  #Distribution of skewness values per column in train data
  geom_density(aes(x=train_skew),kernel='gaussian',show.legend=TRUE,color='green')+theme_classic()+
  #Distribution of skewness values per column in test data
  geom_density(aes(x=test_skew),kernel='gaussian',show.legend=TRUE,color='blue')+
  labs(x='skewness values per column',title="Distribution of skewness values per column in train and test dataset")

#Applying the function to find kurtosis values per row in train and test data.
train_kurtosis<-apply(Train_data[,-c(1,2)],MARGIN=1,FUN=kurtosis)
test_kurtosis<-apply(Test_data[,-c(1)],MARGIN=1,FUN=kurtosis)
ggplot()+
  #Distribution of kurtosis values per row in train data
  geom_density(aes(x=train_kurtosis),kernel='gaussian',show.legend=TRUE,color='blue')+theme_classic()+
  #Distribution of kurtosis values per row in test data
  geom_density(aes(x=test_kurtosis),kernel='gaussian',show.legend=TRUE,color='red')+
  labs(x='kurtosis values per row',title="Distribution of kurtosis values per row in train and test dataset")

#Applying the function to find kurtosis values per column in train and test data.
train_kurtosis<-apply(Train_data[,-c(1,2)],MARGIN=2,FUN=kurtosis)
test_kurtosis<-apply(Test_data[,-c(1)],MARGIN=2,FUN=kurtosis)
ggplot()+
  #Distribution of kurtosis values per column in train data
  geom_density(aes(x=train_kurtosis),kernel='gaussian',show.legend=TRUE,color='blue')+theme_classic()+
  #Distribution of kurtosis values per column in test data
  geom_density(aes(x=test_kurtosis),kernel='gaussian',show.legend=TRUE,color='red')+
  labs(x='kurtosis values per column',title="Distribution of kurtosis values per column in train and test dataset")

#-------------------------------------------------------------------------------------------------
#                                     Missing Value Analysis 
#-------------------------------------------------------------------------------------------------
sum(is.na(Train_data))
sum(is.na(Test_data))

#-------------------------------------------------------------------------------------------------
#                                      OUTLIER ANALYSIS 
#-------------------------------------------------------------------------------------------------
numeric_val=Train_data[c(3:202)]
numeric_features=colnames(numeric_val)
numeric_features
for(i in numeric_features){
  val = Train_data[,i][Train_data[,i] %in% boxplot.stats(Train_data[,i])$out]
  #print(length(val))
  Train_data[,i][Train_data[,i] %in% val] = NA
}


for(i in numeric_features){
  Train_data[,i][is.na(Train_data[,i])] = mean(Train_data[,i], na.rm = T)
}

sum(is.na(Train_data))

Train_data<-Train_data %>% select(-c('ID_code'))
dim(Train_data)
str(Train_data)

#-------------------------------------------------------------------------------------------------
#                                 Feature Engineering 
#-------------------------------------------------------------------------------------------------
#Correlations in train data:-
#convert factor to int
Train_data$target<-as.numeric(Train_data$target)
train_correlation<-cor(Train_data[,c(2:202)])
train_correlation

#Observation:- We can observe that correlation between train attributes is very small.

Train_data<-Train_data %>% select(-c('ID_code'))
dim(Train_data)

Train_data$sum<-apply(Train_data[,-c(1)],2,sum)
Train_data$min<-apply(Train_data[,-c(1,202)],2,min)
Train_data$max<-apply(Train_data[,-c(1,202,203)],2,max)
Train_data$mean<-apply(Train_data[,-c(1,202,203,204)],2,mean)
Train_data$sd<-apply(Train_data[,-c(1,202,203,204,205)],2,sd)
Train_data$skewness<-apply(Train_data[,-c(1,202,203,204,205,206)],2,skewness)
Train_data$Kurtosis<-apply(Train_data[,-c(1,202,203,204,205,206,207)],2,kurtosis)
dim(Train_data)

#-------------------------------------------------------------------------------------------------
#                                MODEL DEVELOPMENT 
#-------------------------------------------------------------------------------------------------
#Clean the environment
rmExcept("Train_data")

#Divide data into train and test using stratified sampling method
set.seed(1234)
train.index = createDataPartition(Train_data$target, p = .80, list = FALSE)
train = Train_data[ train.index,]
dim(train)
test  = Train_data[-train.index,]
dim(test)

#Logistic Regression
logit_model = glm(target ~ ., data = train, family = "binomial")

#summary of the model
summary(logit_model)

#LOGISTIC REGRESSION

logit_Predictions = predict(logit_model, newdata = test, type = "response")
#convert prob
logit_Predictions = ifelse(logit_Predictions > 0.5, 1, 0)

##Evaluate the performance of classification model
ConfMatrix_RF = table(test$target, logit_Predictions)
confusionMatrix(ConfMatrix_RF) #91.62% accuracy

#AUC Score
auc(test$target, logit_Predictions)

##Output:
# Confusion Matrix and Statistics
# 
# logit_Predictions
# 0     1
# 0 35530   481
# 1  2872  1117
# 
# Accuracy : 0.9162          
# 95% CI : (0.9134, 0.9189)
# No Information Rate : 0.96            
# P-Value [Acc > NIR] : 1               
# 
# Kappa : 0.3635          
# 
# Mcnemar's Test P-Value : <2e-16          
#                                           
#             Sensitivity : 0.9252          
#             Specificity : 0.6990          
#          Pos Pred Value : 0.9866          
#          Neg Pred Value : 0.2800          
#              Prevalence : 0.9600          
#          Detection Rate : 0.8882          
#    Detection Prevalence : 0.9003          
#       Balanced Accuracy : 0.8121          
#                                           
#        'Positive' Class : 0     

#DECISISON TREE CLASSIFIER

#First convert target numeric type to factor
train$target=as.factor(train$target)

#Develop Model on training data
rpart_model = rpart(target ~., train, method = 'class')
#Summary of DT model
#summary(rpart_model)
#Lets predict for test cases
rprat_Predictions = predict(rpart_model, test[,-1], type = "class")

##Evaluate the performance of classification model
ConfMatrix_rpart = table(test$target, rprat_Predictions)
confusionMatrix(ConfMatrix_rpart) #90.03 accuracy
typeof(test$target)
#AUC Score
rprat_Predictions=as.numeric(rprat_Predictions)
auc(test$target, rprat_Predictions)
##Output:
# Confusion Matrix and Statistics
# 
# rprat_Predictions
# 0     1
# 0 36011     0
# 1  3989     0
# 
# Accuracy : 0.9003          
# 95% CI : (0.8973, 0.9032)
# No Information Rate : 1               
# P-Value [Acc > NIR] : 1               
# 
# Kappa : 0               
# 
# Mcnemar's Test P-Value : <2e-16          
#                                           
#             Sensitivity : 0.9003          
#             Specificity :     NA          
#          Pos Pred Value :     NA          
#          Neg Pred Value :     NA          
#              Prevalence : 1.0000          
#          Detection Rate : 0.9003          
#    Detection Prevalence : 0.9003          
#       Balanced Accuracy :     NA          
#                                           
#        'Positive' Class : 0   


#RANDOM FOREST CLASSIFIER

# RF_model = randomForest(target ~ ., train, importance = TRUE, ntree = 1000)
train$target=as.factor(train$target)
str(train)
RF_model = randomForest(target ~ ., train,importance = TRUE,ntree = 20, do.trace=TRUE)
#Presdict test data using random forest model
RF_Predictions = predict(RF_model, test[,-1])

##Evaluate the performance of classification model
ConfMatrix_RF = table(test$target, RF_Predictions)
confusionMatrix(ConfMatrix_RF)

#AUC Score
RF_Predictions=as.numeric(RF_Predictions)
auc(test$target, RF_Predictions)

###Output:
# Confusion Matrix and Statistics
# 
# RF_Predictions
# 0     1
# 0 35994    17
# 1  3939    50
# 
# Accuracy : 0.9011         
# 95% CI : (0.8981, 0.904)
# No Information Rate : 0.9983         
# P-Value [Acc > NIR] : 1              
# 
# Kappa : 0.0214         
# 
# Mcnemar's Test P-Value : <2e-16         
#                                          
#             Sensitivity : 0.90136        
#             Specificity : 0.74627        
#          Pos Pred Value : 0.99953        
#          Neg Pred Value : 0.01253        
#              Prevalence : 0.99833        
#          Detection Rate : 0.89985        
#    Detection Prevalence : 0.90028        
#       Balanced Accuracy : 0.82381        
#                                          
#        'Positive' Class : 0     


#NAIVE BAYES
train$target=as.factor(train$target)
NB_model = naiveBayes(target ~ ., data = train)
#predict on test cases #raw
NB_Predictions = predict(NB_model, test[,-1], type = 'class')

#Look at confusion matrix
Conf_matrix = table(test$target,NB_Predictions)
confusionMatrix(Conf_matrix)

#AUC SCORE
NB_Predictions=as.numeric(NB_Predictions)
auc(test$target, NB_Predictions)


##Ouput:
# Confusion Matrix and Statistics
# 
# NB_Predictions
# 0     1
# 0 35455   556
# 1  2529  1460
# 
# Accuracy : 0.9229          
# 95% CI : (0.9202, 0.9255)
# No Information Rate : 0.9496          
# P-Value [Acc > NIR] : 1               
# 
# Kappa : 0.4494          
# 
# Mcnemar's Test P-Value : <2e-16          
#                                           
#             Sensitivity : 0.9334          
#             Specificity : 0.7242          
#          Pos Pred Value : 0.9846          
#          Neg Pred Value : 0.3660          
#              Prevalence : 0.9496          
#          Detection Rate : 0.8864          
#    Detection Prevalence : 0.9003          
#       Balanced Accuracy : 0.8288          
#                                           
#        'Positive' Class : 0               

#-------------------------------------------------------------------------------------------------
#                          Predict target value for Test dataset 
#-------------------------------------------------------------------------------------------------
Test_data = read.csv("test.csv")

#Outlier Analysis

numeric_val_test=Test_data[c(2:201)]
numeric_features_test=colnames(numeric_val_test)
for(i in numeric_features_test){
  val = Test_data[,i][Test_data[,i] %in% boxplot.stats(Test_data[,i])$out]
  #print(length(val))
  Test_data[,i][Test_data[,i] %in% val] = NA
}

sum(is.na(Test_data)) #27087

for(i in numeric_features_test){
  Test_data[,i][is.na(Test_data[,i])] = mean(Test_data[,i], na.rm = T)
}

sum(is.na(Test_data))
dim(Test_data)


head(Test_data)
str(Test_data)

Id_code= Test_data$ID_code
Test_data<-Test_data %>% select(-c('ID_code'))
dim(Test_data)

Test_data$sum<-apply(Test_data,2,sum)
Test_data$min<-apply(Test_data[,-c(201)],2,min)
Test_data$max<-apply(Test_data[,-c(201,202)],2,max)
Test_data$mean<-apply(Test_data[,-c(201,202,203)],2,mean)
Test_data$sd<-apply(Test_data[,-c(201,202,203,204)],2,sd)
Test_data$skewness<-apply(Test_data[,-c(201,202,203,204,205)],2,skewness)
Test_data$Kurtosis<-apply(Test_data[,-c(201,202,203,204,205,206)],2,kurtosis)

NB_model = naiveBayes(target ~ ., data = train)
#predict on test cases #raw
NB_Predictions = predict(NB_model,Test_data , type = 'class')
df<-data.frame("ID_Code"=Id_code, "Predicted_Target"=NB_Predictions)
df



