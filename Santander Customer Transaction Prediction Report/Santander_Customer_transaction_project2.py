import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn import metrics 
from sklearn.linear_model import LogisticRegression 
from sklearn import tree 
from sklearn.ensemble import RandomForestClassifier 
from sklearn.naive_bayes import GaussianNB 
from sklearn.neighbors import KNeighborsClassifier

#load data
train_data=pd.read_csv("train.csv")
test_data=pd.read_csv("test.csv")

#We have to convert target variable into 'int64' for further arithmatic operation
train_data['target']=train_data['target'].astype('int64')

#Outlier Analysis
numerical_features=train_data.columns[2:]
numerical_features

for i in numerical_features:
    #print(i)
    q75, q25 = np.percentile(sorted(train_data.loc[:,i]), [75 ,25])
    iqr = q75 - q25

    min = q25 - (iqr*1.5)
    max = q75 + (iqr*1.5)
    
    train_data[train_data[i] < min] = np.nan
    train_data[train_data[i] > max] = np.nan
train_data = train_data.fillna(train_data.mean())

numerical_features_test=test_data.columns[1:]
for i in numerical_features_test:

    q75, q25 = np.percentile(test_data.loc[:,i], [75 ,25])
    iqr = q75 - q25

    min = q25 - (iqr*1.5)
    max = q75 + (iqr*1.5)
    
    test_data[test_data[i] < min] = np.nan
    test_data[test_data[i] > max] = np.nan
test_data = test_data.fillna(test_data.mean())

#While doing outlier analysis our target variable again get transform to folat type 
#so here we again type cast it to int64 
train_data['target']=train_data['target'].astype('int64')

#Feature Engineering
# At this point of time we should check for among 200 features which variables are useful for us.
# As the features are anonymous. This can be done by using PCA
# scale the data before using PCA
mmscale = MinMaxScaler()  
X_train = mmscale.fit_transform(train_data.drop(['target','ID_code'],axis=1))  
X_test = mmscale.transform(test_data.drop(['ID_code'],axis=1)) 

pca = PCA()  
a = pca.fit_transform(X_train) 
print("Trasformed shape of train data: ", a.shape)
b = pca.transform(X_test)
print("Trasformed shape of test data: ", b.shape)

explained_variance = pca.explained_variance_ratio_
plt.figure(figsize=(8,5))
plt.plot(np.arange(200), np.cumsum(explained_variance))

#Adding Fearture
for df in [train_data,test_data]:
    df['sum'] = df[numerical_features].sum(axis=1)  
    df['min'] = df[numerical_features].min(axis=1)
    df['max'] = df[numerical_features].max(axis=1)
    df['mean'] = df[numerical_features].mean(axis=1)
    df['std'] = df[numerical_features].std(axis=1)
    df['skew'] = df[numerical_features].skew(axis=1)
    df['kurt'] = df[numerical_features].kurtosis(axis=1)
    df['med'] = df[numerical_features].median(axis=1)
print(train_data.shape)
print(test_data.shape)

#Model Development
#Divide data into train and test
X = train_data.values[:, 2:]
Y = train_data.values[:,1] #ValueError: Unknown label type: 'unknown'
#Therefore we convert Y object to int
Y=Y.astype('int') 

#Logistic Regression
X_train, X_test, y_train, y_test = train_test_split( X, Y, test_size = 0.2,random_state=100)
logreg = LogisticRegression().fit(X_train, y_train)
logit_pred = logreg.predict(X_test)

CM = pd.crosstab(y_test, logit_pred)
#let us save TP, TN, FP, FN
TN = CM.iloc[0,0]
FN = CM.iloc[1,0]
TP = CM.iloc[1,1]
FP = CM.iloc[0,1]

#check accuracy of model
#accuracy_score(y_test, y_pred)*100
print("Accuracy score:", ((TP+TN)*100)/(TP+TN+FP+FN))
print("False Positive rate: ", (FP*100)/(FP+TN)) #Type-1 error
print("False Negative rate: ", (FN*100)/(FN+TP)) #Type-2 error
print("Specificity/True Negative rate: ", (TN*100)/(FP+TN))
print("Recall/ True Positive rate: ", (TP*100)/(FN+TP))

fpr, tpr, thresholds = metrics.roc_curve(y_test, logit_pred)
auc_score=metrics.auc(fpr, tpr)
print("AUC Score is: ", auc_score)

#Decision Tree Classification
#Decision Tree
C50_model = tree.DecisionTreeClassifier(criterion='entropy').fit(X_train, y_train)

#predict new test cases
C50_Predictions = C50_model.predict(X_test)
#Build confusion matrix
CM = pd.crosstab(y_test, C50_Predictions)
#let us save TP, TN, FP, FN
TN = CM.iloc[0,0]
FN = CM.iloc[1,0]
TP = CM.iloc[1,1]
FP = CM.iloc[0,1]

#check accuracy of model
#accuracy_score(y_test, y_pred)*100
print("Accuracy score:", ((TP+TN)*100)/(TP+TN+FP+FN))
print("False Positive rate: ", (FP*100)/(FP+TN)) #Type-1 error
print("False Negative rate: ", (FN*100)/(FN+TP)) #Type-2 error
print("Specificity/True Negative rate: ", (TN*100)/(FP+TN))
print("Recall/ True Positive rate: ", (TP*100)/(FN+TP))

fpr, tpr, thresholds = metrics.roc_curve(y_test, C50_Predictions)
auc_score=metrics.auc(fpr, tpr)
print("AUC Score is: ", auc_score)

#Random Forest Classifier
#Random Forest
RF_model = RandomForestClassifier(n_estimators = 20).fit(X_train, y_train)
RF_Predictions = RF_model.predict(X_test)
CM = pd.crosstab(y_test, RF_Predictions)

#let us save TP, TN, FP, FN
TN = CM.iloc[0,0]
FN = CM.iloc[1,0]
TP = CM.iloc[1,1]
FP = CM.iloc[0,1]

#check accuracy of model
#check accuracy of model
#accuracy_score(y_test, y_pred)*100
print("Accuracy score:", ((TP+TN)*100)/(TP+TN+FP+FN))
print("False Positive rate: ", (FP*100)/(FP+TN)) #Type-1 error
print("False Negative rate: ", (FN*100)/(FN+TP)) #Type-2 error
print("Specificity/True Negative rate: ", (TN*100)/(FP+TN))
print("Recall/ True Positive rate: ", (TP*100)/(FN+TP))

fpr, tpr, thresholds = metrics.roc_curve(y_test, RF_Predictions)
auc_score=metrics.auc(fpr, tpr)
print("AUC Score is: ", auc_score)

#Naive bayes classifier
#Naive Bayes implementation
NB_model = GaussianNB().fit(X_train, y_train)
#predict test cases
NB_Predictions = NB_model.predict(X_test)
CM = pd.crosstab(y_test, NB_Predictions)
print(CM)
#let us save TP, TN, FP, FN
TN = CM.iloc[0,0]
FN = CM.iloc[1,0]
TP = CM.iloc[1,1]
FP = CM.iloc[0,1]
#check accuracy of model
print("Accuracy score:", ((TP+TN)*100)/(TP+TN+FP+FN))
print("False Positive rate: ", (FP*100)/(FP+TN)) #Type-1 error
print("False Negative rate: ", (FN*100)/(FN+TP)) #Type-2 error
print("Specificity/True Negative rate: ", (TN*100)/(FP+TN))
print("Recall/ True Positive rate: ", (TP*100)/(FN+TP))

fpr, tpr, thresholds = metrics.roc_curve(y_test, NB_Predictions)
auc_score=metrics.auc(fpr, tpr)
print("AUC Score is: ", auc_score)

store_id= test_data['ID_code']
test_data.drop(['ID_code'],axis=1,inplace=True)
NB_Predictions = NB_model.predict(test_data)
new_df=pd.DataFrame({'ID_code': store_id, 'Predicted_target':NB_Predictions})
new_df.to_csv('New_data.csv',index=False)
mydf=pd.read_csv('New_data.csv')
mydf.head()







