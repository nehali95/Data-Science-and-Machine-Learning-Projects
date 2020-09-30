# Data-Science-and-Machine-Learning-Projects

# 1. Cab fare prediction 

### Problem Statement -
You are a cab rental start-up company. You have successfully run the pilot project and
now want to launch your cab service across the country. You have collected the
historical data from your pilot project and now have a requirement to apply analytics for
fare prediction. You need to design a system that predicts the fare amount for a cab ride
in the city.
### Data Set :
1) train_cab.zip
2) test.zip
### Number of attributes:
· pickup_datetime - timestamp value indicating when the cab ride started.
· pickup_longitude - float for longitude coordinate of where the cab ride started.
· pickup_latitude - float for latitude coordinate of where the cab ride started.
· dropoff_longitude - float for longitude coordinate of where the cab ride ended.
· dropoff_latitude - float for latitude coordinate of where the cab ride ended.
· passenger_count - an integer indicating the number of passengers in the cab
ride.
#### Missing Values : Yes

# 2. Santander Customer Transaction Prediction
### Background -
At Santander , mission is to help people and businesses prosper. We are always looking
for ways to help our customers understand their financial health and identify which
products and services might help them achieve their monetary goals.
Our data science team is continually challenging our machine learning algorithms,
working with the global data science community to make sure we can more accurately
identify new ways to solve our most common challenge, binary classification problems
such as: is a customer satisfied? Will a customer buy this product? Can a customer pay
this loan?

### Problem Statement -
In this challenge, we need to identify which customers will make a specific transaction in
the future, irrespective of the amount of money transacted.

### The data is taken from kaggle which includes train & test data. Click on the link to access the dataset https://www.kaggle.com/c/santander-customer-transaction-prediction/data

# 3.NLP 
### Background -
Missinon is to find Spam and Ham massages and apply that techniques to test data with best accuracy. t's binary classification problem. There no missing values.

### Technique used -
##### Pre-processing techniques - Tokanization, punctuation remove, steaming , remove stopewords
##### Vectorizing Raw Data: TF-IDF
TF-IDF Creates a document-term matrix where the columns represent single unique terms (unigrams) but the cell represents a weighting meant to represent how important a word is to a document. For making unstructured data to structured format.
##### Feature engineering transformation - adding two new features
##### Applied classification algorithms
Random forest classifier
Gradient Boosting Classifier
Random Forest model with grid-search
Random Forest on a holdout test set

