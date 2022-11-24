import sklearn
import numpy as np
import pickle
import pandas as pd
import numpy as np
import sklearn
import sklearn.datasets
import pandas as pd 
# getting  the dataset
breast_cancer = sklearn.datasets.load_breast_cancer()
print(breast_cancer) 

X = breast_cancer.data
Y = breast_cancer.target
print(X)
print(Y)
print(X.shape, Y.shape)

data = pd.DataFrame(breast_cancer.data, columns = breast_cancer.feature_names)
data['class'] = breast_cancer.target # the class  represent the label that is 0 and at all
data.head()
from operator import index
# now we will  see how many example for benigan and how many for maligan
print(data['class'].value_counts())
print(breast_cancer.target_names)
data.head()
# now class 0 is for malingat and class 0 is for 1 is for benigan
a = ['malignant' 'benign'] # now there is 212 cases for maligant and 357 for 357 for maligant 
print(breast_cancer.target_names)
data.groupby('class').mean()
from sklearn.model_selection import train_test_split
X_train,X_train,Y_train,Y_test = train_test_split(X,Y)
print(Y.shape,Y_train.shape,Y_test.shape)
X_train, X_test, Y_train,Y_test = train_test_split(X,Y,test_size=0.1)
# point 0.1 represent the 10 percentage of the data because i needed the 10  percetage of my data
print(Y.shape,Y_train.shape, Y_test.shape)
# SO WE TAKED 10 PERCENTAGE OF OUR TESTING DATA WHICH IS 57
print(Y.mean(),Y_train.mean(), Y_test.mean())
X_train, X_test, Y_train,Y_test = train_test_split(X,Y,test_size=0.1, stratify=Y)
# WE are going to startify based on 0 or 1 into equal part startify is to split the data correctly
# for correct distribution of data
print(Y.mean(),Y_train.mean(),Y_test.mean())

X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size=0.1, random_state=1)
#random_state --> random state is specific split of data each random value has of random_state split the data differently
# there is 30 poples takes default for each
# the random_state is random number generator it may be an

print(X_train.mean(),X_test.mean(),X.mean())
print(X_train)
# let's import import Logistic Regression from sklearn
from sklearn.linear_model import LogisticRegression
# we can develope the logistic regression from scrach but it's time consuming
classifier = LogisticRegression() # loading the logistic regression model to variable classifier
# training the model on training data
# wheater it's malingan or benigan 
classifier.fit(X_train, Y_train)
# my model is trained with the training data we had given
# now let's go ahead and check the evaluation of our model
# import accuracy_score
# WE NEED TO IMPORT ACCURACY SCORE FROM SKLEARN
from sklearn.metrics import accuracy_score
# let's see how model works on test data let's see how model works on test data
# prediction on traninig data
prediction_on_training_data = classifier.predict(X_train)
accuracy_on_trainig_data = accuracy_score(Y_train, prediction_on_training_data)
print('Accuracy on trainig data :', accuracy_on_trainig_data)
# now the accuracy is 95 percentage let's go ahead and see


# we need to import accuracy score from sklearn
# so this is the evaluation method for the 
# now logistic regresion model 
# import accuracy score
# let's see how the model works on training data
# let's see on how model works on test data

# let's make the prediction on test data

# prediciton on test_data
prediction_on_test_data = classifier.predict(X_test)
accuracy_on_test_data = accuracy_score(Y_test,prediction_on_test_data)
print('Accuracy on test  data :', accuracy_on_test_data)
# accuracy on test data is greater than the accuracy on training data
input_data = (13.08,15.71,85.63,520,0.1075,0.127,0.04568,0.0311,0.1967,0.06811,0.1852,0.7477,1.383,14.67,0.004097,0.01898,0.01698,0.00649,0.01678,0.002425,14.5,20.49,96.09,630.5,0.1312,0.2776,0.189,0.07283,0.3184,0.08183)
# change the input data to numpy arrays to make prediction
input_data_as_numpy_array = np.asarray(input_data)
print(input_data)
input_data_reshaped = input_data_as_numpy_array.reshape(1,-1)
prediction = classifier.predict(input_data_reshaped)
print(prediction) # returns a list with 0 and 1 we have to get from the list 
# returns a list with [0] if maligant; return a list with element[1], if benign

# if 0 it's means malingant  otherwise it's benign
# 
# accutaly it's returns a list element 0 and 1 0 it means maligan if 1 it means maligan
# we have need to get 0 and 1 from this list
# if person's comes we can analyze the person is from which stage of cancer 
if (prediction[0] == 0):
  print('the breast Cancer is Malingant')
else:
  print('The breast Cacer is benigan')


pickle.dump(classifier, open('nmodel.pkl',"wb"))

