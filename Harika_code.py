
## Author : Harika Malapaka
## Unity ID : hsmalapa
   ## CLASSIFIERS ##
## Please use the attached file for the data, as I changed the format of the data (like string to float conversions) which by given by Professor.
## The data file is DATA... Please find this file

## Data Cleaning was done in the Excel file itself, and it's bought to the pyCharm. SO please use the formatted file (DATA).

import pandas as pd
from sklearn import svm
print("This may take time...Please be Patient ")
print(" Knindly ignore the warnings. ")

print("The output is what what asked in the description only")
print("----------------------------------------------------------------")
print(" Loading output...............")
print(" Please wait till the end of the program where it says  DONE !")
print("\n\n")
from sklearn.metrics import accuracy_score
from sklearn.model_selection import  cross_val_score
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import precision_recall_fscore_support

## 2 data structures so that selecting data is easy
new_data_2=pd.read_csv('DATA',sep="\t",usecols=[0,3,5,6,7,9,11,12,13,14,15,16])
new_data_3=pd.read_csv('DATA',sep="\t",usecols=[10]) ## ONTASK data

## To see the feaature set, uncomment the following code:
# print(new_data_2)

## Features in X
X = new_data_2.values[:, 0:12]

## Labels in Y
Y = new_data_3.values[:,0]

## Define train set, test set, train labels, test labels
X_train, X_test, y_train, y_test = train_test_split( X, Y, test_size = 0.3, random_state = 100)

## Define 2 classifiers
dt = DecisionTreeClassifier(max_depth=5,max_leaf_nodes=3,class_weight='balanced')
sv=svm.SVC()

## Fit the training data in the classifier
dt.fit(X_train, y_train)
sv.fit(X_train,y_train)


## Predict the test data now which is 30%
y_pred_dt = dt.predict(X_test)
y_pred_svm=sv.predict(X_test)

## To print the output of predictions for Decision tree classifier, uncomment the following :
# for i in y_pred_dt:
#     if(i==1):
#         print("Y")
#     else:
#         print("N")

## To print the output of predictions for SVM classifier, uncomment the following :
# for i in y_pred_svm:
#     if(i==1):
#         print("Y")
#     else:
#         print("N")

# ## See the accuracy for the 2 classifiers
print("Accuracy for Decision Tree Classifier is ", accuracy_score(y_test,y_pred_dt)*100)
print("\n")
print("Accuracy for SVM Classifier is ", accuracy_score(y_test,y_pred_svm)*100)
pr_dt=precision_recall_fscore_support(y_test, y_pred_dt, average='macro')

## Another metric - F1 : To print F1, uncomment the following code for Decision Tree
print("\n")
print("Precison of Decision Tree Classifier = ",pr_dt[0])
print("Recall of Decision Tree classifier = ",pr_dt[1])
print("F1 of Decision tree classifier = ",pr_dt[2])


## Another metric - F1 : To print F1, uncomment the following code for SVM

pr_svm=precision_recall_fscore_support(y_test, y_pred_svm, average='macro')
print("\n")
print("Precison of SVM Classifier = ",pr_svm[0])
print("Recall of SVM classifier = ",pr_svm[1])
print("F1 of SVM classifier = ",pr_svm[2])

#
## Cross Validation code where there are 10 folds
scores_dt= cross_val_score(dt, X_train,y_train , cv=10)
scores_sv= cross_val_score(sv, X_train,y_train , cv=10)
#
#
# ## To see the accuracy in each of the 10 folds in Decision tree Classifier, uncomment the following code
# for i in scores_dt:
#     print(i)
# #
# ## To see the accuracy in each of the 10 folds in SVM Classifier, uncomment the following code
# for i in scores_sv:
#     print(i)
#
print("The mean of the 10 Cross validations (accuracy)on Decistion tree classifiers is : ",scores_dt.mean()*100)
print("\n")
print("The mean of the 10 Cross validations (accuracy)on SVM classifier is : ",scores_sv.mean()*100)
print("\n")
print("Thank you for your patience ")
print("DONE !")

