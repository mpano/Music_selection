import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

dataset=pd.read_excel('QUIZ4L1.xlsx')

input("View Dataset")

print(dataset)
input("Press Enter to replace empty cells with median at axis=0")
x = dataset.median()
dataset.fillna(x, inplace = True, axis=0)
print(dataset.to_string())

input("Press enter to view duplicates")
print(dataset.duplicated())
input("Press Enter to delete duplicates")
dataset.drop_duplicates(inplace = True)
print(dataset.to_string())

input("Press enter to calculate accuracy of models")
#Finding the accuracy of models
datafeatures = ['age','gender']
X=dataset[datafeatures]
y=dataset['genre']
X_train, X_test, y_train, y_test=train_test_split (X,y, test_size=0.3)

#Create a Decision Tree, Logistic Regression, Suport Vector Machine  and Random Forest Classifiers

Decision_tree_model= DecisionTreeClassifier()
Logistic_regression_Model=LogisticRegression(solver='lbfgs',max_iter=10000)
SVM_model=svm.SVC(kernel='linear')
RF_model=RandomForestClassifier(n_estimators=100)

#Train the models using the training sets

Decision_tree_model.fit(X_train, y_train)
Logistic_regression_Model.fit(X_train, y_train)
SVM_model.fit(X_train, y_train)
RF_model.fit(X_train, y_train)

#Predict the response for test dataset

DT_Prediction =Decision_tree_model.predict(X_test)
LR_Prediction =Logistic_regression_Model.predict(X_test)
SVM_Prediction =SVM_model.predict(X_test)
RF_Prediction =RF_model.predict(X_test)

# Calculation of Model Accuracy

DT_score=accuracy_score(y_test, DT_Prediction)
lR_score=accuracy_score(y_test, LR_Prediction)
SVM_score=accuracy_score(y_test, SVM_Prediction)
RF_score=accuracy_score(y_test, RF_Prediction)

# Display Accuracy
print ("Decistion Tree accuracy =", DT_score*100,"%")
print ("Logistic Regression accuracy =", lR_score*100,"%")
print ("Suport Vector Machine accuracy =", SVM_score*100,"%")
print ("Random Forest accuracy =", RF_score*100,"%")
