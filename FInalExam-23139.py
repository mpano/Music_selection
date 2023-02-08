import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import joblib


dataf = pd.read_excel('DS.xlsx')
input("View Dataset")
print(dataf)



input ("Press enter for replacing empty cells with the median")

dataf["AGE "].fillna(dataf["AGE "].median(axis=0),inplace=True)
dataf["GENDER"].fillna(dataf["GENDER"].median(axis=0),inplace=True)
dataf["DISTRICT NUMBER"].fillna(dataf["DISTRICT NUMBER"].median(axis=0),inplace=True)

print (dataf.to_string())

input("Press enter to highlight the duplicates")

print(dataf.duplicated())

input("Press enter for Removing  the duplicate")
dataf.drop_duplicates(inplace = True)
print(dataf.to_string)

X=dataf.drop (columns=['S/N', 'DISTRICT NAME', 'ARTIST '])
y=dataf['ARTIST ']
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.3)

#Create a Decision Tree, Logistic Regression, Suport Vector Machine  and Random Forest Classifiers
input("Press enter for create classifiers")
Decision_tree_model= DecisionTreeClassifier()
Logistic_regression_Model=LogisticRegression(solver='lbfgs',max_iter=10000)
SVM_model=svm.SVC(kernel='linear')
RF_model=RandomForestClassifier(n_estimators=100)

#Train the models using the training sets
input("Press enter to train the model")
Decision_tree_model.fit(X_train, y_train)
Logistic_regression_Model.fit(X_train, y_train)
SVM_model.fit(X_train, y_train)
RF_model.fit(X_train, y_train)

#Predict the response for test dataset
input("Press enter to test the model")
DT_Prediction =Decision_tree_model.predict(X_test)
LR_Prediction =Logistic_regression_Model.predict(X_test)
SVM_Prediction =SVM_model.predict(X_test)
RF_Prediction =RF_model.predict(X_test)

# Calculation of Model Accuracy
input("Press enter to calculate the accuracy")
DT_acc=accuracy_score(y_test, DT_Prediction)
LR_acc=accuracy_score(y_test, LR_Prediction)
SVM_acc=accuracy_score(y_test, SVM_Prediction)
RF_acc=accuracy_score(y_test, RF_Prediction)

input("Press enter to print the accuracy data...")

print ("Decistion Tree accuracy =", DT_acc * 100,"%")
print ("Logistic Regression accuracy =", LR_acc * 100,"%")
print ("Suport Vector Machine accuracy =", SVM_acc * 100,"%")
print ("Random Forest accuracy =", RF_acc * 100,"%")


input("Press enter to persist the model...")
prediction=Decision_tree_model.predict(X_test)
joblib.dump(Decision_tree_model,'mpano.joblib')

input("Finding the best Artist According to district")
persistedModel=joblib.load('music.joblib')
print(" ")
print(" ARTIST PREDICTION SYSTEM")
print("****************************")
print(" ")
age=int (input ("Enter Your Age:"))
gender= int(input ("Enter your gender (i.e '1' for male, '0' for female): "))
district= int(input ("Enter your district number (i.e '21' for NYARUGENGE, '22' for KICUKIRO & '23' for GASABO): "))
prediction = persistedModel.predict ([[age,gender,district]])
print(" ")
print("The best artist is:", prediction[0])
print(" ")



