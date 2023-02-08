import pandas as pd
from sklearn.tree import DecisionTreeClassifier
import joblib


dataset=pd.read_excel('QUIZ4L1.xlsx')
datafeatures = ['age','gender']
X=dataset[datafeatures]
y=dataset['genre']
model= DecisionTreeClassifier()
model.fit(X.values, y)

# Create a persistent Model

joblib.dump(model, 'Music is a cultural')
