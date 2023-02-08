import pandas as pd
from sklearn.tree import DecisionTreeClassifier
import joblib
model=joblib.load('Music is a cultural')
age=int (input ("Enter your age :"))
gender= int(input ("Enter your gender ie 0 for 'female' and 1 for 'male': "))
predictions = model.predict ([[age,gender]])
print("The genre that suits you is:",predictions)

import pandas as pd
from sklearn.tree import DecisionTreeClassifier
import joblib
model=joblib.load('Music is a cultural')
age=int (input ("Enter your age :"))
gender= int(input ("Enter your gender ie 0 for 'female' and 1 for 'male': "))
predictions = model.predict ([[age,gender]])
print("The genre that suits you is:",predictions)

