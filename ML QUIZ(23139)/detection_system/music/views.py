from django.shortcuts import render,redirect
from django.http import HttpResponse
# Create your views here.

def home(request):
    return render(request,"index.html")

def result(request):
    return render(request,"result.html")


import pandas as pd 
import joblib
model = joblib.load('./model/Music is a cultural')

def predict(request):
    if request.method=='POST':
        temp={}
        temp['age'] = int(request.POST.get('age'))
        temp['gender'] = int(request.POST.get('gender'))
        testData = pd.DataFrame({'x':temp}).transpose()
        feedback=model.predict(testData)[0]

        return render(request , 'result.html' , {'result': feedback})