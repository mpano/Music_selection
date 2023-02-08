import tkinter as t
from sklearn.tree import DecisionTreeClassifier
import joblib
home=t.Tk()
home.geometry('500x400')

t.Label(text='Enter Your Age: ').place(x=50,y=50)
t.Label(text='Enter Your Gender: ').place(x=50,y=100)
t.Label(text='choose 1 for male and 0 for female').place(x=180,y=80)


age=t.Entry(width=23)
age.place(x=180,y=50)

gender=t.Entry(width=23)
gender.place(x=180,y=100)

def clickonbtn():
    model=joblib.load('Music is a cultural')
    precide=model.predict([[age.get(),gender.get()]])
    p='the genre is '+precide
    t.Label(text=p).place(x=160,y=180)


button1=t.Button(text="PREDICT",width=25,bg='#999',fg='#000',command=clickonbtn)
button1.place(x=160,y=140)
home.mainloop()