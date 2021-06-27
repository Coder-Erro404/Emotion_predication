import tkinter as tk
from tkinter import Button, Entry, Label, Message ,Tk
from tkinter.constants import CHAR
import pandas as pd
import numpy as np
import seaborn as sns
import tkinter as tk
import neattext.functions as nfx
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

def Done():
        df = pd.read_csv(r"emotion_dataset_2.csv")
        df.head()
        var = df.columns
        #df.info()
        df['Emotion'].value_counts()
        sns.countplot(x='Emotion',data=df)
        dir(nfx)
        df['Clean_Text'] = df['Text'].apply(nfx.remove_userhandles)
        df['Clean_Text'] = df['Clean_Text'].apply(nfx.remove_stopwords)
        df.isna().sum()
        Xfeatures = df['Clean_Text']
        ylabels = df['Emotion']
        x_train,x_test,y_train,y_test = train_test_split(Xfeatures,ylabels,test_size=0.3,random_state=42)
        pipe_lr = Pipeline(steps=[('cv',CountVectorizer()),('lr',LogisticRegression())])
        pipe_lr.fit(x_train,y_train)
        pipe_lr.score(x_test,y_test)
        exe5 = str(e1.get())
        p=pipe_lr.predict([exe5])
        resuult.configure(text=p)

def clean():
    e1.delete(0,'end')
    resuult.configure(text='result:')

w1 = tk.Tk()
w1.title("Human Emotion recognition")


l1=Label(w1,text='enter Text')
e1=Entry(w1, width=12)

b1=Button(w1,text='Done',command=Done) 
b2=Button(w1,text='clean',command=clean)

resuult=Message(w1,text='resuult:',width=300)

l1.grid(row=0, column=0)
e1.grid(row=0, column=1)
b1.grid(row=1, column=0)
b2.grid(row=1, column=1)
resuult.grid(row=2, column=0 ,columnspan=2)

w1.mainloop()