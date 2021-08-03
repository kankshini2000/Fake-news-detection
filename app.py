# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

from flask import Flask, render_template, request
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
import pickle
import pandas as pd
from sklearn.model_selection import train_test_split

app = Flask(__name__)
count_vectorizer = CountVectorizer(ngram_range=(1, 2), stop_words='english') 
loaded_model= pickle.load(open('model.pkl','rb'))
train=pd.read_csv(r"C:\Users\User\Seminar project\train.csv")
test=pd.read_csv(r"C:\Users\User\Seminar project\test.csv")
test['total']=test['title']+' '+test['author']+' '+test['text']
train['total']=train['title']+' '+train['author']+' '+train['text']
X_train, X_test, y_train, y_test = train_test_split(train['total'], train.label, test_size=0.20, random_state=0)

def fake_news_det1(news):
    count_train = count_vectorizer.fit_transform(X_train)
    count_test = count_vectorizer.transform(X_test)
    input_data = [news]
    logreg1 = LogisticRegression(C=1e5)
    logreg1.fit(count_train, y_train)
    vectorized_input_data = count_vectorizer.transform(input_data)
    prediction = logreg1.predict(vectorized_input_data)
    return prediction

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        message = request.form['message']
        pred = fake_news_det1(message)
        print(pred)
        return render_template('index.html', prediction=pred)
    else:
        return render_template('index.html', prediction="Something went wrong")

if __name__ == '__main__':
    app.run(debug=True)