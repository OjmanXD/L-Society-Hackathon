import numpy as np
import pandas as pd
import spacy_sentence_bert
from tensorflow import keras
from flask import Flask, request, jsonify, render_template, redirect, url_for, session
import pickle
import json

import pathlib
temp = pathlib.PosixPath
pathlib.PosixPath = pathlib.WindowsPath

empusers = {
    'kunal': '1234',
    'user2': 'password2'
}

itusers = {
    'manoj': '1234',
    'enar': 'password2'
}

nlp = spacy_sentence_bert.load_model('en_stsb_roberta_large')
infile = open('sub_category_1_labelmap.pkl', 'rb')
sub_category_1_labelmap = pickle.load(infile)
infile.close()
infile = open('sub_category_2_labelmap.pkl', 'rb')
sub_category_2_labelmap = pickle.load(infile)
infile.close()
infile = open('sub_category_3_labelmap.pkl', 'rb')
sub_category_3_labelmap = pickle.load(infile)
infile.close()
model_sub_cat1 = keras.models.load_model('./sub_category_1_model/')
model_sub_cat2 = keras.models.load_model('./sub_category_2_model/')
model_sub_cat3 = keras.models.load_model('./sub_category_3_model/')

pkl_filename = "Dataset.pkl"
with open(pkl_filename, 'rb') as file:
    df=pd.read_pickle(file)
    
    
def get_reason(vector):
    prediction = model_sub_cat1.predict(np.array([vector]))
    class_name = list(sub_category_1_labelmap.keys())
    sub_cat_1 = class_name[np.argmax(prediction[0])]

    prediction = model_sub_cat2.predict(np.array([vector]))
    class_name = list(sub_category_2_labelmap.keys())
    sub_cat_2 = class_name[np.argmax(prediction[0])]

    prediction = model_sub_cat3.predict(np.array([vector]))
    class_name = list(sub_category_3_labelmap.keys())
    sub_cat_3 = class_name[np.argmax(prediction[0])]

    return (sub_cat_1, sub_cat_2, sub_cat_3)

questions = []
categories = []

app = Flask(__name__)
app.secret_key = 'manojkumar'

@app.route('/')
def home():
    return render_template('Employee Login.html')

@app.route('/login', methods = ['POST'])
def login():
    uname = request.form['uname']
    password = request.form['pass']
    if uname in empusers and empusers[uname] == password:
        return render_template('employee.html')
    elif uname in itusers and itusers[uname] == password:
        return redirect(url_for('it'))

@app.route('/employee', methods = ['POST'])
def employee():
    problem_statement = request.form['problem']
    cats = get_reason(nlp(problem_statement).vector)
    session['problem'] = problem_statement
    session['cats'] = cats
    return render_template('success.html')

@app.route('/it')
def it():
    d = session.pop('cats', None)
    p = session.pop('problem', None)
    data = [{"Problem":p, "Category_1":d[0], "Category_2":d[1], "Category_3":d[2]}]
    return render_template('it.html', data=data)
# @app.route('/emp', methods=['POST'])
# def emp_page():
    
# @app.route('/it', methods=['POST'])
# def it_page():

if __name__ == "__main__":
    app.run(debug=True)