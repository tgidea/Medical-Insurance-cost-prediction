from flask import Flask,render_template,request,redirect
from flask_cors import CORS,cross_origin
import pickle
import pandas as pd
import numpy as np

app=Flask(__name__)
cors=CORS(app)
model=pickle.load(open('LinearRegression.pkl','rb'))
insurance=pd.read_csv('insurance.csv')

@app.route('/',methods=['GET','POST'])
def index():
    return render_template('index.html')


@app.route('/predict',methods=['POST'])
@cross_origin()
def predict():

    age=request.form.get('age')
    bmi=request.form.get('bmi')
    child=request.form.get('child')
    region=request.form.get('region')
    gender=request.form.get('gender')
    smoker = request.form.get('smoker')
    
    print(age,gender,bmi,child,smoker,region)
    data = np.array([[age,gender,bmi,child,smoker,region]])
    prediction=model.predict(data)
   
    return str(abs(np.round(prediction[0],2)))



if __name__=='__main__':
    app.run()