from flask import Flask,render_template,request
import numpy as np
import pickle

model = pickle.load(open('data.pkl', 'rb'))

app = Flask(__name__)

@app.route("/")
def home():
    return render_template('diabeties.html')

@app.route("/predict",methods=['post'])
def predict():
    data1 = request.form['Pregnancies']
    data2 = request.form['Glucose']
    data3 = request.form['BloodPressure']
    data4 = request.form['SkinThickness']
    data5 = request.form['Insulin']
    data6 = request.form['BMI']
    data7 = request.form['DiabetesPedigreeFunction']
    data8 = request.form['Age']
    arr = np.array([[data1, data2, data3, data4,data5,data6,data7,data8]])
    pred = model.predict(arr)
    if pred == 0:
        pred = 'Diabetes Patient'
    else:
        pred = 'Normal Patient'
    return render_template('prediction diabetes.html', data=pred)
if __name__ == "__main__":
    app.run(debug = True)