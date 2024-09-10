import pickle
import pandas as pd
from flask import Flask, request, render_template

app = Flask(__name__)

# Load the trained model and scaler
model = pickle.load(open('best_model.pkl', 'rb'))
scaler = pickle.load(open('scaler.pkl', 'rb'))

@app.route('/')  # Route to display the home page
def index():
    return render_template('index.html')  # Rendering the home pagep

@app.route('/predict', methods=["POST", "GET"])
def predict():
    return render_template('predict.html')

@app.route('/output', methods=["POST", "GET"])  # Route to show the prediction
def output():
    # Reading the inputs given by the user
    input_features = {
        'Date': [float(request.form['Date'])],
        'Location': [float(request.form['Location'])],
        'MinTemp': [float(request.form['MinTemp'])],
        'MaxTemp': [float(request.form['MaxTemp'])],
        'Rainfall': [float(request.form['Rainfall'])],
        'WindGustSpeed': [float(request.form['WindGustSpeed'])],
        'WindSpeed9am': [float(request.form['WindSpeed9am'])],
        'WindSpeed3pm': [float(request.form['WindSpeed3pm'])],
        'Humidity9am': [float(request.form['Humidity9am'])],
        'Humidity3pm': [float(request.form['Humidity3pm'])],
        'Pressure9am': [float(request.form['Pressure9am'])],
        'Pressure3pm': [float(request.form['Pressure3pm'])],
        'Temp9am': [float(request.form['Temp9am'])],
        'Temp3pm': [float(request.form['Temp3pm'])],  # Corrected from Temp3am to Temp3pm
        'RainToday': [float(request.form['RainToday'])],
        'WindGustDir': [float(request.form['WindGustDir'])],
        'WindDir9am': [float(request.form['WindDir9am'])],
        'WindDir3pm': [float(request.form['WindDir3pm'])]}  # Corrected from WindDir3am to WindDir3pm

    # Creating a DataFrame from the input features
    data = pd.DataFrame(input_features)

    # Perform any necessary data preprocessing
    numeric_cols = [
        'Date', 'Location','MinTemp', 'MaxTemp', 'Rainfall', 'WindGustSpeed', 'WindSpeed9am',
        'WindSpeed3pm', 'Humidity9am', 'Humidity3pm', 'Pressure9am', 'Pressure3pm',
        'Temp9am', 'Temp3pm',"RainToday", "WindGustDir", 'WindDir9am', 'WindDir3pm'
    ]

    # Make sure numeric columns are numeric
    data[numeric_cols] = data[numeric_cols].apply(pd.to_numeric)

    # Scale the input dasssssssssta using the loaded scaler
    data[numeric_cols] = scaler.transform(data[numeric_cols])

    # Make prediction using the loaded model
    prediction = model.predict(data)

    if prediction[0] == 1:
        return render_template("chance.html")  # Render a template indicating chance of rain
    else:
        return render_template("nochance.html")  # Render a template indicating no chance of rain

if __name__ == "__main__":
    app.run(debug=True)
