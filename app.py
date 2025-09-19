
from flask import Flask, render_template, request
import numpy as np
from src.pipeline import predict_pipeline
from src.pipeline.predict_pipeline import CustomData, get_data_as_dataframe

app = Flask(__name__)

@app.route('/', methods=['GET', "POST"])
def home():
    return render_template('index.html')

@app.route('/predict', methods=['GET', "POST"])
def predict():
    form = request.form
    df = get_data_as_dataframe(CustomData(form))
    
    prediction = predict_pipeline.predict(df)[0]

    return render_template('index.html', prediction=int(np.exp(prediction)))

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080, )