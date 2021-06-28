# Required Libraries
from flask import Flask, render_template, request, make_response
import jsonify
import requests
import json
from requests.sessions import Request
import pickle
import numpy as np


# Importing the model
model = pickle.load(open('xgb_fmodel.pkl','rb'))
cluster = pickle.load(open('knn_classifier.pkl','rb'))
scaler = pickle.load(open('scaler.pkl','rb'))


app = Flask(__name__)


# Templates
# Home page
@app.route('/',methods=['GET'])
def Home():
    return render_template('index.html')



@app.route("/to_model", methods=['POST'])
def to_model():

    req = request.get_json()
    array = req['val_array']
    
    input_array = np.array([array])
    prediction = model.predict(input_array)

    output=prediction[0]

    outs = "ERROR"
    if output==0:
        outs = "LOW"
    elif output==1:
        outs = "AVERAGE"
    elif output==2:
        outs = "HIGH"

    x = {"output": outs}
    y = json.dumps(x)

    return y



@app.route("/to_cluster", methods=['POST'])
def to_cluster():

    req = request.get_json()
    array = req['val_array2']
    
    input_array = np.array([array])
    
    scaled = scaler.transform(input_array)
    prediction = cluster.predict(scaled)
    output = prediction[0]

    outs = "ERROR"
    if output==0:
        outs = "LOW"
    elif output==1:
        outs = "AVERAGE"
    elif output==2:
        outs = "HIGH"

    x = {"output": outs}
    y = json.dumps(x)

    return y



if __name__=="__main__":
    app.run(debug=True)

