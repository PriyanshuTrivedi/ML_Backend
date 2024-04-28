from flask import Flask,request,jsonify
from flask_cors import CORS, cross_origin
from detect_heart import *
import numpy as np

app = Flask(__name__)
CORS(app)

@app.route('/',methods=["GET"])
def home():
    return "Heart Detect ML Backend"

# @app.route('/results',methods=["GET","POST"])
# def main_api():
#     return detectHeart(Request.data)

@app.route('/results',methods=["GET","POST"])
def main_api():
    data=request.get_json()
    print(data)
    result=detectHeart(data)[0]
    result=result.astype(int).tolist()
    return jsonify({'res':result})

if __name__=="__main__":
    app.run(debug=True)
