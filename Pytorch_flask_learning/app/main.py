from flask import Flask , request, jsonify

app = Flask(__name__)

@app.route('/predict',methods=['POST'])
def predict():
    # return pass

    return jsonify({'result':1})