import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle
import math

app = Flask(__name__)
model1= pickle.load(open('taxi.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    #All the values coming in from forms will be stored in this int_features list
    int_features = [int(x) for x in request.form.values()]
    
    #Converting this list into array
    final_features = [np.array(int_features)]

    prediction = model1.predict(final_features)
    output = round(prediction[0],2)

    return render_template('index.html', prediction_text='Number of weekly rides should be {}'.format(math.floor(output)))

if __name__ == '__main__':
    app.run(debug=True)
