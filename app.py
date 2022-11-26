import pickle
import numpy as np
from flask import Flask, render_template, request

model = pickle.load(open('dt_model.pkl','rb'))

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        init_features = [float(x) for x in request.form.values()]
        final_features = [np.array(init_features)]
        pred = model.predict(final_features)
        if pred == 'Iris-setosa':
            prediction = 'Iris-setosa'
        elif pred == 'Iris-versicolor':
            prediction = 'Iris - versicolor'
        else:
            prediction = 'Iris-virginica '
        return render_template('index.html', prediction_text = 'Predicted class: {}'.format(prediction))

if __name__ == '__main__':
    app.run()