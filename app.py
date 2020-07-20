import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle
from sklearn.svm import SVC

app = Flask(__name__)

# load the model from disk

# REg
model = pickle.load(open('model.pkl', 'rb'))

# Classification
clf = pickle.load(open('nlp_model.pkl', 'rb'))
cv = pickle.load(open('tranform.pkl', 'rb'))
#for cancer
s1 = pickle.load(open('model_cancer_new1.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')


@app.route('/email')
def email():
    return render_template('home.html')

@app.route('/cancer')
def cancer():
    return render_template('cancer.html')

@app.route('/predict', methods=['POST'])
def predict():
    '''
    For rendering results on HTML GUI
    '''
    print(request.form['test_score'])
    format = request.args.get('format')
    print("here", format)
    print("new line")

    int_features = [int(x) for x in request.form.values()]
    final_features = [np.array(int_features)]
    prediction = model.predict(final_features)

    output = round(prediction[0], 2)

    if(format == 'json'):
        return jsonify({'salary': output})

    return render_template('index.html', prediction_text='Employee Salary should be $ {}'.format(output))


@app.route('/predict_email', methods=['POST'])
def predict_email():

    if request.method == 'POST':
        message = request.form['message']
        data = [message]
        vect = cv.transform(data).toarray()
        my_prediction = clf.predict(vect)
    return render_template('result.html', prediction=my_prediction)

@app.route('/predict_cancer', methods=['POST'])
def predict_cancer():
    '''
    For rendering results on HTML GUI
    '''
    '''
    print(request.form['test_score'])
    format = request.args.get('format')
    print("here", format)
    print("new line")
    '''

    int_features = [float(x) for x in request.form.values()]
    final_features = [np.array(int_features)]
    prediction = s1.predict(final_features)
    print("prediction: ", prediction)
    output = round(prediction[0], 2)
    print("output: ", output)

    if (str(output) == "0"):
        val = "malignant"
    else:
        val = "benign"

    #if(format == 'json'):
       # return jsonify({'salary': output})

    return render_template('cancer.html', prediction_text='The state is {}'.format(val))


if __name__ == "__main__":
    app.run(debug=True)  # auto-reload on code change
