from __future__ import division, print_function
import sys
import os
import glob
import re
import numpy as np

# Keras
from keras.applications.imagenet_utils import preprocess_input, decode_predictions
from keras.models import load_model
from keras.preprocessing import image

# Flask utils
from flask import Flask, redirect, url_for, request, render_template
from werkzeug.utils import secure_filename
    
# Define a flask app
app = Flask(__name__)

# Model saved with Keras model.save()
MODEL_PATH = './classTumor2.h5'

model = load_model(MODEL_PATH)


labels = ['Glioma_Tumor','No_Tumor','Meningioma_Tumor','Pituitary_Tumor']

def model_predict(img_path, model):
    img = image.load_img(img_path)
    img = image.smart_resize(img, (150,150))

    x = image.img_to_array(img)
    x = np.true_divide(x, 255)
    x = np.expand_dims(x, axis=0)


    preds = model.predict(x)
    return preds


@app.route('/', methods=['GET'])
def index():
    # Main page
    return render_template('index.html')


@app.route('/predict', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        # Get the file from post request
        f = request.files['file']

        # Save the file to ./uploads
        basepath = os.path.dirname(__file__)
        file_path = os.path.join(
            basepath, 'uploads', secure_filename(f.filename))
        f.save(file_path)

        preds = model_predict(file_path, model)

        pred_class = np.argmax(preds) 
        print(pred_class, preds)        
        result = str(labels[pred_class])
        print(result)           
        return result
    return None


if (__name__) == ('__main__'):
    app.run(port=(3000), debug=True)