from flask import Flask, render_template, request, render_template_string, jsonify
from flask_wtf.csrf import CSRFProtect
import cv2
from PIL import Image
import tensorflow as tf
import tensorflow_ranking as tfr
from tensorflow.keras import models, preprocessing

import base64
import numpy as np
import os
import re

SECRET_KEY = os.urandom(32)
PATH = os.getcwd()
N_CLASSES = 4

app = Flask(__name__)
app.config["TEMPLATES_AUTO_RELOAD"] = True
app.config['SECRET_KEY'] = SECRET_KEY

CSRFProtect(app)

def loaded_model():
    load_model = models.load_model(PATH + "/model/model_adam3.h5")
    load_model.compile(tf.keras.optimizers.Adam(learning_rate=0.001), 
            loss = tf.keras.losses.CategoricalCrossentropy(from_logits = False), 
           metrics = ['accuracy', tf.keras.metrics.CategoricalAccuracy(), tfr.keras.metrics.MeanAveragePrecisionMetric(topn=N_CLASSES)])
    return load_model


def decode_base64(data, altchars=b'+/'):
    """Decode base64, padding being optional.

    :param data: Base64 data as an ASCII byte string
    :returns: The decoded byte string.

    """
    data = re.sub(rb'[^a-zA-Z0-9%s]+' % altchars, b'', data)  # normalize
    missing_padding = len(data) % 4
    if missing_padding:
        data += b'=' * (4 - missing_padding)
    return base64.b64decode(data, altchars)

def prepare_image2 (img):
    # convert the color from BGR to RGB then convert to PIL array
    cvt_image =  cv2.imdecode(np.array(im_arr), cv2.COLOR_BGR2RGB)
    im_pil = Image.fromarray(cvt_image)

    # resize the array (image) then PIL image
    im_resized = im_pil.resize((224, 224))
    img_array = preprocessing.image.img_to_array(im_resized)
    image_array_expanded = np.expand_dims(img_array, axis = 0)
    return image_array_expanded

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/prediction/",  methods = ['GET', 'POST'])
def predict():
    if request.method == "POST":
        # Input Citra dari Base64
        images = request.files['images']
        image_stream = images.stream.read()
        data = base64.b64encode(image_stream)
        data = decode_base64(data)
        im_arr = np.fromstring(data, dtype=np.uint8)
        img = cv2.imdecode(np.array(im_arr), cv2.COLOR_BGR2RGB)
        print("Input Citra Awal : ", img.shape)
        
        cv2_image = prepare_image2 (img)
        print(cv2_image.shape)
        # # Resize Citra
        # img_resize = cv2.resize(img, (224, 224))
        # print("Resize : ", img_resize.shape)

        # # Load Model
        # load_model = loaded_model()
        # # img_resize = np.asarray(img_resize)
        # im_pil = Image.fromarray(img_resize)
        # img_array = preprocessing.image.img_to_array(im_pil)
        # img_final = np.expand_dims(img_resize,axis=0)
        # print(img_final.shape)

        # result = load_model.predict([img_final])
        # preds = np.argmax(result, axis= -1)
        # print(preds)
        return jsonify({
                'msg': 'success'
           })
    else:
        return "Hayoh"


if __name__ == "__main__":
    app.run(debug=True)
    
