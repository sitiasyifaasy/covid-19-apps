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
from dotenv import load_dotenv

# Disable CUDA
# os.environ['CUDA_VISIBLE_DEVICES'] ="0"
# Disable Warning oneDNN and AVX AVX2
# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# ENV
dotenv_path = os.path.join(os.path.dirname(__file__), '.env')  # Path to .env file
load_dotenv(dotenv_path)


# GLOBAL VARIABLES
SECRET_KEY = os.urandom(32)
PATH = os.getcwd()
N_CLASSES = 4

# Apps init
app = Flask(__name__)
app.config["TEMPLATES_AUTO_RELOAD"] = True
app.config['SECRET_KEY'] = SECRET_KEY

# CSRF Token
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

def preprocessing_img(img):
    # convert the color from BGR to RGB then convert to PIL array
    im_cvt = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    im_pil = Image.fromarray(im_cvt)

    # resize the array (image) then PIL image
    im_resized = im_pil.resize((224, 224))
    img_array = preprocessing.image.img_to_array(im_resized)
    # expand dimension to 4D because the model expects this format (1, 224, 224, 3)
    image_array_expanded = np.expand_dims(img_array, axis = 0)
    return image_array_expanded

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/prediction/",  methods = ['GET', 'POST'])
def predict():
    if request.method == "POST":
        # Input Citra image stream
        images = request.files['images']
        image_stream = images.stream.read()
        print(image_stream)
        # Convert ke base64
        data = base64.b64encode(image_stream)
        # Decode base64 ke array
        data = decode_base64(data)
        im_arr = np.fromstring(data, dtype=np.uint8)
        img = cv2.imdecode(np.array(im_arr), cv2.IMREAD_UNCHANGED)
        size_awal = img.shape
        print("Input Citra Awal : ", img.shape)
        
        cv2_image = preprocessing_img(img)
        print(cv2_image.shape)
        size_akhir = cv2_image.shape[1:-1]

        # tampil resize citra 224x224 untuk di html
        resize = cv2.resize(img, (224, 224))
        resize_en = cv2.imencode('.jpg', resize)[1].tobytes()
        resize_base64 = base64.b64encode(resize_en).decode('utf-8')

        # Load Model
        load_model = loaded_model()      
        result = load_model.predict([cv2_image])
        preds = np.argmax(result, axis= -1)
        label = "Normal" if preds[0] == 0 else "Covid-19" if preds[0] == 1 else "Lung Opacity" if preds[0] == 2 else "Viral Pneumonia"
        print(label)
        return jsonify({
            'status': 'success',
            'label': str(label),
            'prediksi': str(preds[0]),
            'size_awal': str(size_awal),
            'size_akhir': str(size_akhir),
            'image_resize': "data:image/png;base64," + resize_base64
        })
    else:
        return "Hayoh"


if __name__ == "__main__":
    app.run(debug=True)
    
