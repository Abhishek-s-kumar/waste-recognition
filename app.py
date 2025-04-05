from flask import Flask, request, render_template
import cv2
import numpy as np
from tensorflow.keras.models import load_model

app = Flask(__name__)
model = load_model('model/waste_classifier.h5')
classes = ['cardboard', 'glass', 'metal', 'paper', 'plastic', 'trash']
#Load Model and Initialize Flask
@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        file = request.files['file']
        img = cv2.imdecode(np.fromstring(file.read(), np.uint8), cv2.IMREAD_COLOR)
        img = cv2.resize(img, (224, 224)) / 255.0
        pred = model.predict(img[np.newaxis, ...])
        return f'Predicted: {classes[np.argmax(pred)]}'
    return render_template('index.html')
#Define Route for Uploading Images
if __name__ == '__main__':
    app.run(debug=True)