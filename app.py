
from flask import Flask, render_template, Response
from camera import VideoCamera
import tensorflow.compat.v1 as tf
from PIL import Image
from numpy import asarray
import numpy as np



app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

def gen(camera):
    while True:
        frame = camera.get_frame()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')

@app.route('/video_feed')
def video_feed():
    return Response(gen(VideoCamera()),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    # facenet_model = tf.keras.models.load_model('facenet_keras.h5')
    # print('Loaded Model')
    app.run(host='0.0.0.0', debug=True)
