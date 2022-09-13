from flask import Flask, render_template, request
import torch
import cv2
import os
from model_functions import predict

BASE_PATH = os.getcwd()
UPLOAD_PATH = os.path.join(BASE_PATH, 'static/in')
OUT_PATH = os.path.join(BASE_PATH, 'static/pred')
app = Flask(__name__)

model = torch.hub.load('ultralytics/yolov5', 'custom', 'best.pt')


@app.route('/')
def index():
    return render_template('ocr.html', name="")


@app.route('/submitted', methods=['POST', 'GET'])
def profile_ocr():
    if request.method == 'POST':
        upload_file = request.files['image_name']
        out_filename = upload_file.filename
        in_file_path = os.path.join(UPLOAD_PATH, out_filename)
        upload_file.save(in_file_path)

        bgr, rgb, text = predict(in_file_path)
        cv2.imwrite(os.path.join(OUT_PATH, out_filename), bgr)
        text_dict = {'Number plate': text}
        return render_template('ocr.html', upload=True, upload_image=out_filename, text=text_dict)

    return render_template('ocr.html')


if __name__ == '__main__':
    app.run(debug=True)
