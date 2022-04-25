import json
import os
import base64
import logging
import gc
import torch
import torch.nn as nn

from PIL import Image
import numpy as np

from flask import Flask, g, request, send_from_directory
from flask_cors import CORS
from obj_detection_yolo.object_detection_yolo import *


class PretrainedCNN(nn.Module):
    def __init__(self, hidden_size, num_classes):
        super(PretrainedCNN, self).__init__()
        self.resnet = models.resnet18(pretrained=True)  # TODO
        self.resnet.fc = nn.Linear(
            self.resnet.fc.in_features, hidden_size)  # TODO
        self.fc = nn.Linear(hidden_size, num_classes)

        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.2)

    def forward(self, images):
        resnet_features = self.dropout(self.relu(self.resnet(images)))

        return self.fc(resnet_features)


modelResnet = torch.load('./model/model.pt', map_location=torch.device('cpu'))

UPLOAD_FOLDER = 'uploaded_images'


dct = {
    0: {'name': 'Apple', 'calories': 52},
    1: {'name': 'Banana', 'calories': 89},
    2: {'name': 'Bean', 'calories': 347},
    3: {'name': 'Bread', 'calories': 265},
    4: {'name': 'Carrot', 'calories': 41},
    5: {'name': 'Cheese', 'calories': 402},
    6: {'name': 'Cucumber', 'calories': 45},
    7: {'name': 'Egg', 'calories': 155},
    8: {'name': 'Grape', 'calories': 67},
    9: {'name': 'Grape & Apple', 'calories': 119},
    10: {'name': 'Onion', 'calories': 40},
    11: {'name': 'Orange', 'calories': 47},
    12: {'name': 'Pasta', 'calories': 131},
    13: {'name': 'Pepper', 'calories': 40},
    14: {'name': 'Qiwi', 'calories': 61},
    15: {'name': 'Tomato', 'calories': 18},
    16: {'name': 'Watermelon', 'calories': 30},
    17: {'name': 'sauce', 'calories': 29}
}

this_app = Flask(__name__)
this_app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
CORS(this_app)
APP_ROOT = os.path.dirname(os.path.abspath(__file__))

# log_print("loading net...")
# net, classes = net_init()
# log_print("net loaded")


@this_app.route('/')
def return_html():
    return send_from_directory('', 'index.html')


@this_app.route('/js')
def return_js():
    return send_from_directory('js', "data_worker.js")


@this_app.route('/favicon.ico')
def favicon():
    return send_from_directory(os.path.join(this_app.root_path, 'uploaded_images'),
                               'favicon.ico', mimetype='image/vnd.microsoft.icon')


@this_app.route('/uploaded_images/<filename>')
def return_processed(filename):
    return send_from_directory(this_app.config['UPLOAD_FOLDER'],
                               filename)


@this_app.route('/handle_img', methods=['POST'])
def handle_img():
    gc.collect()
    target = os.path.join(APP_ROOT, 'uploaded_images')

    # create image directory if not found
    if not os.path.isdir(target):
        os.mkdir(target)

    log_print("start")
    if 'image' not in request.files:
        resp = json.dumps({'message': 'No file part in the request'})
        resp.status_code = 400
        return resp

    img_file = request.files['image']
    log_print("got file")
    filename = img_file.filename
    ext = os.path.splitext(filename)[1]
    if (ext == ".jpg") or (ext == ".png") or (ext == ".bmp"):
        print("File accepted")
    else:
        resp = json.dumps({'message': 'Not accepted extension'})
        resp.status_code = 400
        return resp

    img_path = "/".join([target, filename])
    print("File saved to to:", img_path)
    img_file.save(img_path)
    print(11111)

    img = Image.open(img_path).resize((100, 100), Image.ANTIALIAS)
    X = torch.FloatTensor(
        np.array(img).reshape(-1, 100, 100)[:3]).unsqueeze_(0)
    modelResnet.eval()
    outputs = modelResnet(X)
    _, predicted = torch.max(outputs.data, 1)
    return json.dumps(dct[int(predicted[0])]), 200


if __name__ == '__main__':
    this_app.run(debug=True)

if __name__ != '__main__':
    gunicorn_logger = logging.getLogger('gunicorn.error')
    this_app.logger.handlers = gunicorn_logger.handlers
    this_app.logger.setLevel(gunicorn_logger.level)
