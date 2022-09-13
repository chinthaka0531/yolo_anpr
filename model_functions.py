import cv2
import numpy as np
import easyocr
import torch
import re

reader = easyocr.Reader(['en'])
model = torch.hub.load('ultralytics/yolov5', 'custom', 'best.pt')


def ocr(plate):
    cv2.imwrite('plate.jpg', plate)
    bounds = reader.readtext('plate.jpg')
    txt = bounds[0][1].upper()
    txt = re.sub(r'[^\w]', '', txt)

    return txt


def plot_boxes(img, boxes_df):
    bgr_img = img.copy()
    number = ""
    for box in boxes_df.to_numpy():
        xmin, ymin, xmax, ymax, confidence, _, cls_name = box
        xmin, ymin, xmax, ymax, confidence = int(xmin), int(ymin), int(xmax), int(ymax), np.round(confidence, 2)
        txt = 'License Plate' + ' ' + str(confidence)
        labelSize = cv2.getTextSize(txt, cv2.FONT_HERSHEY_SIMPLEX, 0.65, 2)
        ((lw, lh), _) = labelSize
        lh = lh + 10
        bgr_img = cv2.rectangle(bgr_img, (xmin, ymin - lh), (xmin + lw, ymin), [255, 0, 0], -1)
        bgr_img = cv2.rectangle(bgr_img, (xmin, ymin), (xmax, ymax), [255, 0, 0], 2)
        bgr_img = cv2.putText(bgr_img, txt, (xmin, ymin - 9), cv2.FONT_HERSHEY_SIMPLEX, 0.6, [255, 255, 255], 2,
                              cv2.LINE_AA)
        cropped_plate = bgr_img[ymin:ymax, xmin:xmax]
        # Number plate
        number = ocr(cropped_plate)
        labelSize = cv2.getTextSize(number, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
        ((lw, lh), _) = labelSize
        bgr_img = cv2.rectangle(bgr_img, (xmin, ymax), (xmin + lw, ymax + lh + 10), [255, 0, 0], -1)
        bgr_img = cv2.putText(bgr_img, number, (xmin, ymax + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.65, [255, 255, 255], 2,
                              cv2.LINE_AA)
    rgb_img = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2RGB)

    return bgr_img, rgb_img, number


def predict(img_path):
    im = cv2.imread(img_path)
    results = model(im)
    df = results.pandas().xyxy[0]
    bgr, rgb, text = plot_boxes(im, df)
    return bgr, rgb, text
