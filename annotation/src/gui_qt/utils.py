#!usr/bin/env python
# -*- coding: utf-8 -*-

import cv2
import numpy as np
from PyQt5.QtGui import QImage
from PyQt5.QtGui import QPixmap

def cv_img_resize(cv_img, width, height):
    img_resized = cv2.resize(cv_img, dsize=(width, height), interpolation=cv2.INTER_LINEAR)
    return img_resized

def cv_img_to_q_image(cv_img):
    height, width, channel = cv_img.shape
    bytes_per_line = 3 * width
    q_img = QImage(cv_img.data, width, height, bytes_per_line, QImage.Format_RGB888)
    q_img = q_img.rgbSwapped()

    return q_img

def q_image_to_q_pixmap(q_img):
    return QPixmap.fromImage(q_img)
