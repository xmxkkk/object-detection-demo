import cv2
import numpy as np
import random
import matplotlib.pyplot as plt

import hashlib
import io
import logging
import os
import random
import re

import contextlib2
from lxml import etree
import numpy as np
import PIL.Image
import tensorflow as tf
import sys
import shutil

WORK_DIR = '/Users/xmx/anaconda/envs/py36/lib/python3.6/site-packages/tensorflow/models/research'

# sys.path.append(WORK_DIR)
# sys.path.append(WORK_DIR+'/slim')

from object_detection.dataset_tools import tf_record_creation_util
from object_detection.utils import dataset_util
from object_detection.utils import label_map_util


def getCircleShape(bg):
    width, height = bg.shape[0], bg.shape[1]
    while True:
        x = random.randint(0, width - 20) + 10
        y = random.randint(0, height - 20) + 10

        r = random.randint(0, 40) + 10

        if x + r < width - 10 and y + r < height - 10 and x - r > 10 and y - r > 10:
            break
    return x, y, r


def getRectShape(bg):
    width, height = bg.shape[0], bg.shape[1]
    while True:
        x = random.randint(0, width - 20) + 10
        y = random.randint(0, height - 20) + 10

        w = random.randint(0, 40) + 20 + x
        h = random.randint(0, 40) + 20 + y

        if w < width - 10 and h < height - 10:
            break
    return x, y, w, h


def getColor():
    return random.randint(20, 255), random.randint(20, 255), random.randint(20, 255)


def make_shape(max_num, width, height, bg_color=True, filename=None, label_map_dict=None):
    if bg_color:
        r_s = np.random.randint(20, 160)
        r = np.random.randint(r_s, r_s + random.randint(10, 90), width * height, np.uint8).reshape((width, height))
        g_s = np.random.randint(20, 160)
        g = np.random.randint(g_s, g_s + +random.randint(10, 90), width * height, np.uint8).reshape((width, height))
        b_s = np.random.randint(20, 160)
        b = np.random.randint(b_s, b_s + +random.randint(10, 90), width * height, np.uint8).reshape((width, height))
    else:
        r = np.zeros((width, height), np.uint8)
        g = np.zeros((width, height), np.uint8)
        b = np.zeros((width, height), np.uint8)

    bg = cv2.merge([r, g, b])

    xmins = []
    ymins = []
    xmaxs = []
    ymaxs = []
    classes = []
    classes_text = []
    truncated = []
    poses = []
    difficult_obj = []

    for i in range(max_num):
        label = random.randint(1, 2)
        x1, y1, x2, y2 = 0, 0, 0, 0
        if label == 1:
            shp = getCircleShape(bg)
            x1, y1, x2, y2 = shp[0] - shp[2], shp[1] - shp[2], shp[0] + shp[2], shp[1] + shp[2]
            cv2.circle(bg, (shp[0], shp[1]), shp[2], getColor(), -1)
            label_text = 'circle'
        elif label == 2:
            shp = getRectShape(bg)
            x1, y1, x2, y2 = shp[0], shp[1], shp[2], shp[3]
            cv2.rectangle(bg, (shp[0], shp[1]), (shp[2], shp[3]), getColor(), -1)
            label_text = 'rect'

        xmins.append(float(x1 / width))
        ymins.append(float(y1 / height))
        xmaxs.append(float(x2 / width))
        ymaxs.append(float(y2 / height))

        classes.append(label_map_dict[label_text])
        classes_text.append(label_text.encode('utf8'))
        truncated.append(0)
        poses.append("0".encode('utf8'))
        difficult_obj.append(0)

    filename=filename + '_temp.jpg'
    img_path = os.path.join('./data', filename + '_temp.jpg')

    cv2.imwrite(img_path, bg)

    with tf.gfile.GFile(img_path, 'rb') as fid:
        encoded_jpg = fid.read()
    encoded_jpg_io = io.BytesIO(encoded_jpg)
    image = PIL.Image.open(encoded_jpg_io)
    if image.format != 'JPEG':
        raise ValueError('Image format not JPEG')
    key = hashlib.sha256(encoded_jpg).hexdigest()

    feature_dict = {
        'image/height': dataset_util.int64_feature(height),
        'image/width': dataset_util.int64_feature(width),
        'image/filename': dataset_util.bytes_feature(filename.encode('utf8')),
        'image/source_id': dataset_util.bytes_feature(filename.encode('utf8')),
        'image/key/sha256': dataset_util.bytes_feature(key.encode('utf8')),
        'image/encoded': dataset_util.bytes_feature(encoded_jpg),
        'image/format': dataset_util.bytes_feature('jpeg'.encode('utf8')),
        'image/object/bbox/xmin': dataset_util.float_list_feature(xmins),
        'image/object/bbox/xmax': dataset_util.float_list_feature(xmaxs),
        'image/object/bbox/ymin': dataset_util.float_list_feature(ymins),
        'image/object/bbox/ymax': dataset_util.float_list_feature(ymaxs),
        'image/object/class/text': dataset_util.bytes_list_feature(classes_text),
        'image/object/class/label': dataset_util.int64_list_feature(classes),
        'image/object/difficult': dataset_util.int64_list_feature(difficult_obj),
        'image/object/truncated': dataset_util.int64_list_feature(truncated),
        'image/object/view': dataset_util.bytes_list_feature(poses)

    }
    example = tf.train.Example(features=tf.train.Features(feature=feature_dict))
    os.remove(img_path)
    return example


def save(size, max_num, width, height, bg_color=True,type='train'):
    output_dir = os.path.join('./data',type)
    shutil.rmtree(output_dir)
    os.makedirs(output_dir)

    label_map_path = './data/label/shape_label_map.pbtxt'
    num_shards = 10
    train_output_path = os.path.join(output_dir, 'shape_{}.record'.format(type))

    with contextlib2.ExitStack() as tf_record_close_stack:
        output_tfrecords = tf_record_creation_util.open_sharded_output_tfrecords(
            tf_record_close_stack, train_output_path, num_shards)
        label_map_dict = label_map_util.get_label_map_dict(label_map_path)

        for idx in range(size):
            tf_example = make_shape(max_num, width, height, bg_color, str(idx), label_map_dict)
            if tf_example:
                shard_idx = idx % num_shards
                output_tfrecords[shard_idx].write(tf_example.SerializeToString())



# save(10,1,64,64)
save(10000, 3, 128, 128, False,'train')
save(1000, 3, 128, 128, False,'test')

# save(1000,1,128,128,False)
# save(10000)
