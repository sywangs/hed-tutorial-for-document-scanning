#!/usr/bin/python
# coding=utf8

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from edage_detect.DetectEdgeVersion6 import Main
from util import *
from hed_net import *

from tensorflow import flags

from flask import Flask
from flask import request
import numpy as np
import cv2
import json

flags.DEFINE_string('image', './test_image_del/test15.jpg',
                    'Image path to run hed, must be jpg image.')
flags.DEFINE_string('checkpoint_dir', './checkpoint',
                    'Checkpoint directory.')
flags.DEFINE_string('output_dir', './test_image_del',
                    'Output directory.')
FLAGS = flags.FLAGS




class DocScan(object):
    def __init__(self):
        batch_size = 1
        self.image_path_placeholder = tf.placeholder(tf.string)
        self.is_training_placeholder = tf.placeholder(tf.bool)

        image_tensor = tf.read_file(self.image_path_placeholder)
        image_tensor = tf.image.decode_jpeg(image_tensor, channels=3)
        image_tensor = tf.image.resize_images(image_tensor, [const.image_height, const.image_width])

        image_float = tf.to_float(image_tensor)

        if const.use_batch_norm == True:
            image_float = image_float / 255.0
        else:
            # for VGG style HED net
            image_float = mean_image_subtraction(image_float, [R_MEAN, G_MEAN, B_MEAN])
        image_float = tf.expand_dims(image_float, axis=0)

        self.dsn_fuse, self.dsn1, self.dsn2, self.dsn3, self.dsn4, self.dsn5 = mobilenet_v2_style_hed(image_float,\
                                                            batch_size,self.is_training_placeholder)

        global_init = tf.global_variables_initializer()

        # Saver
        hed_weights = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='hed')
        saver = tf.train.Saver(hed_weights)

        # with tf.Session() as sess:
        sess = tf.Session()
        sess.run(global_init)

        latest_ck_file = tf.train.latest_checkpoint(FLAGS.checkpoint_dir)
        if latest_ck_file:
            print('restore from latest checkpoint file : {}'.format(latest_ck_file))
            saver.restore(sess, latest_ck_file)
        else:
            print('no checkpoint file to restore, exit()')
            exit()
        self.sess = sess

    def getImage(self,image,outPutpath):
        feed_dict_to_use = {self.image_path_placeholder: image,
                            self.is_training_placeholder: False}

        _dsn_fuse, \
        _dsn1, \
        _dsn2, \
        _dsn3, \
        _dsn4, \
        _dsn5 = self.sess.run([self.dsn_fuse,
                          self.dsn1, self.dsn2,
                          self.dsn3, self.dsn4,
                          self.dsn5],
                         feed_dict=feed_dict_to_use)

        threshold = 0.0
        dsn_fuse_image = np.where(_dsn_fuse[0] > threshold, 255, 0)

        cv2.imwrite(outPutpath, dsn_fuse_image)

        pointsNeed = Main(outPutpath, '')

        return pointsNeed

    def __del__(self):
        self.sess.close()
        print("session closed!")

def save_result(imgpath,pointsNeed,imageDir,filePath):
    image = cv2.imread(imgpath)
    rows, cols, chans = image.shape

    pointsNeed = [
        [int(pointsNeed[0][0] * cols), int(pointsNeed[0][1] * rows)],
        [int(pointsNeed[1][0] * cols), int(pointsNeed[1][1] * rows)],
        [int(pointsNeed[2][0] * cols), int(pointsNeed[2][1] * rows)],
        [int(pointsNeed[3][0] * cols), int(pointsNeed[3][1] * rows)]
    ]

    pos1 = (pointsNeed[0][0], pointsNeed[0][1])
    pos2 = (pointsNeed[1][0], pointsNeed[1][1])
    pos3 = (pointsNeed[2][0], pointsNeed[2][1])
    pos4 = (pointsNeed[3][0], pointsNeed[3][1])
    cv2.putText(image, 'C1', pos1, cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255), 2)
    cv2.putText(image, 'C2', pos2, cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255), 2)
    cv2.putText(image, 'C3', pos3, cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255), 2)
    cv2.putText(image, 'C4', pos4, cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255), 2)
    cv2.putText(image, 'cen', (int(cols / 2), int(rows / 2)), cv2.FONT_HERSHEY_COMPLEX, 1,
                (0, 0, 255), 2)
    savePath = os.path.join(imageDir, '' + filePath)
    cv2.imwrite(savePath, image)


# app = Flask(__name__)
# docScan = DocScan()

# @app.route('/get_edage', methods=['GET', 'POST'])
# def getOCR():
#     argv = request.values
#     imgpath = argv.get('imgpath')
#     savePath = argv.get('savePath')
#     pointsNeed = docScan.getImage(imgpath, savePath)
#     return json.dumps(pointsNeed)
# 
# if __name__ == '__main__':
#     app.run()


if __name__ == "__main__":
    start = time.time()
    docScan = DocScan()
    print("init time is :",str(time.time() - start))

    imgType = ['jpeg','jpg','png']

    result_dir = "/Users/developer/Downloads/test_image/"
    dic = {}

    for root, dirs, files in os.walk("/Users/developer/Downloads/new/", topdown=False):
        for name in dirs:
            imageDir = os.path.join(root,name)
            for filePath in os.listdir(imageDir):
                if filePath.split('.')[-1] in imgType:
                    imgpath = os.path.join(imageDir,filePath)
                    savePath = os.path.join(imageDir,'trans' + filePath)
                    print('image handing:', imgpath)
                    start = time.time()

                    pointsNeed = docScan.getImage(imgpath,savePath)
                    print('pointsNeed', pointsNeed)

                    dic[filePath] = pointsNeed

                    # save_result(imgpath,pointsNeed,result_dir,filePath)

                    print('get image time is :', str(time.time() - start))

    fileObject = open('sampleList.json', 'w')
    dic = json.dumps(dic)
    fileObject.write(dic)
#
#     del docScan