# -*- coding: utf-8 -*-

#[prototxt][caffemodel][video_dir][img_save_path]


import os
import cv2
import sys
import numpy as np

caffe_root = '/home/dt/workspace/caffe-ssd'
sys.path.insert(0, caffe_root + '/python')
import caffe


def file_name(file_dir):
    file_list = []
    for root, dirs, files in os.walk(file_dir):
        for file in files:
            # print file
            if cmp(file.split('.')[-1], 'mp4') == 0:
                file_list.append(root + '/' + file)
            if cmp(file.split('.')[-1], 'avi') == 0:
                file_list.append(root + '/' + file)
    return file_list


def cut_img(rect, img):
    L = rect[0][0]
    R = rect[1][0]
    T = rect[0][1]
    B = rect[1][1]
    # print L, R, T, B
    return img[T:B, L:R]


def preprocess(src):
    img = cv2.resize(src, (300, 300))
    img = img - 127.5
    img = img / 127.5
    return img


def main():
    prototxt_path = sys.argv[1]
    model_path = sys.argv[2]
    # prototxt_path = '/home/dt/PycharmProjects/tensorflow/model/MobileNetSSD.prototxt'
    # model_path = '/home/dt/PycharmProjects/tensorflow/model/MobileNetSSD.caffemodel'
    caffe.set_mode_cpu()
    net = caffe.Net(prototxt_path, model_path, caffe.TEST)
    video_list = file_name(sys.argv[3])
    img_save_path = sys.argv[4]
    # video_list = '/home/dt/PycharmProjects/tensorflow/model/1'
    # video_list = file_name(video_list)
    # img_save_path = '/home/dt/PycharmProjects/tensorflow/model/3/'
    # print video_list

    for video in video_list:

        cap = cv2.VideoCapture(video)
        img_name = video.split('/')[-1]
        img_name = img_name[::-1]
        img_name = img_name.split('.', 1)[-1]
        img_name = img_name[::-1]
        p_id = 0
        while cap.isOpened():
            ret, image = cap.read()

            if not ret:
                break
            tmp = preprocess(image)
            img = tmp.astype(np.float32)
            img = img.transpose((2, 0, 1))
            net.blobs['data'].data[...] = img
            detections = net.forward()['detection_out']
            det_label = detections[0, 0, :, 1]
            det_conf = detections[0, 0, :, 2]
            det_xmin = detections[0, 0, :, 3] * image.shape[1]
            det_ymin = detections[0, 0, :, 4] * image.shape[0]
            det_xmax = detections[0, 0, :, 5] * image.shape[1]
            det_ymax = detections[0, 0, :, 6] * image.shape[0]
            # Get detections with confidence higher than 0.6.
            top_indices = [i for i, conf in enumerate(det_conf) if conf >= 0.5]
            top_conf = det_conf[top_indices]
            top_label_indices = det_label[top_indices].tolist()
            top_xmin = det_xmin[top_indices]
            top_ymin = det_ymin[top_indices]
            top_xmax = det_xmax[top_indices]
            top_ymax = det_ymax[top_indices]
            now_boxes = []
            for i in xrange(top_conf.shape[0]):
                p1 = (int(top_xmin[i]), int(top_ymin[i]))
                p2 = (int(top_xmax[i]), int(top_ymax[i]))
                # cv2.rectangle(image, p1, p2, (0, 255, 0), 4)
                now_boxes.append((p1, p2))
            for box in now_boxes:
                img_c = cut_img(box, image)
                p_id = p_id + 1
                cv2.imwrite(img_save_path + '/' + img_name + '_' + str(p_id) + '.jpg', img_c)


if __name__ == '__main__':
    main()
