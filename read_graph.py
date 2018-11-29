import tensorflow as tf
import cv2
import numpy as np
import os
import sys
import imutils
import xml.etree.ElementTree as ET

caffe_root = '/home/dt/workspace/caffe-ssd'
sys.path.insert(0, caffe_root + '/python')
import caffe


# net_file= 'det48.prototxt'
# caffe_model='det48.caffemodel'

# net = caffe.Net(net_file,caffe_model,caffe.TEST)



def graph_create(graphpath):
    with tf.gfile.FastGFile(graphpath, 'rb') as graphfile:
        graphdef = tf.GraphDef()
        graphdef.ParseFromString(graphfile.read())

        return tf.import_graph_def(graphdef, name='', return_elements=[
            'image_tensor:0', 'detection_boxes:0', 'detection_scores:0', 'detection_classes:0'])


def preprocess(src):
    img = cv2.resize(src, (300, 300))
    img = img - 127.5
    img = img / 127.5
    return img


def preprocess1(src):
    img = cv2.resize(src, (48, 48))
    img = img - 127.5
    img = img / 127.5
    return img


def rotate_bound(image, angle):
    # grab the dimensions of the image and then determine the
    # center
    (h, w) = image.shape[:2]
    (cX, cY) = (w // 2, h // 2)

    # grab the rotation matrix (applying the negative of the
    # angle to rotate clockwise), then grab the sine and cosine
    # (i.e., the rotation components of the matrix)
    M = cv2.getRotationMatrix2D((cX, cY), -angle, 1.0)
    cos = np.abs(M[0, 0])
    sin = np.abs(M[0, 1])

    # compute the new bounding dimensions of the image
    nW = int((h * sin) + (w * cos))
    nH = int((h * cos) + (w * sin))

    # adjust the rotation matrix to take into account translation
    M[0, 2] += (nW / 2) - cX
    M[1, 2] += (nH / 2) - cY

    # perform the actual rotation and return the image
    return cv2.warpAffine(image, M, (nW, nH))


def readxml(path):
    try:
        _file = open(path, 'r')
    except IOError:
        print 'no such file', path
        return []
    strxml = _file.read()
    root = ET.XML(strxml)
    boxlist = []
    for i in root.findall('object'):
        for j in i.iter('name'):
            object_name = j.text
        for j in i.iter('xmin'):
            object_xmin = int(j.text)
        for j in i.iter('ymin'):
            object_ymin = int(j.text)
        for j in i.iter('xmax'):
            object_xmax = int(j.text)
        for j in i.iter('ymax'):
            object_ymax = int(j.text)
        boxlist.append(((object_xmin, object_ymin), (object_xmax, object_ymax)))
    return boxlist


def file_name(file_dir):
    L = []
    for root, dirs, files in os.walk(file_dir):
        for file in files:
            if os.path.splitext(file)[1] == '.jpg':
                L.append(os.path.join(root, file))
    return L


def IOU(box1, box2):
    LT1, RB1 = box1
    LT2, RB2 = box2
    _L = max(LT1[0], LT2[0])
    _R = min(RB1[0], RB2[0])

    _T = max(LT1[1], LT2[1])
    _B = min(RB1[1], RB2[1])

    if _L > _R or _T > _B:
        return 0.
    S_jiao = (_R - _L) * (_B - _T)
    S_bing = (RB1[0] - LT1[0]) * (RB1[1] - LT1[1]) + (RB2[0] - LT2[0]) * (RB2[1] - LT2[1]) - S_jiao
    return 1.0 * S_jiao / S_bing


def getxmlpath(xml_root, image_path):
    xml_path = xml_root + '/' + image_path.split('/')[-1]
    xml_path = xml_path[::-1]
    xml_path = xml_path.split('.', 1)[-1]
    xml_path = xml_path[::-1]
    xml_path = xml_path + '.xml'
    return xml_path


p_id = 0
mode = -1
if len(sys.argv) == 5:
    mode = 2
elif len(sys.argv) == 4:
    mode = 1
else:
    mode = -1
TP, FN, FP, TN = 0, 0, 0, 0


def calnum(now_boxes, boxes_list):
    global TP, FN, FP
    for i in now_boxes:
        _flag = False
        for j in boxes_list:
            if IOU(i, j) > 0.5:
                TP = TP + 1
                _flag = True
                break
        if not _flag:
            FP = FP + 1
            FN_list.append(_i + '\n')
    for i in boxes_list:
        _flag = False
        for j in now_boxes:
            if IOU(i, j) > 0.5:
                _flag = True
                break
        if not _flag:
            FN = FN + 1


FN_list = []
if (mode == 1):
    # cap = cv2.VideoCapture("1.mp4")
    #image_tensor, box, score, cls = graph_create('/home/dt/PycharmProjects/tensorflow/frozen_inference_graph.pb')
    image_tensor, box, score, cls = graph_create(sys.argv[1])
    with tf.Session() as sess:
        #img_list = file_name('/home/dt/PycharmProjects/mypython/wc/img')
        img_list = file_name(sys.argv[2])
        #xml_root = '/home/dt/PycharmProjects/mypython/wc/xml'
        xml_root = sys.argv[3]
        total_image = len(img_list)
        for _i in img_list:
            image = cv2.imread(_i)
            # ret, image = cap.read()
            #image = image[180:1080, 455:1580]
            # image=image[0:720,0:1280]
            # image=imutils.rotate(image,90)
            image_data = np.expand_dims(image, axis=0).astype(np.uint8)
            p_id += 1
            # if p_id > 100:
            #     break
            b, s, c = sess.run([box, score, cls], {image_tensor: image_data})
            boxes = b[0]
            conf = s[0]
            clses = c[0]
            # writer = tf.summary.FileWriter('debug', sess.graph)
            now_boxes = []
            for i in range(8):
                bx = boxes[i]
                if conf[i] < 0.5:
                    continue
                print "OK"
                h = image.shape[0]
                w = image.shape[1]
                if bx[1] < 0 or bx[3] > 1 or bx[0] < 0 or bx[2] > 1 or (bx[2] - bx[0] < 0.01) or (bx[3] - bx[1] < 0.01):
                    continue
                p1 = (int(w * bx[1]), int(h * bx[0]))
                p2 = (int(w * bx[3]), int(h * bx[2]))
                # cv2.rectangle(image, p1, p2, (0, 255, 0))
                now_boxes.append((p1, p2))
            # cv2.imwrite(str(p_id)+".jpg",image)
            # cv2.imshow("mobilenet-ssd", image)
            # cv2.waitKey(0)
            xml_path = getxmlpath(xml_root, _i)
            boxes_list = readxml(xml_path)
            # for i in boxes_list:
            #     cv2.rectangle(image, i[0], i[1], (255, 0, 0), 4)
            calnum(now_boxes, boxes_list)
            # print TP, FP, FN
            print 'total:', total_image, 'now:', p_id
    log_file = open('bad_image.txt', 'w')
    for i in FN_list:
        log_file.writelines(i)
    log_file.close()
    print TP, FP, FN
elif (mode == 2):
    ssd_file = sys.argv[1]
    ssd_model = sys.argv[2]
    caffe.set_mode_cpu()
    ssdnet = caffe.Net(ssd_file, ssd_model, caffe.TEST)
    # cap = cv2.VideoCapture("2.avi")
    img_list = file_name(sys.argv[3])
    xml_root = sys.argv[4]
    total_image = len(img_list)
    for _i in img_list:
        p_id = p_id + 1
        # if p_id > 100:
        #     break
        image = cv2.imread(_i)
        # image = image[180:1080, 455:1580]
        # image=rotate_bound(image,-90)
        tmp = preprocess(image)
        img = tmp.astype(np.float32)
        img = img.transpose((2, 0, 1))
        ssdnet.blobs['data'].data[...] = img
        detections = ssdnet.forward()['detection_out']
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
            cv2.rectangle(image, p1, p2, (0, 255, 0), 4)
            now_boxes.append((p1, p2))

        xml_path = getxmlpath(xml_root, _i)
        boxes_list = readxml(xml_path)
        # for i in boxes_list:
        #     cv2.rectangle(image, i[0], i[1], (255, 0, 0), 4)
        calnum(now_boxes, boxes_list)
        # print TP, FP, FN
        print 'total:', total_image, 'now:', p_id
        # cv2.imshow("mobilenet-ssd", image)
        # cv2.waitKey(0)
    log_file = open('bad_image.txt', 'w')
    for i in FN_list:
        log_file.writelines(i)
    log_file.close()
    print TP, FP, FN

# if (mode == 3):
#     cap = cv2.VideoCapture("1.mp4")
#     with tf.Session() as sess:
#         while (cap.isOpened()):
#             ret, image = cap.read()
#             image = image[180:1080, 455:1580]
#             image_data = np.expand_dims(image, axis=0).astype(np.uint8)
#             p_id += 1
#             b, s, c = sess.run([box, score, cls], {image_tensor: image_data})
#             boxes = b[0]
#             conf = s[0]
#             clses = c[0]
#             # writer = tf.summary.FileWriter('debug', sess.graph)
#             print "PK"
#             for i in range(8):
#                 bx = boxes[i]
#                 if conf[i] < 0.5:
#                     continue
#                 print "OK"
#                 h = image.shape[0]
#                 w = image.shape[1]
#                 if bx[1] < 0 or bx[3] > 1 or bx[0] < 0 or bx[2] > 1 or (bx[2] - bx[0] < 0.01) or (bx[3] - bx[1] < 0.01):
#                     continue
#                 p1 = (int(w * bx[1]), int(h * bx[0]))
#                 p2 = (int(w * bx[3]), int(h * bx[2]))
#                 cv2.rectangle(image, p1, p2, (0, 255, 0))
#                 tmp = image[int(h * bx[0]):int(h * bx[2]), int(w * bx[1]):int(w * bx[3])]
#                 cv2.imshow("R", tmp)
#                 tmp = preprocess(tmp)
#                 img = tmp.astype(np.float32)
#                 img = img.transpose((2, 0, 1))
#                 net.blobs['data'].data[...] = img
#                 out = net.forward()
#                 scores = out['prob1'][0][1]
#                 print scores
#                 result_text = "SSD: " + str(round(conf[i], 2)) + " R: " + str(round(scores, 2))
#                 cv2.putText(image, result_text, (int(w * bx[1]), int(h * bx[2])), cv2.FONT_HERSHEY_COMPLEX, 0.5,
#                             (0, 0, 255), 2)
#             # cv2.imwrite(str(p_id)+".jpg",image)
#             cv2.imshow("mobilenet-ssd", image)
#             cv2.waitKey(0)
# if (mode == 4):
#     cap = cv2.VideoCapture("1.mp4")
#     with tf.Session() as sess:
#         while (cap.isOpened()):
#             ret, image = cap.read()
#             image = image[180:1080, 455:1580]
#             image_data = np.expand_dims(image, axis=0).astype(np.uint8)
#             p_id += 1
#             b, s, c = sess.run([box, score, cls], {image_tensor: image_data})
#             boxes = b[0]
#             conf = s[0]
#             clses = c[0]
#             # writer = tf.summary.FileWriter('debug', sess.graph)
#             print "PK"
#             for i in range(8):
#                 bx = boxes[i]
#                 if conf[i] < 0.5:
#                     continue
#                 print "OK"
#                 h = image.shape[0]
#                 w = image.shape[1]
#                 if bx[1] < 0 or bx[3] > 1 or bx[0] < 0 or bx[2] > 1 or (bx[2] - bx[0] < 0.01) or (bx[3] - bx[1] < 0.01):
#                     continue
#                 p1 = (int(w * bx[1]), int(h * bx[0]))
#                 p2 = (int(w * bx[3]), int(h * bx[2]))
#                 cv2.rectangle(image, p1, p2, (0, 255, 0))
#                 tmp = image[int(h * bx[0]):int(h * bx[2]), int(w * bx[1]):int(w * bx[3])]
#                 cv2.imshow("R", tmp)
#                 tmp = preprocess(tmp)
#                 img = tmp.astype(np.float32)
#                 img = img.transpose((2, 0, 1))
#                 net.blobs['data'].data[...] = img
#                 out = net.forward()
#                 scores = out['prob1'][0][1]
#                 print scores
#                 result_text = "SSD: " + str(round(conf[i], 2)) + " R: " + str(round(scores, 2))
#                 cv2.putText(image, result_text, (int(w * bx[1]), int(h * bx[2])), cv2.FONT_HERSHEY_COMPLEX, 0.5,
#                             (0, 0, 255), 2)
#             # cv2.imwrite(str(p_id)+".jpg",image)
#             cv2.imshow("mobilenet-ssd", image)
#             cv2.waitKey(0)
''' 
with tf.Session() as sess:   
    for f in os.listdir(im_dir):
        image_file=im_dir+f   
        image = cv2.imread(image_file)
        print image_file
        image_data = np.expand_dims(image, axis=0).astype(np.uint8)
        p_id+=1
        b, s, c = sess.run([box, score, cls], {image_tensor: image_data})
        boxes = b[0]
        conf = s[0]
        clses = c[0]
        #writer = tf.summary.FileWriter('debug', sess.graph)
        print "PK"
        for i in range(8):
            bx = boxes[i]
            if conf[i] < 0.5:
              continue
            print "OK"
            h = image.shape[0]
            w = image.shape[1]
            if bx[1]<0 or bx[3]>1 or bx[0]<0 or bx[2]>1 or (bx[2]-bx[0]<0.01) or (bx[3]-bx[1]<0.01):
              continue
            p1 = (int(w * bx[1]), int(h * bx[0]))
            p2 = (int(w * bx[3]) ,int(h * bx[2]))
            cv2.rectangle(image, p1, p2, (0,255,0))
            tmp=image[int(h*bx[0]):int(h*bx[2]),int(w*bx[1]):int(w*bx[3])]
	    cv2.imshow("R",tmp)
            tmp=preprocess(tmp)
            img = tmp.astype(np.float32)
            img = img.transpose((2, 0, 1))
            net.blobs['data'].data[...] = img
            out = net.forward()
            scores=out['prob1'][0][1]
            print scores
            result_text="SSD: "+str(round(conf[i],2))+" R: "+str(round(scores,2))
            cv2.putText(image,result_text,(int(w*bx[1]),int(h * bx[2])),cv2.FONT_HERSHEY_COMPLEX,0.5,(0,0,255),2)
        #cv2.imwrite(str(p_id)+".jpg",image)
        cv2.imshow("mobilenet-ssd", image)
        cv2.waitKey(0) 
'''
'''
python read_graph.py /home/dt/PycharmProjects/tensorflow/frozen_inference_graph.pb /home/dt/PycharmProjects/mypython/VOC2007_new_rule/JPEGImages /home/dt/PycharmProjects/mypython/VOC2007_new_rule/Annotations

'''