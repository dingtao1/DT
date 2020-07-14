import os
import matplotlib.pyplot as plt
import cv2
list_x1 = []
list_y1 = []
list_x2 = []
list_y2 = []

def cmp(x):
    return x[0]
def s_box(box):
    return (box[2]-box[0])*(box[3]-box[1])
def get_IOU(box1, box2):
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    dx = x2 - x1
    if dx < 0:
        dx = 0
    dy = y2 - y1
    if dy < 0:
        dy = 0
    s1 = dx * dy
    s2 = s_box(box1) + s_box(box2) - s1
    return s1 / s2

def get_box(gt_path, pre_path):
    gt_lines = open(gt_path, "r").readlines()
    pre_lines = open(pre_path, "r").readlines()
    gt_box = []
    for line in gt_lines:
        temp = line.split(' ')
        gt_box.append((int(temp[1]), int(temp[2]), int(temp[3]), int(temp[4])))
    pre_box = []
    for line in pre_lines:
        temp = line.split(' ')
        pre_box.append((int(temp[2]), int(temp[3]), int(temp[4]), int(temp[5])))
    return gt_box, pre_box

def work(gt_path, pre_path):
    gt_lines = open(gt_path, "r").readlines()
    pre_lines = open(pre_path, "r").readlines()
    gt_box = []
    for line in gt_lines:
        temp = line.split(' ')
        gt_box.append((int(temp[1]), int(temp[2]), int(temp[3]), int(temp[4])))
    pre_box = []
    for line in pre_lines:
        temp = line.split(' ')
        pre_box.append((float(temp[1]), (int(temp[2]), int(temp[3]), int(temp[4]), int(temp[5]))))
    pre_box = sorted(pre_box, key=cmp)
    pre_box.reverse()

    for conf, box in pre_box:
        maxx = 0
        choose_gt = -1
        for gt_b in gt_box:
            if get_IOU(box, gt_b) > maxx:
                maxx = get_IOU(box, gt_b)
                choose_gt = gt_b
        if maxx > 0.5:
            list_x1.append((box[0] - choose_gt[0], gt_path.split('/')[-1]))
            list_y1.append((box[1] - choose_gt[1], gt_path.split('/')[-1]))
            list_x2.append((box[2] - choose_gt[2], gt_path.split('/')[-1]))
            list_y2.append((box[3] - choose_gt[3], gt_path.split('/')[-1]))
            gt_box.remove(choose_gt)
def autolabel(rects):
    for rect in rects:
        height = rect.get_height()
        plt.text(rect.get_x()+rect.get_width()/2.- 0.2, 1.03*height, '%s' % int(height))

def fun(list):
    _map = {}
    for x, y in list:
        x = x//10
        x = x * 10
        if _map.get(x, -1) == -1:
            _map[x] = 1
        else:
            _map[x] += 1
    xx = []
    yy = []
    for key in _map:
        xx.append(key)
        yy.append(_map[key])
    autolabel(plt.bar(range(len(yy)), yy, color='rgb', tick_label=xx))
    plt.show()
def fun_2(str, list):
    num = 0
    maxx = -1000
    minn = 1000
    for x, y in list:
        if x >= -10 and x < 10:
            num += 1
        maxx = max(maxx, x)
        minn = min(minn, x)
    print(str + "#:{0}, min:{1}, max{2}".format(num/len(list), minn, maxx))
if __name__ == '__main__':
    #A = "/home/user1/workspace/mAP/input/ground-truth/human_body_03474.txt"
    #B = "/home/user1/workspace/mAP/input/detection-results/human_body_03474.txt"
    #work(A, B)
    A_dir = "/home/user1/workspace/mAP/input/ground-truth/"
    B_dir = "/home/user1/workspace/mAP/input/detection-results/"
    C_dir = "/home/user1/workspace/mAP/input/images-optional/"
    A_files = os.listdir(A_dir)
    for A in A_files:
        work(A_dir+A, B_dir+A)
    list_x1 = sorted(list_x1, key=cmp)
    list_y1 = sorted(list_y1, key=cmp)
    list_x2 = sorted(list_x2, key=cmp)
    list_y2 = sorted(list_y2, key=cmp)

    for i, j in enumerate(list_y2):
        if i > 10:
            break
        print(j)
        j = j[1]
        j = j.split('.')[0]

        gt_box, pre_box = get_box(A_dir + j + '.txt', B_dir + j + '.txt')
        img = cv2.imread(C_dir + j + '.jpg')
        for box in gt_box:
            img = cv2.rectangle(img, (box[0], box[1]), (box[2], box[3]), (0, 255, 100), 2)#绿色
        for box in pre_box:
            img = cv2.rectangle(img, (box[0], box[1]), (box[2], box[3]), (0, 0, 255), 2)#红色
        cv2.imshow("image",img)
        cv2.waitKey(0)
    #fun_2("x1", list_x1)
    #fun_2("y1", list_y1)
    #fun_2("x2", list_x2)
    #fun_2("y2", list_y2)