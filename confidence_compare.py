import os
import matplotlib.pyplot as plt
def autolabel(rects):
    for rect in rects:
        height = rect.get_height()
        #plt.text(rect.get_x()+rect.get_width()/2.- 0.2, 1.03*height, '%s' % int(height))

def fun(list):
    _map = {}
    list = [x*100 for x in list]
    for x in list:
        x = x//1
        #x = x * 10
        if _map.get(x, -1) == -1:
            _map[x] = 1
        else:
            _map[x] += 1
    xx = []
    yy = []
    for key in _map:
        xx.append(key)
        yy.append(_map[key])
    #xx = [x / 100 for x in xx]
    _xx = []
    _yy = []

    for (x, y) in zip(xx, yy):
        if x == 0:
            _xx.append(x)
            _yy.append(y)
        else:
            _xx.append(None)
            _yy.append(0)
    xx = _xx
    autolabel(plt.bar(range(len(yy)), yy, color='rgb', tick_label=xx))
    plt.show()

def cmp(x):
    return x[0]
def cmp1(x):
    return x[1]
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

def work(gt_path, pre_path):
    conf_list = []
    gt_lines = open(gt_path, "r").readlines()
    pre_lines = open(pre_path, "r").readlines()
    gt_box = []
    for line in gt_lines:
        temp = line.split(' ')
        gt_box.append((float(temp[1]), (int(temp[2]), int(temp[3]), int(temp[4]), int(temp[5]))))
    pre_box = []
    for line in pre_lines:
        temp = line.split(' ')
        pre_box.append((float(temp[1]), (int(temp[2]), int(temp[3]), int(temp[4]), int(temp[5]))))
    pre_box = sorted(pre_box, key=cmp)
    pre_box.reverse()
    for conf, box in pre_box:
        maxx = 0
        choose_gt = -1
        for gt_conf, gt_b in gt_box:
            if get_IOU(box, gt_b) > maxx:
                maxx = get_IOU(box, gt_b)
                choose_gt = (gt_conf, gt_b)
        if maxx > 0.5:
            conf_list.append((choose_gt[0], conf))
            gt_box.remove(choose_gt)
    return conf_list
if __name__ == '__main__':
    #A = "/home/user1/workspace/mAP/input/object_0.3/caffe/human_body_00024.txt"
    #B = "/home/user1/workspace/mAP/input/object_0.3/object_raw/human_body_00024.txt"
    Gt_Dir = "/home/user1/workspace/mAP/input/object_0.3/caffe/"
    Pre_Dir = "/home/user1/workspace/mAP/input/object_0.3/object/"
    files = os.listdir(Gt_Dir)
    conf_lists = []
    for file in files:
        temp = work(Gt_Dir + file, Pre_Dir + file)
        for it in temp:
            conf_lists.append((file, it))
    maxx = 0
    lists = []
    for file, it in conf_lists:
        lists.append((file,it[1] - it[0]))

    lists = sorted(lists, key=cmp1)
    #for it in lists:
        #print(it)
    fun([x[1] for x in lists])
    num = 0
    for x in lists:
        if abs(x[1]) < 0.05:
            num += 1
    print("acc: " + str(num / len(lists)))