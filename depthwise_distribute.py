import os
import matplotlib.pyplot as plt


def draw(lines, n):
    now = 0
    m = len(lines) // n
    for i in range(n):
        _lines = lines[now:now+m]

        now = now + m
        x = []
        y = []
        for line in _lines:
            x.append(i)
            y.append(line)
        print("#######:" + str(i))
        #for (i, j) in zip(x, y):
           # print((i, j))
        plt.scatter(x, y, marker='.', color='red', s = 10)

def work(path):
    lines = open(path, "r").readlines()
    lines = [float(x) for x in lines]
    draw(lines, 1024)
    plt.legend(loc='best')
    plt.show()

if __name__ == '__main__':
    path = './mapper_param/conv6_dw_blob0_weight.float'
    work(path)