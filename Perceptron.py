# coding=utf-8
'''
感知机输入（x_i, y_i）,x_i in R^n,y_i = {+1,-1}
1.任意选取w,b
2.选取(x,y)
3.如果y_i(wx_i+b)<=0
    w = w + \eta*y_i*x_i
    b = b + \eta*y_i
    转到步骤(2)
否则结束

可以证明如果数据可分，则感知机一定收敛，否则不收敛。
'''

dataset = [(0, 0), (0, 1), (1, 0), (1, 1)]
label = [1, 1, -1, -1]

W = [100, 1111]
B = -992


def fun(x, y):
    global B
    return y * (x[0] * W[0] + x[1] * W[1] + B)


def fun2(x):
    global B
    return x[0] * W[0] + x[1] * W[1] + B


def update(x, y, eta):
    global B
    W[0] = W[0] + eta * y * x[0]
    W[1] = W[1] + eta * y * x[1]
    B = B + eta * y


if __name__ == '__main__':
    while True:
        flag = True
        for i in range(len(dataset)):
            if fun(dataset[i], label[i]) <= 0:
                update(dataset[i], label[i], 0.01)
                flag = False
        if flag:
            break
    print W, B
    for i in dataset:
        if fun2(i) >= 0:
            print '1'
        else:
            print '-1'
