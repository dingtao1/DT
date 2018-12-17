# coding=utf-8
'''
梯度下降法求解最小二乘
'''
x = [1, 2, 3, 4]
y = [1, 2, 3, 4]

w, b = 10221, -12122


def fun(_x):
    global w, b
    return w * _x + b


if __name__ == '__main__':
    for _i in range(100000):
        dw, db = 0, 0
        for i in range(len(x)):
            dw = dw + (fun(x[i]) - y[i]) * x[i]
            db = db + (fun(x[i]) - y[i])
        w = w - 0.001 * dw / len(x)
        b = b - 0.001 * db / len(x)
    for i in x:
        print fun(i)
