import cv2
img = cv2.imread('./human_body_00024.jpg')
g_x = 0
g_y = 0

def on_EVENT_LBUTTONDOWN(event, x, y, flags, param):
    global g_x
    global g_y
    if event == cv2.EVENT_LBUTTONDOWN:
        xy = '%d,%d' % (x, y)
        print("x, y = {}, {}".format(x, y))
        g_x, g_y = x, y
        cv2.circle(img, (x, y), 1, (255, 0, 0), thickness=-1)
        cv2.putText(img, xy, (x, y), cv2.FONT_HERSHEY_PLAIN,1.0, (0, 255, 0), thickness=2)
        cv2.imshow("image", img)
    if event == cv2.EVENT_RBUTTONDOWN:
        cv2.rectangle(img, (g_x, g_y), (x, y), (0, 255, 0), 2)
        cv2.imshow("image", img)
def on_EVENT_RBUTTONDOWN(event, x, y, flags, param):
    if event == cv2.EVENT_RBUTTONDOWN:
        cv2.rectangle(img, (g_x, g_y), (x, y), (0, 255, 0), 2)
        cv2.imshow("image", img)

cv2.namedWindow("image")
loc = cv2.setMouseCallback("image", on_EVENT_LBUTTONDOWN)
#loc = cv2.setMouseCallback("image", on_EVENT_RBUTTONDOWN)
cv2.imshow("image", img)
cv2.waitKey(0)
