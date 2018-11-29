#include<opencv2/core/core.hpp>
#include<opencv2/highgui/highgui.hpp>
#include "opencv2/imgproc/imgproc.hpp"
using	namespace cv;
/*
 * img 输入图像,x,y,w,h矩形的左上角位置和宽长,Scalar_Bugle角框和虚线的颜色,Scalar_rect是内部填充的颜色.
 *
 */
void paintrect(cv::Mat &img, int x, int y, int w, int h, Scalar Scalar_Bugle = Scalar(0,255,255),
        Scalar Scalar_rect = Scalar(255, 255, 255))
{
    int delta = 0.1 * max(w, h), space = 0.05 * max(w, h);
    line(img, Point(x, y), Point(x, y + delta), Scalar_Bugle, 6);
    line(img, Point(x, y), Point(x + delta, y), Scalar_Bugle, 6);

    line(img, Point(x, y + h), Point(x, y + h - delta), Scalar_Bugle, 6);
    line(img, Point(x, y + h), Point(x + delta, y + h), Scalar_Bugle, 6);

    line(img, Point(x + w, y), Point(x + w - delta, y), Scalar_Bugle, 6);
    line(img, Point(x + w, y), Point(x + w, y + delta), Scalar_Bugle, 6);

    line(img, Point(x + w, y + h), Point(x + w, y + h - delta), Scalar_Bugle, 6);
    line(img, Point(x + w, y + h), Point(x + w - delta, y + h), Scalar_Bugle, 6);

    for(int i = x + 2*delta; i < x + w - delta; i += space){
        line(img, Point(i - space / 2, y), Point(i, y), Scalar_Bugle, 2);
    }

    for(int i = y + 2*delta; i < y + h - delta; i += space){
        line(img, Point(x , i - space / 2), Point(x, i), Scalar_Bugle, 2);
    }

    for(int i = y + 2*delta; i < y + h - delta; i += space){
        line(img, Point(x + w , i - space / 2), Point(x + w, i), Scalar_Bugle, 2);
    }

    for(int i = x + 2*delta; i < x + w - delta; i += space){
        line(img, Point(i - space / 2, y + h), Point(i, y + h), Scalar_Bugle, 2);
    }
    Mat rec = img.clone();
    rectangle(rec, Point(x, y), Point(x + w, y + h), Scalar_rect, -1);
    addWeighted(img, 0.75, rec, 0.25, 0.0, img);
}

int main()
{
    Mat img = imread("../test.jpg");
    paintrect(img, 100, 100, 300, 300);
    imshow("123",  img);
    waitKey();
    return 0;
}