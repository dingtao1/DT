#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <opencv2/opencv.hpp>
#include <opencv2/highgui.hpp>

void avi2WriteYuv(const char *in, const char *out)
{
    cv::VideoCapture vc;
    bool flag = vc.open(in);
    if (!flag) {
        printf("avi file open error \n");
        system("pause");
        exit(-1);
    }

    int frmCount = vc.get(CV_CAP_PROP_FRAME_COUNT);
    //frmCount -= 5;
    printf("frmCount: %d \n", frmCount);

    int w = vc.get(CV_CAP_PROP_FRAME_WIDTH);
    int h = vc.get(CV_CAP_PROP_FRAME_HEIGHT);
    //printf("wwwwww%d\n", w);
    //printf("hhhhhh%d\n", h);
    int bufLen = w * h * 3 / 2;
    unsigned char *pYuvBuf = new unsigned char[bufLen];
    FILE *pFileOut = fopen(out, "w+");
    if (!pFileOut) {
        printf("pFileOut open error \n");
        system("pause");
        exit(-1);
    }
    //printf("pFileOut open ok \n");

    for (int i = 0; i < frmCount; i++) {
        printf("%d/%d \n", i + 1, frmCount);

        cv::Mat srcImg;
        vc >> srcImg;

        //cv::imshow("img", srcImg);
        //cv::waitKey(1);

        cv::Mat yuvImg;
        cv::cvtColor(srcImg, yuvImg, CV_BGR2YUV_I420);
        memcpy(pYuvBuf, yuvImg.data, bufLen * sizeof(unsigned char));

        fwrite(pYuvBuf, bufLen * sizeof(unsigned char), 1, pFileOut);
    }

    fclose(pFileOut);
    delete[] pYuvBuf;
}

void DisplayYUV(const char *in, int _w, int _h)
{
    int w = _w;
    int h = _h;
    printf("yuv file w: %d, h: %d \n", w, h);

    FILE *pFileIn = fopen(in, "rb+");
    int bufLen = w * h * 3 / 2;
    unsigned char *pYuvBuf = new unsigned char[bufLen];
    int iCount = 0;
    while(1){
        int ret = fread(pYuvBuf, bufLen * sizeof(unsigned char), 1, pFileIn);
        if(ret == 0){
            break;
        }
        cv::Mat yuvImg;
        yuvImg.create(h * 3 / 2, w, CV_8UC1);
        memcpy(yuvImg.data, pYuvBuf, bufLen*sizeof(unsigned char));
        cv::Mat rgbImg;
        cv::cvtColor(yuvImg, rgbImg, CV_YUV2BGR_I420);

        cv::imshow("img", rgbImg);
        cv::waitKey(40);

        printf("%d %d\n", iCount++, ret);
    }

    delete[] pYuvBuf;

    fclose(pFileIn);
}
void YUV2avi(const char *in, const char *out, int _w, int _h)
{
    int w = _w;
    int h = _h;
    printf("yuv file w: %d, h: %d \n", w, h);

    FILE *pFileIn = fopen(in, "rb+");
    int bufLen = w * h * 3 / 2;
    unsigned char *pYuvBuf = new unsigned char[bufLen];
    int iCount = 0;
    CvVideoWriter *writer = cvCreateVideoWriter(
            out, CV_FOURCC('X','2','6','4'),25,cvSize(w,h),1
    );

    while(1){
        int ret = fread(pYuvBuf, bufLen * sizeof(unsigned char), 1, pFileIn);
        if(ret == 0){
            break;
        }
        cv::Mat yuvImg;
        yuvImg.create(h * 3 / 2, w, CV_8UC1);
        memcpy(yuvImg.data, pYuvBuf, bufLen * sizeof(unsigned char));
        cv::Mat rgbImg;
        cv::cvtColor(yuvImg, rgbImg, CV_YUV2BGR_I420);
        IplImage _img = rgbImg;
        cvWriteFrame(writer, &_img);
        char s[100];
        //sprintf(s, "%spic%d%s", dir, i, ".jpg");
        //cv::imwrite(s, rgbImg);
        //cv::waitKey(40);
        printf("%d\n", iCount++);
    }

    delete[] pYuvBuf;

    cvReleaseVideoWriter(&writer);
    fclose(pFileIn);
}
int getfilelen(const char *path)
{
    FILE *stream = fopen(path, "rb++");
    if (stream == NULL) {
        printf("error\n");
    }

    fseek(stream, 0L, SEEK_END);
    int length = ftell(stream);
    fseek(stream, 0, SEEK_SET);
    fclose(stream);
    return length;
}

int main(int argc, char **argv)
{
    const char *in = argv[1];
    const char *out = argv[2];
    avi2WriteYuv(in, out);
    DisplayYUV(out, 1920, 1080);
    return 0;
}