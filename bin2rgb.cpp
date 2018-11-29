#include <opencv2/opencv.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/core.hpp>
#include <iostream>

using namespace std;

using namespace cv;


int getfilelen()
{
    FILE *stream;
    stream = fopen("../dog_bike_car_300x300.bgr", "rb");
    if (stream == NULL) {
        printf("error\n");
    }

    fseek(stream, 0L, SEEK_END);
    int length = ftell(stream);
    fseek(stream, 0, SEEK_SET);
    fclose(stream);
    return length;
}


int main()
{
    printf("%d\n",getfilelen());
    Mat img = imread("../dog_bike_car.jpg");
    //imshow("1", img);
    Mat dstimg;
    resize(img, dstimg, Size(300, 300), (0, 0), (0, 0), INTER_LINEAR);
    //imshow("2", dstimg);
    FILE *fp = fopen("../dog_bike_car_300x300.bgr", "rb");

    Mat imgb;
    imgb.create(Size(300, 300), CV_8UC3);
    printf("%d %d %d\n", imgb.rows, imgb.cols, imgb.channels());
    for (int k = 0; k < 3; k++) {
        for (int i = 0; i < 300; i++) {
            for (int j = 0; j < 300; j++) {
                uchar *ret = imgb.data + 3 * (i * 300 + j) + k;
                fread(ret, sizeof(uchar), 1, fp);
                //printf("%d\n",(int)*ret);
            }
        }
    }

    imshow("3", imgb);


    for (int i = 0; i < 300 * 300 * 3; i++) {
        if ((*(dstimg.data + i)) != (*(imgb.data + i))) {
            printf("no\n");
        }
        //printf("%d %d\n", *(dstimg.data + i), *(imgb.data + i));
    }

    waitKey(0);
    return 0;
}