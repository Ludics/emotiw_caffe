#include "opencv2/objdetect.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"
#include <cstring>
#include <stdio.h>
#include <fstream>
#include <iostream>

using namespace std;
using namespace cv;

static void help()
{

}

Mat getwarpAffineImg(Mat &src, vector<Point2f> &landmarks);

int main( int argc, const char** argv )
{
    VideoCapture capture;
    Mat frame, image;
    string inputName_img,inputName_txt;
    Mat img;
    cv::CommandLineParser parser(argc, argv,
        "{help h||}"
        "{@filename||}"
    );
    inputName_img= parser.get<string>("@filename");
	inputName_txt= inputName_img.substr(0,inputName_img.length()-4)+".txt";
	//cout<<inputName_img<<' '<<inputName_txt<<endl;
	image = imread( inputName_img , 0 );
	vector<Point2f> landmarks;
	//imshow("img",image);
	//ifstream fin("testdata_xf_3/CuiZe.txt");
	char filename[100];
	int i;
	for(i=0;i<=inputName_txt.length();i++)
		filename[i]=inputName_txt[i];
	filename[i]='\0';
	ifstream fin(filename);
	//char p[]="testdata_xf_3/CuiZe.txt";
	//fin=fopen(&p,'r');
	for(i=0;i<67;i++)
	{
		//int p_x,p_y;
		Point2f p;
		fin>>p.x>>p.y;
		landmarks.push_back(p);
	}
	img=getwarpAffineImg(image,landmarks);
	//imshow("img",img);
	resize(img,img,Size(128,128),0,0,CV_INTER_LINEAR);
	imwrite(inputName_img,img);
	waitKey(0);
	return 0;
}

    //根据眼睛坐标对图像进行仿射变换
    //src - 原图像
    //landmarks - 原图像中68个关键点
    Mat getwarpAffineImg(Mat &src, vector<Point2f> &landmarks)
    {
        Mat oral;
		src.copyTo(oral);
        for (int j = 0; j < landmarks.size(); j++)
        {
            circle(oral, landmarks[j], 2, Scalar(255, 0, 0));
        }
		//imshow("img_origin",oral);
	//
   Point2f srcTri[3];
   Point2f dstTri[3];

   Mat rot_mat( 2, 3, CV_32FC1 );
   Mat warp_mat( 2, 3, CV_32FC1 );
   Mat warp_dst, warp_rotate_dst;

   /// 设置目标图像的大小和类型与源图像一致
   warp_dst = Mat::zeros( src.rows, src.cols, src.type() );

   /// 设置源图像和目标图像上的三组点以计算仿射变换
   srcTri[0] = Point2f( landmarks[37].x ,  landmarks[37].y);
   srcTri[1] = Point2f( landmarks[44].x ,  landmarks[44].y);
   srcTri[2] = Point2f( landmarks[8].x ,  landmarks[8].y);

   dstTri[0] = Point2f( src.cols*0.2, src.rows*0.2 );
   dstTri[1] = Point2f( src.cols*0.8, src.rows*0.2 );
   dstTri[2] = Point2f( src.cols*0.5, src.rows*0.99 );

   /// 求得仿射变换
   warp_mat = getAffineTransform( srcTri, dstTri );

   /// 对源图像应用上面求得的仿射变换
   warpAffine( src, warp_dst, warp_mat, warp_dst.size() );

   /** 对图像扭曲后再旋转 */

   /// 计算绕图像中点顺时针旋转50度缩放因子为0.6的旋转矩阵
   Point center = Point( warp_dst.cols/2, warp_dst.rows/2 );
   double angle = -50.0;
   double scale = 1;

   /// 通过上面的旋转细节信息求得旋转矩阵
   rot_mat = getRotationMatrix2D( center, angle, scale );

   /// 旋转已扭曲图像
   warpAffine( warp_dst, warp_rotate_dst, rot_mat, warp_dst.size() );

   /// 显示结果
/*
   namedWindow( "Source image", CV_WINDOW_AUTOSIZE );
   imshow(  "Source image", src );

   namedWindow( "Warp", CV_WINDOW_AUTOSIZE );
   imshow( "Warp", warp_dst );

   namedWindow( "Warp + Rotate", CV_WINDOW_AUTOSIZE );
   imshow(  "Warp + Rotate", warp_rotate_dst );
*/
   return warp_dst;
    }


