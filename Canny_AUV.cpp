#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"
#include <stdlib.h>
#include <stdio.h>
#include <string>

using namespace cv;
Mat src,result,detected_edges;
int edgeThresh = 1;
int lowThreshold;
int const max_lowThreshold = 100;
int ratio = 3;
int kernel_size = 3;
string window_name = "Canny Edge Detected Image";


Mat Canny_User(Mat src, int upperThreshold, int lowerThreshold, double size = 3)
{
	Mat workImg = Mat(src); workImg = src.clone(); 
	workImg = src.clone();
	GaussianBlur(src, workImg, Size(5, 5), 1.4); 
	Mat magX = Mat(src.rows, src.cols, CV_32F); 
	Mat magY = Mat(src.rows, src.cols, CV_32F); 
	Sobel(workImg, magX, CV_32F, 1, 0, size); 
	Sobel(workImg, magY, CV_32F, 0, 1, size); 
	Mat direction = Mat(workImg.rows, workImg.cols, CV_32F); 
	divide(magY, magX, direction);
	Mat sum = Mat(workImg.rows, workImg.cols, CV_64F);
	Mat prodX = Mat(workImg.rows, workImg.cols, CV_64F);
	Mat prodY = Mat(workImg.rows, workImg.cols, CV_64F);
	multiply(magX, magX, prodX); multiply(magY, magY, prodY);
	sum = prodX + prodY; sqrt(sum, sum); 
	Mat returnImg = Mat(src.rows, src.cols, CV_8U); 
	returnImg.setTo(Scalar(0));
	MatIterator_<float>itMag = sum.begin<float>();
    MatIterator_<float>itDirection = direction.begin<float>();
    MatIterator_<unsigned char>itRet = returnImg.begin<unsigned char>();
    MatIterator_<float>itend = sum.end<float>();

	for(;itMag!=itend;++itDirection, ++itRet, ++itMag)
	{ 
		const Point pos = itRet.pos(); 
		float currentDirection = atan(*itDirection) * 180 / 3.142; 
		while(currentDirection<0) currentDirection+=180; 
		*itDirection = currentDirection; if(*itMag<upperThreshold) continue; 
		bool flag = true;
		if(currentDirection>112.5 && currentDirection <=157.5) 
		{ 
			if(pos.y>0 && pos.x<workImg.cols-1 && *itMag<=sum.at<float>(pos.y-1, pos.x+1)) flag = false; 
			if(pos.y<workImg.rows-1 && pos.x>0 && *itMag<=sum.at<float>(pos.y+1, pos.x-1)) flag = false; 
		} 
		else if(currentDirection>67.5 && currentDirection <= 112.5) 
		{ 
			if(pos.y>0 && *itMag<=sum.at<float>(pos.y-1, pos.x)) flag = false; 
			if(pos.y<workImg.rows-1 && *itMag<=sum.at<float>(pos.y+1, pos.x)) flag = false; 
		} 
		else if(currentDirection > 22.5 && currentDirection <= 67.5) 
		{ 
			if(pos.y>0 && pos.x>0 && *itMag<=sum.at<float>(pos.y-1, pos.x-1)) flag = false; 
			if(pos.y<workImg.rows-1 && pos.x<workImg.cols-1 && *itMag<=sum.at<float>(pos.y+1, pos.x+1)) flag = false; 
		} 
		else { 
			if(pos.x>0 && *itMag<=sum.at<float>(pos.y, pos.x-1)) flag = false; 
			if(pos.x<workImg.cols-1 && *itMag<=sum.at<float>(pos.y, pos.x+1)) flag = false; 
		} 
		if(flag) { 
			*itRet = 255; 
		} 
	}
	bool imageChanged = true; 
	int i=0; 
	while(imageChanged) { 
	imageChanged = false; 
	i++; 
	itMag = sum.begin<float>(); 
	itDirection = direction.begin<float>(); 
	itRet = returnImg.begin<unsigned char>(); 
	itend = sum.end<float>(); 
	for(;itMag!=itend;++itMag, ++itDirection, ++itRet) 
	{
		 Point pos = itRet.pos(); 
		 if(pos.x<2 || pos.x>src.cols-2 || pos.y<2 || pos.y>src.rows-2) continue; 
		 float currentDirection = *itDirection;
		 if(*itRet==255) { 
		 	*itRet=(unsigned char)64;
		 	if(currentDirection>112.5 && currentDirection <= 157.5) 
		 	{ 
		 		if(pos.y>0 && pos.x>0) 
		 		{ 
		 			if(lowerThreshold<=sum.at<float>(pos.y-1, pos.x-1) && 
		 				returnImg.at<unsigned char>(pos.y-1, pos.x-1)!=64 && 
		 				direction.at<float>(pos.y-1, pos.x-1) > 112.5 && 
		 				direction.at<float>(pos.y-1, pos.x-1) <= 157.5 && 
		 				sum.at<float>(pos.y-1, pos.x-1) > sum.at<float>(pos.y-2, pos.x) && 
		 				sum.at<float>(pos.y-1, pos.x-1) > sum.at<float>(pos.y, pos.x-2)) 
		 			{ 
		 				returnImg.ptr<unsigned char>(pos.y-1, pos.x-1)[0] = 255; 
		 				imageChanged = true; 
		 			} 
		 		} 
		 		if(pos.y<workImg.rows-1 && pos.x<workImg.cols-1) 
		 		{ 
		 			if(lowerThreshold<=sum.at<float>(Point(pos.x+1, pos.y+1)) && 
		 				returnImg.at<unsigned char>(pos.y+1, pos.x+1)!=64 && 
		 				direction.at<float>(pos.y+1, pos.x+1) > 112.5 && 
		 				direction.at<float>(pos.y+1, pos.x+1) <= 157.5 && 
		 				sum.at<float>(pos.y-1, pos.x-1) > sum.at<float>(pos.y+2, pos.x) && 
		 				sum.at<float>(pos.y-1, pos.x-1) > sum.at<float>(pos.y, pos.x+2)) 
		 			{ 
		 				returnImg.ptr<unsigned char>(pos.y+1, pos.x+1)[0] = 255; 
		 				imageChanged = true; 
		 			} 
		 		} 
		 	} 
		 	else if(currentDirection>67.5 && currentDirection <= 112.5) 
		 	{ 
		 		if(pos.x>0) 
		 		{ 
		 			if(lowerThreshold<=sum.at<float>(Point(pos.x-1, pos.y)) && 
		 				returnImg.at<unsigned char>(pos.y, pos.x-1)!=64 && 
		 				direction.at<float>(pos.y, pos.x-1) > 67.5 && 
		 				direction.at<float>(pos.y, pos.x-1) <= 112.5 && 
		 				sum.at<float>(pos.y, pos.x-1) > sum.at<float>(pos.y-1, pos.x-1) && 
		 				sum.at<float>(pos.y, pos.x-1) > sum.at<float>(pos.y+1, pos.x-1)) 
		 			{ 
		 				returnImg.ptr<unsigned char>(pos.y, pos.x-1)[0] = 255; 
		 				imageChanged = true; 
		 			} 
		 		} 
		 		if(pos.x<workImg.cols-1) 
		 		{ 
		 			if(lowerThreshold<=sum.at<float>(Point(pos.x+1, pos.y)) && 
		 				returnImg.at<unsigned char>(pos.y, pos.x+1)!=64 && 
		 				direction.at<float>(pos.y, pos.x+1) > 67.5 && 
		 				direction.at<float>(pos.y, pos.x+1) <= 112.5 && 
		 				sum.at<float>(pos.y, pos.x+1) > sum.at<float>(pos.y-1, pos.x+1) && 
		 				sum.at<float>(pos.y, pos.x+1) > sum.at<float>(pos.y+1, pos.x+1)) 
		 			{ 
		 				returnImg.ptr<unsigned char>(pos.y, pos.x+1)[0] = 255; 
		 				imageChanged = true; 
		 			} 
		 		} 
		 	} 
		 	else if(currentDirection > 22.5 && currentDirection <= 67.5) 
		 	{ 
		 		if(pos.y>0 && pos.x<workImg.cols-1) 
		 		{ 
		 			if(lowerThreshold<=sum.at<float>(Point(pos.x+1, pos.y-1)) && 
		 				returnImg.at<unsigned char>(pos.y-1, pos.x+1)!=64 && 
		 				direction.at<float>(pos.y-1, pos.x+1) > 22.5 && 
		 				direction.at<float>(pos.y-1, pos.x+1) <= 67.5 && 
		 				sum.at<float>(pos.y-1, pos.x+1) > sum.at<float>(pos.y-2, pos.x) && 
		 				sum.at<float>(pos.y-1, pos.x+1) > sum.at<float>(pos.y, pos.x+2)) 
		 			{ 
		 				returnImg.ptr<unsigned char>(pos.y-1, pos.x+1)[0] = 255; 
		 				imageChanged = true; 
		 			} 
		 		} 
		 		if(pos.y<workImg.rows-1 && pos.x>0) 
		 		{ 
		 			if(lowerThreshold<=sum.at<float>(Point(pos.x-1, pos.y+1)) && 
		 				returnImg.at<unsigned char>(pos.y+1, pos.x-1)!=64 && 
		 				direction.at<float>(pos.y+1, pos.x-1) > 22.5 && 
		 				direction.at<float>(pos.y+1, pos.x-1) <= 67.5 && 
		 				sum.at<float>(pos.y+1, pos.x-1) > sum.at<float>(pos.y, pos.x-2) && 
		 				sum.at<float>(pos.y+1, pos.x-1) > sum.at<float>(pos.y+2, pos.x)) 
		 			{ 
		 				returnImg.ptr<unsigned char>(pos.y+1, pos.x-1)[0] = 255; 
		 				imageChanged = true; 
		 			} 
		 		} 
		 	} 
		 	else 
		 	{ 
		 		if(pos.y>0) 
		 		{ 
		 			if(lowerThreshold<=sum.at<float>(Point(pos.x, pos.y-1)) && 
		 				returnImg.at<unsigned char>(pos.y-1, pos.x)!=64 && 
		 				(direction.at<float>(pos.y-1, pos.x) < 22.5 || direction.at<float>(pos.y-1, pos.x) >=157.5) && 
		 				sum.at<float>(pos.y-1, pos.x) > sum.at<float>(pos.y-1, pos.x-1) && 
		 				sum.at<float>(pos.y-1, pos.x) > sum.at<float>(pos.y-1, pos.x+2)) 
		 			{ 
		 				returnImg.ptr<unsigned char>(pos.y-1, pos.x)[0] = 255; 
		 				imageChanged = true; 
		 			} 
		 		} 
		 		if(pos.y<workImg.rows-1) 
		 		{ 
		 			if(lowerThreshold<=sum.at<float>(Point(pos.x, pos.y+1)) && 
		 				returnImg.at<unsigned char>(pos.y+1, pos.x)!=64 && 
		 				(direction.at<float>(pos.y+1, pos.x) < 22.5 || direction.at<float>(pos.y+1, pos.x) >=157.5) && 
		 				sum.at<float>(pos.y+1, pos.x) > sum.at<float>(pos.y+1, pos.x-1) && 
		 				sum.at<float>(pos.y+1, pos.x) > sum.at<float>(pos.y+1, pos.x+1)) 
		 			{ 
		 					returnImg.ptr<unsigned char>(pos.y+1, pos.x)[0] = 255; 
		 					imageChanged = true; 
		 			} 
		 		} 
		 	} 
		 } 
		} 
	}
	MatIterator_<unsigned char>current = returnImg.begin<unsigned char>(); 
	MatIterator_<unsigned char>final = returnImg.end<unsigned char>(); 
	for(;current!=final;++current) 
	{ 
		if(*current==64) *current = 255; 
	} 
	return returnImg; 
}

void Threshold(int, void *)
{
result=Canny_User( detected_edges, lowThreshold, lowThreshold*ratio, kernel_size );
imshow( window_name, result );
}

int main( int argc, char ** argv )
{
src = imread( argv[1] );
if( !src.data )
{ return -1; }
cvtColor( src, detected_edges, CV_BGR2GRAY );
namedWindow( window_name, CV_WINDOW_AUTOSIZE );
createTrackbar( "Min Threshold:", window_name, &lowThreshold, max_lowThreshold, Threshold );
Threshold(0, 0);
waitKey(0);
return 0;
}
