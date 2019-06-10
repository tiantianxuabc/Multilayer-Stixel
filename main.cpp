#include <iostream>
#include <stdint.h>
#include <stdio.h>
#include <bitset>
#include <smmintrin.h> // intrinsics
#include <emmintrin.h>
#include<fstream>
#include<stdio.h>

#include <opencv.hpp>

#include "StereoBMHelper.h"
#include "FastFilters.h"
#include "StereoSGM.h"



#include<time.h>
#include <vector>
#include <list>
#include <algorithm>
#include <numeric>
#include <iostream>

#include "multilayer_stixel_world.h"


#ifdef _DEBUG
#pragma comment(lib, "opencv_world400d")
#else
#pragma comment(lib, "opencv_world400")
#endif


const int dispRange = 128;


void jet(float x, int& r, int& g, int& b)
{
	if (x < 0) x = -0.05;
	if (x > 1) x = 1.05;
	x = x / 1.15 + 0.1; // use slightly asymmetric range to avoid darkest shades of blue.
	r = __max(0, __min(255, (int)(round(255 * (1.5 - 4 * fabs(x - .75))))));
	g = __max(0, __min(255, (int)(round(255 * (1.5 - 4 * fabs(x - .5))))));
	b = __max(0, __min(255, (int)(round(255 * (1.5 - 4 * fabs(x - .25))))));
}

cv::Mat Float2ColorJet(cv::Mat &fimg, float dmin, float dmax)
{

	int width = fimg.cols, height = fimg.rows;
	cv::Mat img(height, width, CV_8UC3);

	float scale = 1.0 / (dmax - dmin);

	for (int y = 0; y < height; y++) {
		for (int x = 0; x < width; x++) {
			float f = fimg.at<float>(y, x);
			int r = 0;
			int g = 0;
			int b = 0;

			/*if (f != INFINITY)*/ {
				float val = scale * (f - dmin);
				jet(val, r, g, b);
			}

			img.at<cv::Vec3b>(y, x) = cv::Vec3b(b, g, r);
		}
	}

	return img;
}




template<typename T>
void processCensus5x5SGM(T* leftImg, T* rightImg, float32* output, float32* dispImgRight,
	int width, int height, uint16 paths, const int dispCount)
{
	const int maxDisp = dispCount - 1;

	//std::cout << std::endl << paths << ", " << dispCount << std::endl;

	// get memory and init sgm params
	uint32* leftImgCensus = (uint32*)_mm_malloc(width*height*sizeof(uint32), 16);
	uint32* rightImgCensus = (uint32*)_mm_malloc(width*height*sizeof(uint32), 16);

	StereoSGMParams_t params;
	params.lrCheck = true;
	params.MedianFilter = true;
	params.Paths = paths;

	params.NoPasses = 2;

	
	uint16* dsi = (uint16*)_mm_malloc(width*height*(maxDisp + 1)*sizeof(uint16), 32);
	StereoSGM<T> m_sgm16(width, height, maxDisp, params);


	census5x5_16bit_SSE(leftImg, leftImgCensus, width, height);
	census5x5_16bit_SSE(rightImg, rightImgCensus, width, height);
	costMeasureCensus5x5_xyd_SSE(leftImgCensus, rightImgCensus, height, width, dispCount, params.InvalidDispCost, dsi);

	m_sgm16.process(dsi, leftImg, output, dispImgRight);
	_mm_free(dsi);
}

void onMouse(int event, int x, int y, int flags, void *param)
{
	cv::Mat *im = reinterpret_cast<cv::Mat *>(param);
	switch (event)
	{
	case cv::EVENT_LBUTTONDBLCLK:
		std::cout << "at (" << std::setw(3) << x << "," << std::setw(3) << y << ") value is: "
			<< static_cast<int>(im->at<uchar>(cv::Point(x, y))) << std::endl;
		break;
	}
}



int formatJPG(cv::Mat& imgL, cv::Mat& imgR, cv::Mat &imgDisp)
{
	int cols_ = imgL.cols;
	int rows_ = imgL.rows;


	uint16* leftImg = (uint16*)_mm_malloc(rows_*cols_*sizeof(uint16), 16);
	uint16* rightImg = (uint16*)_mm_malloc(rows_*cols_*sizeof(uint16), 16);
	for (int i = 0; i < rows_; i++)
	{
		for (int j = 0; j < cols_; j++)
		{
			leftImg[i * cols_ + j] = *(imgL.data + i*imgL.step + j * imgL.elemSize());
			rightImg[i * cols_ + j] = *(imgR.data + i*imgR.step + j * imgR.elemSize());
		}
	}

	//左右图像的视差图分配存储空间（width*height*sizeof(float32)）
	float32* dispImg = (float32*)_mm_malloc(rows_*cols_*sizeof(float32), 16);
	float32* dispImgRight = (float32*)_mm_malloc(rows_*cols_*sizeof(float32), 16);
	
	const int numPaths = 8;

	processCensus5x5SGM(leftImg, rightImg, dispImg, dispImgRight, cols_, rows_, numPaths, dispRange);	

	cv::Mat tmpDisp(cv::Size(cols_, rows_), CV_32FC1, dispImg);
	tmpDisp.copyTo(imgDisp);

	
	_mm_free(leftImg);
	_mm_free(rightImg);

	return 0;
}

static cv::Scalar computeColor(float val)
{
	const float hscale = 6.f;
	float h = 0.6f * (1.f - val), s = 1.f, v = 1.f;
	float r, g, b;

	static const int sector_data[][3] =
	{ { 1, 3, 0 }, { 1, 0, 2 }, { 3, 0, 1 }, { 0, 2, 1 }, { 0, 1, 3 }, { 2, 1, 0 } };
	float tab[4];
	int sector;
	h *= hscale;
	if (h < 0)
		do h += 6; while (h < 0);
	else if (h >= 6)
		do h -= 6; while (h >= 6);
	sector = cvFloor(h);
	h -= sector;
	if ((unsigned)sector >= 6u)
	{
		sector = 0;
		h = 0.f;
	}

	tab[0] = v;
	tab[1] = v*(1.f - s);
	tab[2] = v*(1.f - s*h);
	tab[3] = v*(1.f - s*(1.f - h));

	b = tab[sector_data[sector][0]];
	g = tab[sector_data[sector][1]];
	r = tab[sector_data[sector][2]];
	//return 255 * cv::Scalar(b, g, r);

	return cv::Scalar(255 * b, 255 * g, 255 * r);
}

static cv::Scalar dispToColor(float disp, float maxdisp)
{
	if (disp < 0)
		return cv::Scalar(128, 128, 128);
	return computeColor(std::min(disp, maxdisp) / maxdisp);
}


static void drawStixel(cv::Mat& img, const Stixel& stixel, cv::Scalar color)
{
	const int radius = std::max(stixel.width / 2, 1);
	const cv::Point tl(stixel.u - radius, stixel.vT);
	const cv::Point br(stixel.u + radius, stixel.vB);
	cv::rectangle(img, cv::Rect(tl, br), color, -1);
	cv::rectangle(img, cv::Rect(tl, br), cv::Scalar(255, 255, 255), 1);
}

int main()
{

	// input camera parameters
	const cv::FileStorage cvfs("camera.xml", cv::FileStorage::READ);
	CV_Assert(cvfs.isOpened());


	// input parameters
	MultiLayerStixelWrold::Parameters param;
	param.camera.fu = cvfs["FocalLengthX"];
	param.camera.fv = cvfs["FocalLengthY"];
	param.camera.u0 = cvfs["CenterX"];
	param.camera.v0 = cvfs["CenterY"];
	param.camera.baseline = cvfs["BaseLine"];
	param.camera.height = cvfs["Height"];
	param.camera.tilt = cvfs["Tilt"];
	param.dmax = dispRange;

	MultiLayerStixelWrold stixelWorld(param);


	std::string dir = "E:/Image Set/";

	cv::VideoWriter DemoWrite;
	//std::string outputName = dir + "0005.avi";

	for (int frameno = 0;; frameno++)
	{
		
		char base_name[256];
		sprintf(base_name, "%06d.png", frameno);
		std::string bufl = dir + "1/testing/0020/" + base_name;
		std::string bufr = dir + "2/testing/0020/" + base_name;

		std::cout << " " << frameno << std::endl;

		cv::Mat leftBGR = cv::imread(bufl, cv::IMREAD_COLOR);
		cv::Mat right = cv::imread(bufr, cv::IMREAD_GRAYSCALE);

		
		if (leftBGR.empty()  || right.empty())
		{
			std::cout << "Left image no exist!" << std::endl;
			frameno = 0;			
			continue;
			
			//break;
		}
		cv::Mat left;
		if (leftBGR.channels() == 3)
		{
			cv::cvtColor(leftBGR, left, cv::COLOR_BGR2GRAY);
		}
		else
		{
			left = leftBGR.clone();
		}
		
		CV_Assert(left.size() == right.size() && left.type() == right.type());


		cv::Rect roiRect(0, 0, left.cols - left.cols % 16, left.rows);
		cv::Mat leftROI(left, roiRect);
		cv::Mat rightROI(right, roiRect);
		cv::Mat showImage(leftBGR, roiRect);

		

		
		// calculate disparity SGM-Based
		cv::Mat imgDisp;
		formatJPG(leftROI, rightROI, imgDisp);


		const std::vector<Stixel> stixels = stixelWorld.compute(imgDisp);
	

		// draw stixels
		cv::Mat draw;
		cv::cvtColor(leftROI, draw, cv::COLOR_GRAY2BGRA);

		cv::Mat stixelImg = cv::Mat::zeros(leftROI.size(), draw.type());
		for (const auto& stixel : stixels)
			drawStixel(stixelImg, stixel, dispToColor(stixel.disp, 64));



		draw = draw + 0.5 * stixelImg;

		cv::imshow("disparity", Float2ColorJet(imgDisp, 0, dispRange));
		cv::imshow("stixels", draw);

		cv::waitKey(10);
	}
}