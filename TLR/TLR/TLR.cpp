#include<opencv2/opencv.hpp>
#include<opencv2/core/core.hpp>
#include<opencv2/highgui/highgui.hpp>
#include<opencv2/features2d.hpp>
#include<opencv2/imgproc/imgproc.hpp>
#include<iostream>
#include<vector> 
#include<string>

using namespace cv;
using namespace std;

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
static Scalar colors[]
{
	{ 255, 255, 255 },
	{ 0, 128, 255 },
	{ 0, 255, 0 },
	{ 255, 0, 0 },
	{ 0, 0, 255 },//5
	{ 0, 255, 255 },
	{ 128, 255, 0 },
	{ 255, 128, 0 },
	{ 22, 255, 0 }
};

//overlap ratio
float OverlapRatio( Rect box1, Rect box2)
{
	if (box1.x > box2.x + box2.width) { return 0.0; }
	if (box1.y > box2.y + box2.height) { return 0.0; }
	if (box1.x + box1.width < box2.x) { return 0.0; }
	if (box1.y + box1.height < box2.y) { return 0.0; }
	float colInt = min(box1.x + box1.width, box2.x + box2.width) - max(box1.x, box2.x);
	float rowInt = min(box1.y + box1.height, box2.y + box2.height) - max(box1.y, box2.y);
	float intersection = colInt * rowInt;
	float area = box1.width*box1.height;
	return intersection / area;
}

//filter rectangles with high overlap ratio
void CombinedHighOverlappedRect( vector<Rect>& box )
{
	for (auto i = box.begin(); i != box.end() ; i++)
	{
		for (auto j = i + 1; j != box.end(); )
		{
			if (  OverlapRatio((*i), (*j)) >= 0.4){
				j = box.erase(j);
			}
			else{
				++j;
			}
		}
	}
}

//ignore thin bars
void reduceThinBars(Mat& inputImg, Rect& rect){//传灰度图
	int flag = 1;
	Mat mat(rect.width, rect.height, CV_8UC1);
	imshow("1234", inputImg);
	mat = Mat(inputImg, rect);//小灰度图
	
	//原中心在现矩形中的坐标
	int x = inputImg.cols / 2 - rect.x;
	int y = inputImg.rows / 2 - rect.y;

	Mat rowsSum(mat.cols, 1, CV_64FC1, Scalar(0));
	Mat colsSum(1, mat.rows, CV_64FC1, Scalar(0));

	reduce(mat, rowsSum, 1, CV_REDUCE_SUM, CV_32S);
	//cout << rowsSum / 255 << endl;
	reduce(mat, colsSum, 0, CV_REDUCE_SUM, CV_32S);
	//cout << colsSum / 255 << endl;

	if (1.0*rowsSum.at<int>(y, 0) / 255 / mat.cols >= 0.6){
		flag = 0;//horizon
		//修改rect高度
		int i=0, j=0;

		for (i = y; i >= 0; i--){
			float sum = 1.0*rowsSum.at<int>(i, 0) / 255;
			float sum2 = 1.0*rowsSum.at<int>(i, 0) / 255 / mat.cols;

			if (1.0*rowsSum.at<int>(i, 0) / 255 / mat.cols < 0.6){
				rect.y += i;
				break;
			}
		}
		for (j = y; j <= mat.rows; j++){
			if (1.0*rowsSum.at<int>(j, 0) / 255 / mat.cols < 0.6){
				rect.height = j - i;
				break;
			}
		}
	}
	else if (1.0*colsSum.at<int>(0, x) / 255 / mat.rows >= 0.6){
		flag = 1;
		int i=0, j=0;
		for (i = x; i >= 0; i--){
			if (1.0*colsSum.at<int>(0, i) / 255 / mat.rows < 0.6){
				rect.x += i;
				break;
			}
		}
		for (j = x; j <= mat.cols; j++){
			if (1.0*colsSum.at<int>(0, j) / 255 / mat.rows < 0.6){
				rect.width = j - i;
				break;
			}
		}
	}

}

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
int main(){

	// 1 Color Segmentation-------------------------------------------------------------------------------------------------------------------------------------------

	int rLowH = 0;
	int rHighH = 18;

	int LowS = 140;
	int HighS = 255;

	int LowV = 140;
	int HighV = 255;//

	int rLowH2 = 156;
	int rHighH2 = 180;//

	int yLowH = 11;
	int yHighH = 34;//

	int gLowH = 35;
	int gHighH = 100;

	vector<Mat> imgSplit;
	Mat imgRGB = imread("17.jpg");
	Mat imgRGB2 = imread("17.jpg");
	Mat imgRGB3 = imread("17.jpg");
	Mat imgRGB4 = imread("17.jpg");
	Mat imgRGB5 = imread("17.jpg");
	Mat imgHSV, imgHSV2;
	Mat imgThresholded;
	Mat Thresholded[4];
	vector<Mat> hsvSplit;

	vector<vector<Point> > contours;
	vector<Vec4i> hierarchy;

	Scalar color = Scalar(255, 0, 255);
	Mat element1 = getStructuringElement(MORPH_RECT, Size(2, 2));
	Mat element2 = getStructuringElement(MORPH_RECT, Size(2, 2));

	//imshow("yuantu", imgRGB);


	//RGB三通道分离  
	split(imgRGB, imgSplit);
	//求原始图像的RGB分量的均值  
	double R, G, B;
	B = mean(imgSplit[0])[0];
	G = mean(imgSplit[1])[0];
	R = mean(imgSplit[2])[0];
	//需要调整的RGB分量的增益  
	double KR, KG, KB;
	KB = (R + G + B) / (3 * B);
	KG = (R + G + B) / (3 * G);
	KR = (R + G + B) / (3 * R);
	//调整RGB三个通道各自的值  
	imgSplit[0] = imgSplit[0] * KB;
	imgSplit[1] = imgSplit[1] * KG;
	imgSplit[2] = imgSplit[2] * KR;
	//RGB三通道图像合并  
	merge(imgSplit, imgRGB);


	cvtColor(imgRGB, imgHSV, CV_BGR2HSV);
	cvtColor(imgRGB, imgHSV2, CV_BGR2HSV);
	split(imgHSV, hsvSplit);
	equalizeHist(hsvSplit[2], hsvSplit[2]);
	merge(hsvSplit, imgHSV);

	inRange(imgHSV, Scalar(rLowH, LowS, LowV), Scalar(rHighH, HighS, HighV), Thresholded[0]);
	inRange(imgHSV, Scalar(rLowH2, LowS, LowV), Scalar(rHighH2, HighS, HighV), Thresholded[1]);
	inRange(imgHSV, Scalar(yLowH, LowS, LowV), Scalar(yHighH, HighS, HighV), Thresholded[2]);
	inRange(imgHSV, Scalar(gLowH, LowS, LowV), Scalar(gHighH, HighS, HighV), Thresholded[3]);//Threshold the imag
	
	imgThresholded = Thresholded[0] | Thresholded[1] |Thresholded[2] | Thresholded[3];

	imshow("imgThreshold",imgThresholded);

	// 2 Shape Filtering------------------------------------------------------------------------------------------------------------------------------------------------

	float ratio = 0.0;
	float area = 0.0;
	int countLightPiexls=0;
	morphologyEx(imgThresholded, imgThresholded, MORPH_OPEN, element1);
	morphologyEx(imgThresholded, imgThresholded, MORPH_CLOSE, element1);

	findContours(imgThresholded, contours, hierarchy, CV_RETR_TREE, CV_CHAIN_APPROX_SIMPLE, Point(0, 0));

	vector<vector<Point> > contoursPoly(contours.size());
	vector<Rect> TLboundBoxes(contours.size());

	for (int i = 0; i < contours.size(); i++)
	{
		approxPolyDP(Mat(contours[i]), contoursPoly[i], 3, true);
		TLboundBoxes[i] = boundingRect(Mat(contoursPoly[i]));
		rectangle(imgRGB, TLboundBoxes[i], Scalar(255, 0, 255), 1);
	}
	//imshow("TLBound original",imgRGB);
	//
	auto iter2 = contoursPoly.begin();
	for (auto iter = TLboundBoxes.begin(); iter != TLboundBoxes.end();)
	{
		ratio = max(iter->height, iter->width) / min(iter->height, iter->width);//aspect ratio
		area = (iter->height)*(iter->width);
		countLightPiexls = countNonZero(Mat(imgThresholded, (*iter)));
		if (ratio > 2 || area <= 8 || area > 8100 || (countLightPiexls / area) < 0.2){
			iter = TLboundBoxes.erase(iter);
			iter2 = contoursPoly.erase(iter2);
		} else{
			++iter;
			++iter2;
		}	
	}
	vector<Point> centerPoints;// centers of rectangles
	for (int i = 0; i < TLboundBoxes.size(); i++)
	{
		centerPoints.push_back(Point((TLboundBoxes[i].x + TLboundBoxes[i].width / 2), (TLboundBoxes[i].y + TLboundBoxes[i].height / 2)));
		rectangle(imgRGB2, TLboundBoxes[i].tl(), TLboundBoxes[i].br(), color, 1, 8, 0);
	}	
	
	//imshow("TLBound after shaping", imgRGB2);

	// 3 Bounding Box of detected traffic lights---------------------------------------------------------------------------------------------------------------

	vector<Rect> TLStructureBoundBoxes;
	int x = 0;
	int y = 0;
	int width = 90;
	int height = 160;
	for (auto i = 0; i < centerPoints.size(); i++)
	{	
		centerPoints[i].x - 45 < 0 ? x = 0 : x = centerPoints[i].x - 45;
		centerPoints[i].y - 80 < 0 ? y = 0 : y = centerPoints[i].y - 80;
		centerPoints[i].x + 45 > imgRGB.cols ? width = imgRGB.cols - centerPoints[i].x + 45 : width = 90;
		centerPoints[i].y + 80 > imgRGB.rows ? height = imgRGB.rows - centerPoints[i].y + 80 : height = 160;

		TLStructureBoundBoxes.push_back(Rect(x, y, width, height));
	}

	CombinedHighOverlappedRect(TLStructureBoundBoxes);// erase overlapped rectangles

	for (auto i = 0; i < TLStructureBoundBoxes.size(); i++)
	{
		rectangle(imgRGB3, TLStructureBoundBoxes[i], Scalar(255, 0, 255), 1);
	}

	imshow("TLStruture", imgRGB3);

	// 4 MSER to each TLStructureBox------------------------------------------------------------------------------------------------------------------------------------------
	
	int bHighH;
	int bLowH;
	int bHighS;
	int bLowS;
	int bHighV;
	int bLowV;

	Mat src;
	vector<Mat> imgSplit2;
	Mat Thresholded2[3];
	Mat imgThresholded2;
	vector<vector<Point> > contours2;
	vector<Vec4i> hierarchy2;
	int count = 0;
	int nnn = 0;
	for (auto i = TLStructureBoundBoxes.begin(); i != TLStructureBoundBoxes.end(); )
	{		
		src = Mat(imgRGB4, (*i));
		split(src, imgSplit2);
		for (int j = 0; j < 3; j++){
			equalizeHist(imgSplit2[j], imgSplit2[j]);
			threshold(imgSplit2[j], Thresholded2[j], 30, 255, THRESH_BINARY_INV);
		}
		imgThresholded2 = Thresholded2[0] | Thresholded2[1] | Thresholded2[2];
		//rectangle(imgThresholded2, Rect(Point(src.cols / 2 - 7, src.rows / 2 - 7), Point(src.cols / 2 + 7, src.rows / 2 + 7)), Scalar(255), -1);//补全灯	
		//
		//imshow("structure"+to_string(i), imgThresholded2);
		//++count;			
		//moveWindow(to_string(count), TLStructureBoundBoxes[i].x, TLStructureBoundBoxes[i].y);
		//
		cv::Mat tmp = imgThresholded2.clone();
		findContours(tmp, contours2, hierarchy2, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_SIMPLE, Point(0, 0));
		vector<vector<Point> > contoursPoly2(contours2.size());
		vector<Rect> TLboundBoxes2(contours2.size());
		for (int j = 0; j < contours2.size(); j++)
		{
			approxPolyDP(Mat(contours2[j]), contoursPoly2[j], 3, true);
			TLboundBoxes2[j] = boundingRect(Mat(contoursPoly2[j]));
		}
		//
		
		for (auto k = TLboundBoxes2.begin(); k != TLboundBoxes2.end();){
			if ((*k).contains(Point((*i).width / 2, (*i).height / 2))){	
				//cout << "00000原图中的中心点："<<centerPoints[i] << endl;
				
				if (nnn != 0){
					rectangle(imgRGB5, Rect((*k).x + (*i).x, (*k).y + (*i).y, (*k).width, (*k).height), Scalar(255, 0, 255), 1);
				}
				nnn++;
				reduceThinBars(imgThresholded2, (*k));
				if ((*k).area() > 300){
					rectangle(imgRGB5, Rect((*k).x + (*i).x, (*k).y + (*i).y, (*k).width, (*k).height), Scalar(0, 255, 255), 2);
					++k;
				}
				else{
					k = TLboundBoxes2.erase(k);
				}			
				cout << "----------------------------------------------------------------------" << endl;
			} 
			else{
				k = TLboundBoxes2.erase(k);
			}

		}
		//如果在structure中没有找到符合条件的红绿灯
		if ( TLboundBoxes2.size() == 0 ){
			i = TLStructureBoundBoxes.erase(i);
		}
		else{
			++count;
			imshow("src" + to_string(count), src);
			++i;
		}			
	}
	for (auto i = 0; i < TLStructureBoundBoxes.size(); i++)
	{
		rectangle(imgRGB5, TLStructureBoundBoxes[i], Scalar(255, 0, 255), 1);
	}
	imshow("img5", imgRGB5);

	waitKey(0);
	return 0;
}



//cvtColor(src, imgHSV2, CV_BGR2HSV);
//
//Ptr<MSER> ms = MSER::create(2, 60, 14400, 0.3, 0.3, 100, 1.007, 0.003, 3);   //5,60,14400,0.25,0.2,200,0.5,0.003,5 ;5, 60, 14400, 0.25, 0.2, 80, 1.06, 0.02, 5
////
//ms->detectRegions(imgHSV2, regions, boundBoxes);
////s
//cout << "region size : " << regions.size() << endl;
//for (int j = 0; j < regions.size(); j++){
//	drawContours(src, regions, j, colors[j % 9], -1);
//	rectangle(src,boundBoxes[j],Scalar(255,0,255),1);
//	cout << i << "---" << j << endl;
//}		








/*cout << TLboundBoxes.size() << endl;
int count = 0;
for (auto i = 0; i < TLboundBoxes.size(); i++)
{
cout <<count++<<" "<< TLboundBoxes[i] << endl;
}
cout << centerPoints.size() << endl;
count = 0;
for (auto i = 0; i < centerPoints.size(); i++)
{
cout << count++ << " " << centerPoints[i] << endl;
}
cout << TLStructureBoundBoxes.size() << endl;
count = 0;
for (auto i = 0; i < TLStructureBoundBoxes.size(); i++)
{
cout << count++ << " " << TLStructureBoundBoxes[i] << endl;
}*/






/*

Mat imgGray, imgGray_neg;
cvtColor(imgRGB, imgGray, CV_BGR2GRAY);
imgGray_neg = 255 - imgGray;

vector<vector<Point> > contours1;
vector<vector<Point> > contours2;
vector<Rect> boundBoxes1;
vector<Rect> boundBoxes2;
vector<vector<Point>> clusterContours;
vector<Rect> candidates;


Ptr<MSER> mesr1 = MSER::create();//2, 100, 5000, 0.5, 0.3
Ptr<MSER> mesr2 = MSER::create();//3, 100, 500, 0.25, 0.2, 200, 1.01, 0.003, 5

mesr1->detectRegions(imgGray, contours1, boundBoxes1);
mesr2->detectRegions(imgGray_neg, contours2, boundBoxes2);

Mat mserGray = Mat::zeros(imgGray.size(), CV_8UC1);
Mat mserGray_neg = Mat::zeros(imgGray_neg.size(), CV_8UC1);

// MSER - imgGray  -----------------------------------------

for (int i = 0; i < contours1.size(); i++)
{
const vector<Point>& r = contours1[i];
for (int j = 0; j < r.size(); j++)
{
Point pt = r[j];
mserGray.at<unsigned char>(pt) = 255;
}
}
cout << "boundBoxes1:" << boundBoxes1.size() << endl;
for (auto i = boundBoxes1.begin(); i != boundBoxes1.end(); )
{
ratio = max((*i).width, (*i).height) / min((*i).width, (*i).height);
area = ((*i).width) * ((*i).height);
if ((*i).height > 20 && ratio > 1 && ratio < 5 && area > 200){
rectangle(imgGray, (*i).tl(), (*i).br(), Scalar(0, 0, 0), 2, 8, 0);
++i;
}
else{
i = boundBoxes1.erase(i);
}
}
cout << "boundBoxes1:" << boundBoxes1.size() << endl;

imshow("imgGray_drawing", imgGray);


//MSER - imgGray_neg ---------------------------------------

for (int i = 0; i < contours2.size(); i++)
{
const vector<Point>& r = contours2[i];
for (int j = 0; j < r.size(); j++)
{
cv::Point pt = r[j];
mserGray_neg.at<unsigned char>(pt) = 255;
}
}

//MSER - mserResult

Mat mserResult;
mserResult = mserGray & mserGray_neg;

Canny(mserResult, mserResult, 3, 9);

imshow("mserResult_Canny", mserResult);

findContours(mserResult, clusterContours, CV_RETR_TREE, CV_CHAIN_APPROX_NONE, cv::Point(0, 0));

Rect rect;
for (size_t i = 0; i != clusterContours.size(); ++i)
{
rect = boundingRect(clusterContours[i]);
ratio = max(rect.width, rect.height) / min(rect.width, rect.height);
area = rect.width* rect.height;
if (rect.height > 20 && ratio > 1 && ratio < 5 && area > 200){
candidates.push_back(rect);
rectangle(mserResult, rect.tl(), rect.br(), color, 1, 8, 0);
}
}

imshow("mserResult_Canny_candidates", mserResult);

// 4 Localization -------------------------------------------------------------------------------------------------------------------------

cout << candidates.size() << endl;
bool flag = false;
for (auto i = candidates.begin(); i != candidates.end();)
{

flag = false;
for (auto j = centerPoints.begin(); j != centerPoints.end(); j = j + 2)
{
if ((*i).contains(*j) && (*i).contains(*(j + 1))){
flag = true;
break;
}
}
if (flag == false){
i = candidates.erase(i);
}
else{
++i;
}
}
cout << candidates.size() << endl;

for (auto i = candidates.begin(); i != candidates.end(); i++)
{
rectangle(imgRGB2, (*i).tl(), (*i).br(), Scalar(0, 0, 255), 2, 8, 0);
}

imshow("mserResult_Canny_candidates_afterLocalization", imgRGB2);

cout << "boundBoxes1:" << boundBoxes1.size() << endl;
for (auto i = boundBoxes1.begin(); i != boundBoxes1.end();)
{

flag = false;

for (auto j = centerPoints.begin(); j != centerPoints.end(); j = j + 2)
{
if ( (*i).contains(*j) && (*i).contains(*(j + 1)) ){
flag = true;
break;
}
}
if (flag == false){
i = boundBoxes1.erase(i);
}
else{
++i;
}
}
cout <<"boundBoxes1:"<< boundBoxes1.size() << endl;
for (auto i = boundBoxes1.begin(); i != boundBoxes1.end(); i++)
{
rectangle(imgRGB3, (*i).tl(), (*i).br(), Scalar(255, 0, 0), 2, 8, 0);
}

imshow("boundBoxes1_afterLocalization", imgRGB3);

cout << "end------------------------------------------------------------------------------" << endl;

*/