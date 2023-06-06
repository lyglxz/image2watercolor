#include <iostream>
#include <opencv2/opencv.hpp>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/calib3d.hpp>
#include <vector>

using namespace std;
using namespace cv;

double timer;

void printElapsedTime(double& timer, const string& stage) {
	cout << setprecision(5) << stage + "\t" << (getTickCount() - timer) / getTickFrequency()*1000 << " ms" << endl;
	timer = getTickCount();
}

void PerformMeanshift(Mat &image) {
    pyrMeanShiftFiltering(image, image, 5, 20, 1, TermCriteria(TermCriteria::MAX_ITER+TermCriteria::EPS, 2, 1));
}

void PerformMorph(Mat &image) {
    cv::Mat element = getStructuringElement( MORPH_ELLIPSE,
                    Size( 5, 5 ));
    morphologyEx( image, image, MORPH_CLOSE, element, Point(-1,-1), 1);
    morphologyEx( image, image, MORPH_OPEN, element, Point(-1,-1),1);
    morphologyEx( image, image, MORPH_CLOSE, element, Point(-1,-1),1);
    morphologyEx( image, image, MORPH_OPEN, element, Point(-1,-1),1);
}

void Abstraction(Mat &image) {
    PerformMorph(image);
    printElapsedTime(timer, "Morph");
    PerformMeanshift(image);
    printElapsedTime(timer, "Mean shift");
}

void Rendering(Mat &image, Mat noise) {
    auto no = noise;
    image.convertTo(image, CV_32F, 1.0/255.0);
    noise.convertTo(no, CV_32F, 2.0/255.0);
    // cout << image.size();
    auto seg1 = cv::Scalar(1.0)-image;
    auto seg2 = no - cv::Scalar(1.0);
    seg1 = Scalar(1.0) - seg1.mul(seg2);
    image = image.mul(seg1);
    image.convertTo(image, CV_8UC1, 255);
    
}

void EdgeDarken(Mat &image, Mat noise, Mat edge) {
    auto no = noise;
    auto edge_f = edge;
    image.convertTo(image, CV_32F, 1.0/255.0);
    noise.convertTo(no, CV_32F, 2.0/255.0);
    edge.convertTo(edge_f, CV_32F, 0.25/255.0);

    // cout << image.size();
    auto seg1 = cv::Scalar(1.0)-image;
    auto seg3 = edge_f + cv::Scalar(1.0);
    auto seg2 = no.mul(seg3) - cv::Scalar(1.0);
    seg1 = Scalar(1.0) - seg1.mul(seg2);
    image = image.mul(seg1);
    image.convertTo(image, CV_8UC1, 255);
    
}

void Wobble(Mat &image, Mat &out, Mat &texture, Mat &edge) {
    for(int i = 0; i < image.rows; ++i) {
        for (int j = 0; j < image.cols; ++j) {
            if (edge.at<uchar>(i, j) == 0) {
                continue;
            }
            else {
                int offset = int(0.05*(texture.at<uchar>(i,j) - 128));
                // if (offset > 1) cout << offset << endl;
                out.at<Vec3b>(i,j)[0] = image.at<Vec3b>(i+offset,j)[0];
                out.at<Vec3b>(i,j)[1] = image.at<Vec3b>(i+offset,j)[1];
                out.at<Vec3b>(i,j)[2] = image.at<Vec3b>(i+offset,j)[2];
            }
        }
    }
}


bool watercolorize(Mat &image) {
	timer = getTickCount();
	// totalTimer = timer;
    vector<cv::Mat> channels;
    vector<cv::Mat> channel_edge;
    cv::Mat edge;
    cv::Mat output;
    cv::Mat noise = cv::imread("../img/noise.jpg", 
        cv::IMREAD_GRAYSCALE);
    cv::Mat wobble = cv::imread("../img/perlin_36.jpg", cv::IMREAD_GRAYSCALE);
    cv::Canny(image, edge, 150, 200);
    cv::Mat element = getStructuringElement( MORPH_ELLIPSE,
                    Size( 3, 3 ));
    morphologyEx( edge, edge, cv::MORPH_DILATE, element, Point(-1,-1), 1);
    cv::imwrite("../img/out_edge.jpg",edge);
    // std::cout << edge.dtype << std::endl;
    Abstraction(image);
    split(image, channels);
    // split(edge, channel_edge);


    // Rendering(channels[0], noise);
    // // cv::imwrite("../img/o0.jpg",channels[0]);
    // Rendering(channels[1], noise);
    // // cv::imwrite("../img/o1.jpg",channels[1]);
    // Rendering(channels[2], noise);
    // // cv::imwrite("../img/o2.jpg",channels[2]);



    EdgeDarken(channels[0], noise, edge);
    // cv::imwrite("../img/o0.jpg",channels[0]);
    EdgeDarken(channels[1], noise, edge);
    // cv::imwrite("../img/o1.jpg",channels[1]);
    EdgeDarken(channels[2], noise, edge);
    // cv::imwrite("../img/o2.jpg",channels[2]);
    merge(channels, image);
    // cv::imwrite("../img/out_darken05.jpg",image);
    printElapsedTime(timer, "Render");


    output = image.clone();
    Wobble(image, output, wobble, edge);
    printElapsedTime(timer, "Wobble");
    cv::imwrite("../img/out_wobble.jpg",output);
    // cout << image.size();
    // image = ret;
    return true;
}

int main(int argc, char* argv[]) {

    cout << "start\n";
	if(argc < 2 || argc > 4) {
		return 0;
	}
	Mat image = imread(argv[1], IMREAD_COLOR);

	if(!image.data) {
		cerr << "Cannot load image." << endl;
		return -1;
	}
    // render pipeline
	watercolorize(image);
	bool writeSuccess = false;
	try {
		if(argc == 2) {
			writeSuccess = imwrite("./output.jpg", image);
		}
		else {
			writeSuccess = imwrite(argv[2], image);
		}
	}
	catch (const cv::Exception& ex) {
		//so what?
		cerr << ex.what() << endl;
	}

	if(!writeSuccess) {
		cerr << "Cannot save image." << endl;
		return -1;
	}
	cout << "Done" << endl;
    return 0;
}


