#include <fstream>
#include <sstream>
#include <iostream>


#include <opencv2/dnn.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/dnn/dnn.hpp>
#include <opencv2/imgcodecs.hpp>


using namespace std;
using namespace cv;
using namespace dnn;


int main() {
	try{

	vector<string> classes;
	ifstream ifs("classes.txt"); // Path to the class names file
	if (!ifs.is_open()) {
		cerr << "Error opening classes.txt" << endl;
		return -1;
	}
	string line;
	while (getline(ifs, line)) {
		classes.push_back(line);
	}

	string path = "a1.jpg"; //reading and loading the image
	Mat img = imread(path);
	Mat rgbimg;
	cvtColor(img, rgbimg, COLOR_BGR2RGB); // converting image to RGB channel

	Mat resized_img;
	resize(rgbimg, resized_img, Size(32, 32), INTER_LINEAR); //resizing the image into [32,32]

	//image preprocessing:
	float mean = 120.70063406575521;
	float std = 64.15108741792801;
	Mat img_float;
	resized_img.convertTo(img_float, CV_32F, 1.0/255); // Convert to float
	Mat normalized_img = (img_float - mean) / (std + 1e-7f); //normalizing 

	int IMG_HEIGHT = 32;
	int IMG_WIDTH = 32;
	int sz[4] = { 1, IMG_HEIGHT, IMG_WIDTH, 3 };
	Mat blob = Mat(4, sz, CV_32F, normalized_img.data); //convert to BHWC format
	//cout << "blob:" << blob.size << endl;

	string modelpath = "vgg16.onnx";
	Net net = readNetFromONNX(modelpath);
	net.setPreferableBackend(DNN_BACKEND_OPENCV);
	net.setPreferableTarget(DNN_TARGET_CPU);
	if (net.empty()) {
		cerr << "Failed to load the ONNX model!" << endl;
		return -1;
	}

	
	net.setInput(blob); //passing the inputs to model
	Mat outputs = net.forward();
	
	Point classIdPoint;
	double confidence;
	minMaxLoc(outputs, nullptr, &confidence, nullptr, &classIdPoint);
	int predictedClass = classIdPoint.x;

	string className = (predictedClass < classes.size()) ? classes[predictedClass] : "Unknown";
	cout << "Predicted Class: " << className << endl;
	cout << "Confidence: " << confidence << endl;
	cout << "Output probabilities: " << outputs << endl;

	for (int i = 0; i < classes.size(); ++i) {
		cout << "Class " << i << ": " << classes[i] << endl;
	}

	
	}
	
	catch (const std::exception& e) {
		std::cerr << "Error: " << e.what() << endl;
		return -1;
	}
	return 0;

}


	
	


