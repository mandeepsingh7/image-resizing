#include <opencv2/opencv.hpp>
#include <iostream>
#include <chrono>
#include <cmath>

using namespace cv;
using namespace std;
using namespace std::chrono;

enum InterpolationFlags {
	INTER_NEAREST_CUSTOM = 0,
	INTER_LINEAR_CUSTOM = 1,
	INTER_CUBIC_CUSTOM = 2,
};

float isConsistent(Mat& img_1, Mat& img_2, int tol) {
	if (img_1.size() != img_2.size()) {
		return false;
	}
	Mat diff;
	absdiff(img_1, img_2, diff);

	int count = 0;

	for (int y = 0; y < img_1.rows; y++) {
		for (int x = 0; x < img_1.cols; x++) {
			Vec3b pixels = diff.at<Vec3b>(y, x);
			for (int i = 0; i < 3; i++) {
				if (static_cast<int>(pixels[i]) > tol) {
					count++;
				}
			}
		}
	}
	float consistency = static_cast<float>(count) / (img_1.cols * img_1.rows * 3);
	return (100 - (consistency * 100));
}

Vec3b BiLinear(Mat& src, float x, float y) {
	if (x < 0.0) {
		x = 0.0;
	}
	if (y < 0.0) {
		y = 0.0;
	}

	int x1 = static_cast<int>(x);
	int y1 = static_cast<int>(y);
	int x2 = min(x1 + 1, src.cols - 1);
	int y2 = min(y1 + 1, src.rows - 1);

	Vec3b intensity_1 = src.at<Vec3b>(y1, x1);
	Vec3b intensity_2 = src.at<Vec3b>(y1, x2);
	Vec3b intensity_3 = src.at<Vec3b>(y2, x1);
	Vec3b intensity_4 = src.at<Vec3b>(y2, x2);

	float xAlpha = x - x1;
	float yAlpha = y - y1; 

	Vec3b intensity_new;

	for (int i = 0; i < 3; i++) {
		intensity_new[i] = intensity_1[i] * (1 - xAlpha) * (1 - yAlpha) +
					intensity_2[i] * xAlpha * (1 - yAlpha) +
					intensity_3[i] * (1 - xAlpha) * yAlpha +
					intensity_4[i] * xAlpha * yAlpha;
	}

	return intensity_new;
}


float Spline_Interpolate(float i1, float i2, float i3, float i4, float alpha) {
	return ((-i1 + 3 * i2 - 3 * i3 + i4) * pow(alpha, 3) / 6) +
		((i1 - 2 * i2 + i3) * pow(alpha, 2) / 2) +
		((-2 * i1 - 3 * i2 + 6 * i3 - i4) * alpha / 6) +
		i2;
}

Vec3b BiCubic(Mat& src, float x, float y) {
	
	int x1 = static_cast<int>(x);
	int y1 = static_cast<int>(y);

	int x0 = max(x1 - 1, 0);
	int x2 = min(x1 + 1, src.cols - 1);
	int x3 = min(x1 + 2, src.cols - 1);
	int y0 = max(y1 - 1, 0);
	int y2 = min(y1 + 1, src.rows - 1);
	int y3 = min(y1 + 2, src.rows - 1);

	//cout << "(x_src, y_src) = (" << x << "," << y << ")." << endl;
	//cout << "(x0, x1, x2, x3) = (" << x0 << ", " << x1 << ", " << x2 << ", " << x3 << ")." << endl;
	//cout << "(y0, y1, y2, y3) = (" << y0 << ", " << y1 << ", " << y2 << ", " << y3 << ")." << endl;

	float alphaX = x - x1;
	float alphaY = y - y1; 
	Vec3b intensity_new;
	for (int i = 0; i < 3; i++) {
		float intensity_0 = Spline_Interpolate(
			src.at<Vec3b>(y0, x0)[i],
			src.at<Vec3b>(y0, x1)[i],
			src.at<Vec3b>(y0, x2)[i],
			src.at<Vec3b>(y0, x2)[i],
			alphaX
		);

		float intensity_1 = Spline_Interpolate(
			src.at<Vec3b>(y1, x0)[i],
			src.at<Vec3b>(y1, x1)[i],
			src.at<Vec3b>(y1, x2)[i],
			src.at<Vec3b>(y1, x2)[i],
			alphaX
		);

		float intensity_2 = Spline_Interpolate(
			src.at<Vec3b>(y2, x0)[i],
			src.at<Vec3b>(y2, x1)[i],
			src.at<Vec3b>(y2, x2)[i],
			src.at<Vec3b>(y2, x2)[i],
			alphaX
		);

		float intensity_3 = Spline_Interpolate(
			src.at<Vec3b>(y3, x0)[i],
			src.at<Vec3b>(y3, x1)[i],
			src.at<Vec3b>(y3, x2)[i],
			src.at<Vec3b>(y3, x2)[i],
			alphaX
		);
		//cout << "Color Channel = " << i << endl;
		//cout << "(i0, i1, i2, i3) = (" << intensity_0 << ", " << intensity_1 << ", " << intensity_2 << ", " << intensity_3 << ")." << endl;
		float final_intensity = Spline_Interpolate(intensity_0, intensity_1, intensity_2, intensity_3, alphaY);
		if (final_intensity < 0.0) {
			intensity_new[i] = 0.0;
		}
		else if (final_intensity > 255.0) {
			intensity_new[i] = 255.0;
		}
		else {
			intensity_new[i] = final_intensity;
		}

		//cout << "Final Spline = " << final_intensity << endl;;
		//cout << "Final Intensity = " << static_cast<int>(intensity_new[i]) << endl;;
		//cout << "---------------------------------------------------" << endl;
	}
	return intensity_new;
}

void custom_resize(Mat& src, Mat& dst, Size dsize, double fx = 0.0, double fy = 0.0, int interpolation = INTER_NEAREST_CUSTOM) {

	if (dsize == Size()) {
		dsize = Size(saturate_cast<int>(fx * src.cols), saturate_cast<int>(fy * src.rows));
	}

	if (dsize != dst.size()) {
		dst.create(dsize, src.type());
	}

	float scaling_factor_x = ((static_cast<float>(src.cols)) / (dsize.width));
	float scaling_factor_y = ((static_cast<float>(src.rows)) / (dsize.height));

	for (int y = 0; y < dsize.height; y++) {
		for (int x = 0; x < dsize.width; x++) {
			float x_src = (x+0.5) * scaling_factor_x - 0.5;
			float y_src = (y+0.5) * scaling_factor_y - 0.5;

			//cout << "For (x, y) = (" << x << ", " << y << "), (x_src, y_src) = (" << x_src << ", " << y_src << "), Int Value = (" << static_cast<int>(floor(0.49999 + x_src)) << "," << static_cast<int>(floor(0.49999 + y_src)) << ")" << endl;
			//cout << "-------------------------------------------" << endl; 


			if (interpolation == INTER_NEAREST_CUSTOM) {
				dst.at<Vec3b>(y, x) = src.at<Vec3b>(static_cast<int>(floor(0.49999 + y_src)), static_cast<int>(floor(0.49999 + x_src)));
			}
			else if (interpolation == INTER_LINEAR_CUSTOM) {
				dst.at<Vec3b>(y, x) = BiLinear(src, x_src, y_src);
			}
			else if (interpolation == INTER_CUBIC_CUSTOM) {
				dst.at<Vec3b>(y, x) = BiCubic(src, x_src, y_src);
			}
		}
	}
}


void main() {
	string path = "G178_2 -1080.BMP";
	Mat img = imread(path);
	cout << "Size of Original Image = " << img.size() << endl;
	int height = img.rows;
	int width = img.cols;
	Size new_size(width / 2, height / 2);
	cout << "Size of Resized Image = " << new_size << endl;
	cout << endl;
	//-----------------------------------
	// Step 1
	//-----------------------------------

	Mat resize_nearest, resize_linear, resize_cubic;
	resize(img, resize_nearest, new_size, 0, 0, INTER_NEAREST);
	imwrite("Resize_Nearest_OpenCV.bmp", resize_nearest);
	resize(img, resize_linear, new_size, 0, 0, INTER_LINEAR);
	imwrite("Resize_Linear_OpenCV.bmp", resize_linear);
	resize(img, resize_cubic, new_size, 0, 0, INTER_CUBIC);
	imwrite("Resize_Cubic_OpenCV.bmp", resize_cubic);
	//resize(img, resize_cubic_new, Size(), 0.5, 0.5, INTER_CUBIC);

	cout << "-------------------------------------------------------------------------------------------" << endl;

	//-----------------------------------
	// Step 2 
	//-----------------------------------
	int iteration_count = 10;
	vector<float> built_in_times;

	// Nearest Neighbours Interpolation
	auto start = high_resolution_clock::now();
	for (int i = 0; i < iteration_count; i++) {
		resize(img, resize_nearest, new_size, 0, 0, INTER_NEAREST);
	}
	auto stop = high_resolution_clock::now();
	auto duration = duration_cast<milliseconds>(stop - start).count();
	built_in_times.push_back(duration);
	cout << "Time taken for " << iteration_count << " iterations using built-in resize function (INTER_NEAREST) = " << duration << " ms." << endl;

	// BiLinear Interpolation
	start = high_resolution_clock::now();
	for (int i = 0; i < iteration_count; i++) {
		resize(img, resize_linear, new_size, 0, 0, INTER_LINEAR);
	}
	stop = high_resolution_clock::now();
	duration = duration_cast<milliseconds>(stop - start).count();
	built_in_times.push_back(duration);
	cout << "Time taken for " << iteration_count << " iterations using built-in resize function (INTER_LINEAR) = " << duration << " ms." << endl;

	// BiCubic Interpolation
	start = high_resolution_clock::now();
	for (int i = 0; i < iteration_count; i++) {
		resize(img, resize_cubic, new_size, 0, 0, INTER_CUBIC);
	}
	stop = high_resolution_clock::now();
	duration = duration_cast<milliseconds>(stop - start).count();
	built_in_times.push_back(duration);
	cout << "Time taken for " << iteration_count << " iterations using built-in resize function (INTER_CUBIC) = " << duration << " ms." << endl;
	cout << "-------------------------------------------------------------------------------------------" << endl;
	cout << endl;

	//-----------------------------------
	// Step 3
	//-----------------------------------

	Mat resize_nearest_custom, resize_linear_custom, resize_cubic_custom;
	custom_resize(img, resize_nearest_custom, new_size, 0, 0, INTER_NEAREST_CUSTOM);
	custom_resize(img, resize_linear_custom, new_size, 0, 0, INTER_LINEAR_CUSTOM);
	custom_resize(img, resize_cubic_custom, new_size, 0, 0, INTER_CUBIC_CUSTOM);
	cout << "Consistency of custom resize function (INTER_NEAREST) (tolerance = 0) = " << isConsistent(resize_nearest, resize_nearest_custom, 0) << "%." << endl;
	cout << "-----------------------------------------------------------------------------" << endl;
	cout << endl; 
	cout << "----------------------------------------------------------------------------------------------------------------------" << endl;
	cout << "Tolerance is the maximum difference between pixel intensity of the image generated from custom and built-in functions." << endl;
	cout << "----------------------------------------------------------------------------------------------------------------------" << endl;
	cout << endl ;
	cout << "Consistency of custom resize function (INTER_LINEAR):" << endl;
	cout << "-----------------------------------------------------" << endl;
	for (int i = 0; i < 6; i++) {
		cout << "Consistency (tolerance = " << i << ") = " << fixed << setprecision(2) << isConsistent(resize_linear, resize_linear_custom, i) << "%." << endl;
	}
	cout << endl;
	cout << "Consistency of custom resize function (INTER_CUBIC):" << endl;
	cout << "----------------------------------------------------" << endl;
	for (int i = 0; i < 31; i = i + 5) {
		cout << "Consistency (tolerance = " << i << ") = " << fixed << setprecision(2) << isConsistent(resize_cubic, resize_cubic_custom, i) << "%." << endl;
	}
	cout << endl;
	cout << "-----------------------------------------------------------------------------------------" << endl;
	vector<float> custom_times;

	// Nearest Neighbours Interpolation with Custom Function 
	start = high_resolution_clock::now();
	for (int i = 0; i < iteration_count; i++) {
		custom_resize(img, resize_nearest_custom, new_size, 0, 0, INTER_NEAREST_CUSTOM);
	}
	stop = high_resolution_clock::now();
	duration = duration_cast<milliseconds>(stop - start).count();
	custom_times.push_back(duration);
	cout << "Time taken for " << iteration_count << " iterations using custom resize function (INTER_NEAREST_CUSTOM) = " << duration << " ms." << endl;

	// BiLinear Interpolation with Custom Function 
	start = high_resolution_clock::now();
	for (int i = 0; i < iteration_count; i++) {
		custom_resize(img, resize_linear_custom, new_size, 0, 0, INTER_LINEAR_CUSTOM);
	}
	stop = high_resolution_clock::now();
	duration = duration_cast<milliseconds>(stop - start).count();
	custom_times.push_back(duration);
	cout << "Time taken for " << iteration_count << " iterations using custom resize function (INTER_LINEAR_CUSTOM) = " << duration << " ms." << endl;

	// BiCubic Interpolation with Custom Function 
	start = high_resolution_clock::now();
	for (int i = 0; i < iteration_count; i++) {
		custom_resize(img, resize_cubic_custom, new_size, 0, 0, INTER_CUBIC_CUSTOM);
	}
	stop = high_resolution_clock::now();
	duration = duration_cast<milliseconds>(stop - start).count();
	custom_times.push_back(duration);
	cout << "Time taken for " << iteration_count << " iterations using custom resize function (INTER_CUBIC_CUSTOM) = " << duration << " ms." << endl;
	cout << "-----------------------------------------------------------------------------------------" << endl;
	cout << endl; 

	vector<string> interpolation_methods = { "INTER_NEAREST", "INTER_LINEAR", "INTER_CUBIC" };

	cout << "Comparing time taken by built-in and custom function for " << iteration_count << " iterations:" << endl;

	cout << setw(20) << "--------------------" << setw(20) << "------------------" << setw(20) << "----------------" << endl;
	cout << setw(20) << "Interpolation Method" << setw(20) << "Built-In Times(ms)" << setw(20) << "Custom Times(ms)" << endl;
	cout << setw(20) << "--------------------" << setw(20) << "------------------" << setw(20) << "----------------" << endl;
	for (int i = 0; i < 3; i++) {
		cout << setw(20) << interpolation_methods[i] << setw(20) << built_in_times[i] << setw(20) << custom_times[i] << endl;
	}
}